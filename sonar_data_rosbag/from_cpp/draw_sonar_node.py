#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from marine_acoustic_msgs.msg import ProjectedSonarImage
from sensor_msgs.msg import Image
from std_msgs.msg import String, UInt32MultiArray
from dynamic_reconfigure.server import Server
import time
import json,os,yaml

from sonar_image_proc.color_maps import SonarColorMap, MitchellColorMap, InfernoColorMap, InfernoSaturationColorMap
from sonar_image_proc.sonar_drawer import SonarDrawer
from sonar_image_proc.sonar_image_msg_interface import SonarImageMsgInterface
# from sonar_image_proc.histogram_generator import HistogramGenerator

class DrawSonarNode:
    def __init__(self):
        # 从YAML文件读取配置
        config_path = '/home/clp/catkin_ws/src/sonar_map/sonar_data_rosbag/from_cpp/config.yaml'
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        # config = self.load_yaml_config('config.yaml')
        
        # 设置参数
        self._max_range = config.get('max_range', 0.0)         # Maximum range to display (0 for unlimited)
        self._publish_timing = config.get('publish_timing', True)   # Publish timing information
        self._publish_histogram = config.get('publish_histogram', False) # Publish histogram data
        
        color_map_name = config.get('color_map', "inferno")   # Color map to use
        
        # 设置颜色映射
        self._color_map = self.set_color_map(color_map_name)
        
        # 创建声纳绘制器
        self._sonar_drawer = SonarDrawer()
        
        
        # 日志参数
        self.log_scale = config.get('log_scale', False)
        self.min_db = config.get('min_db', 0.0)
        self.max_db = config.get('max_db', 0.0)
        
        # 输出信息
        if self._max_range > 0.0:
            rospy.loginfo(f"Only drawing to max range {self._max_range}")
        
        # 创建订阅者
        self.sub_sonar_image = rospy.Subscriber(
            '/oculus/sonar_image', ProjectedSonarImage, self.sonar_image_callback, queue_size=10)
        
        # 创建发布者
        self.pub = rospy.Publisher('drawn_sonar', Image, queue_size=10)
        self.osd_pub = rospy.Publisher('drawn_sonar_osd', Image, queue_size=10)
        self.rect_pub = rospy.Publisher('drawn_sonar_rect', Image, queue_size=10)
        
        if self._publish_timing:
            self.timing_pub = rospy.Publisher('sonar_image_proc_timing', String, queue_size=10)
        
        if self._publish_histogram:
            self.histogram_pub = rospy.Publisher('histogram', UInt32MultiArray, queue_size=10)
        
        # CV桥接器
        self.bridge = CvBridge()
        
        rospy.logdebug("draw_sonar ready to run...")
    
    def set_color_map(self, color_map_name):
        # 根据配置返回相应的颜色映射
        return SonarColorMap()
        # return MitchellColorMap()
        # return InfernoColorMap()
        # return InfernoSaturationColorMap()
    
    def cv_bridge_and_publish(self, msg, mat, pub):
        # 转换OpenCV图像到ROS消息并发布
        img_msg = self.bridge.cv2_to_imgmsg(mat, encoding="rgb8")
        img_msg.header = msg.header
        pub.publish(img_msg)
    
    def sonar_image_callback(self, msg):
        if not self._color_map:
            rospy.logfatal("Colormap is undefined, this shouldn't happen")
            return
        
        # 创建消息接口
        interface = SonarImageMsgInterface(msg)
        if self.log_scale:
            interface.do_log_scale(self.min_db, self.max_db)
        
        # 初始化时间统计
        old_api_elapsed = rect_elapsed = map_elapsed = histogram_elapsed = 0
        
      
        
        # # 处理直方图
        # if self._publish_histogram:
        #     start_time = time.time()
            
        #     histogram_out = UInt32MultiArray()
        #     histogram_out.data = HistogramGenerator.generate(interface)
            
        #     self.histogram_pub.publish(histogram_out)
            
        #     histogram_elapsed = time.time() - start_time
        
        # 绘制矩形声纳图像
        start_time = time.time()
        rect_mat = self._sonar_drawer.drawRectSonarImage(interface, self._color_map)
        
        # 旋转矩形图像90度逆时针
        rotated_rect = cv2.rotate(rect_mat, cv2.ROTATE_90_COUNTERCLOCKWISE)
        self.cv_bridge_and_publish(msg, rotated_rect, self.rect_pub)
        
        rect_elapsed = time.time() - start_time
        
        start_time = time.time()
        # 重映射矩形声纳图像
        sonar_mat = self._sonar_drawer.remapRectSonarImage(interface, rect_mat)
        self.cv_bridge_and_publish(msg, sonar_mat, self.pub)
        map_elapsed = time.time() - start_time
        
        # 发布时间统计信息
        if self._publish_timing:
            output = {
                "draw_total": rect_elapsed + map_elapsed,
                "rect": rect_elapsed,
                "map": map_elapsed
            }
            
       
            if self._publish_histogram:
                output["histogram"] = histogram_elapsed
            
            out_msg = String()
            out_msg.data = json.dumps(output)
            
            self.timing_pub.publish(out_msg)
    
  
    
    def calculate_image_size(self, interface, pix_per_range_bin):
        # 简化版的旧API中的calculateImageSize函数
        # 返回一个(width, height)的元组
        # 这里实现简化版的计算逻辑
        return (800, 600)  # 示例尺寸，实际应基于声纳数据计算
    
    def draw_sonar_old_api(self, interface, mat):
        # 简化版的旧API中的drawSonar函数
        # 这里应该实现基于界面数据和颜色映射的绘制逻辑
        # 返回处理后的图像
        return mat  # 示例返回，实际应返回处理后的图像

if __name__ == '__main__':
    rospy.init_node('draw_sonar_node')
    node = DrawSonarNode()
    rospy.spin()
    # DTYPE_UINT8:
