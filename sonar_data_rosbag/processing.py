#!/usr/bin/env python
# coding:utf-8

# ~/ros_workingspace/liboculus 

import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

# 初始化全局变量
previous_frame = None
bridge = CvBridge()

def callback(data):
    global previous_frame, bridge

    # 将ROS图像消息转换为OpenCV图像
    current_frame = bridge.imgmsg_to_cv2(data, "bgr8")

    if previous_frame is not None:
        # 初始化AKAZE检测器
        #akaze = cv2.AKAZE_create()
        #akaze = cv2.AKAZE_create(descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB, threshold=0.001, nOctaves=4, nOctaveLayers=4)
        akaze = cv2.AKAZE_create(descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB_UPRIGHT, threshold=0.001, nOctaves=4, nOctaveLayers=4)


        # 检测特征点和计算描述符
        kp1, des1 = akaze.detectAndCompute(previous_frame, None)
        kp2, des2 = akaze.detectAndCompute(current_frame, None)

        # 使用BFMatcher进行特征匹配
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        for match in matches:
            pt1 = kp1[match.queryIdx].pt
            pt2 = kp2[match.trainIdx].pt
            # 这里假设你有一个函数 get_predicted_position 来获取特征点在下一帧中的预测位置
            predicted_pt2 = pt1
            position_distance = np.linalg.norm(np.array(predicted_pt2) - np.array(pt2))
            descriptor_distance = match.distance
            match.distance = position_distance
    # 筛选距离较小的匹配结果
        matches = [m for m in matches if m.distance < 10]

        # 按照距离排序匹配结果
        matches = sorted(matches, key=lambda x: x.distance)


        print(f"匹配点数量: {len(matches)}")

        # 绘制匹配结果
        matched_img = cv2.drawMatches(previous_frame, kp1, current_frame, kp2, matches[:30], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)



        # 显示匹配结果
        cv2.imshow("Matched Frames", matched_img)
        cv2.waitKey(1)

    # 更新前一帧图像
    previous_frame = current_frame

if __name__ == '__main__':
    rospy.init_node('img_process_node', anonymous=True)
    rospy.Subscriber('/oculus/drawn_sonar', Image, callback)
    rospy.spin()