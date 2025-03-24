#!/usr/bin/env python3

import cv2
import numpy as np
from sonar_image_proc.sonar_image_msg_interface import SonarImageMsgInterface

class OverlayConfig:
    """叠加层配置类"""
    
    def __init__(self):
        self.range_spacing = 0.0
        self.radial_spacing = 20.0
        self.line_alpha = 0.5
        self.line_thickness = 1
    
    def set_range_spacing(self, value):
        self.range_spacing = value
        return self
    
    def set_radial_spacing(self, value):
        self.radial_spacing = value
        return self
    
    def set_line_alpha(self, value):
        self.line_alpha = value
        return self
    
    def set_line_thickness(self, value):
        self.line_thickness = value
        return self

class SonarDrawer:
    """声纳图像绘制类"""
    
    def __init__(self):
        self.overlay_config = OverlayConfig()
    
    def draw_rect_sonar_image(self, interface: SonarImageMsgInterface, color_map):
        """
        绘制矩形声纳图像
        
        参数:
            interface: 声纳图像接口
            color_map: 颜色映射对象
        
        返回:
            矩形声纳图像（OpenCV Mat）
        """
        # 从接口获取声纳数据
        bearings = interface.bearings
        ranges = interface.ranges
        intensities = interface.intensities
        
        # 创建矩形图像
        height = len(ranges)
        width = len(bearings)
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 绘制声纳数据
        for i in range(height):
            for j in range(width):
                intensity = intensities[i, j]
                rgb = [0, 0, 0]
                color_map.apply(intensity, rgb)
                img[i, j] = rgb
        
        return img
    
    def remap_rect_sonar_image(self, interface, rect_img):
        """
        重新映射矩形声纳图像到极坐标形式
        
        参数:
            interface: 声纳图像接口
            rect_img: 矩形声纳图像
        
        返回:
            极坐标形式的声纳图像
        """
        # 获取声纳参数
        bearings = interface.bearings
        ranges = interface.ranges
        
        min_bearing = bearings[0]
        max_bearing = bearings[-1]
        min_range = ranges[0]
        max_range = ranges[-1]
        
        # 创建输出图像
        output_size = 800  # 可根据需要调整
        output = np.zeros((output_size, output_size, 3), dtype=np.uint8)
        
        # 计算极坐标映射
        center_x = output_size // 2
        center_y = output_size // 2
        scale = output_size / (2 * max_range)
        
        # 应用映射
        for y in range(output_size):
            for x in range(output_size):
                # 转换到极坐标
                dx = x - center_x
                dy = y - center_y
                r = np.sqrt(dx*dx + dy*dy) / scale
                theta = np.arctan2(dy, dx)
                
                # 转换到bearing角度
                bearing = np.degrees(theta)
                
                # 检查是否在范围内
                if min_range <= r <= max_range and min_bearing <= bearing <= max_bearing:
                    # 线性插值获取rect_img中的值
                    r_idx = (r - min_range) / (max_range - min_range) * (len(ranges) - 1)
                    b_idx = (bearing - min_bearing) / (max_bearing - min_bearing) * (len(bearings) - 1)
                    
                    r_idx = int(np.clip(r_idx, 0, len(ranges) - 1))
                    b_idx = int(np.clip(b_idx, 0, len(bearings) - 1))
                    
                    output[y, x] = rect_img[r_idx, b_idx]
        
        return output
    
    def draw_overlay(self, interface, sonar_img):
        """
        在声纳图像上绘制叠加层
        
        参数:
            interface: 声纳图像接口
            sonar_img: 声纳图像
        
        返回:
            带叠加层的声纳图像
        """
        # 创建叠加层图像
        overlay = sonar_img.copy()
        
        # 获取图像尺寸
        height, width = sonar_img.shape[:2]
        center_x = width // 2
        center_y = height // 2
        
        # 绘制距离环
        max_range = interface.ranges[-1]
        if self.overlay_config.range_spacing > 0:
            spacing = self.overlay_config.range_spacing
        else:
            # 自动计算合适的距离间隔
            spacing = max_range / 5
        
        # 绘制距离环
        for r in np.arange(spacing, max_range + spacing, spacing):
            radius = int(r / max_range * min(center_x, center_y))
            cv2.circle(overlay, (center_x, center_y), radius, (255, 255, 255), 
                      self.overlay_config.line_thickness)
            
            # 添加距离标签
            label = f"{r:.1f}m"
            cv2.putText(overlay, label, (center_x + radius, center_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 绘制方位角线
        if self.overlay_config.radial_spacing > 0:
            angle_spacing = self.overlay_config.radial_spacing
            max_radius = min(center_x, center_y)
            
            for angle in np.arange(0, 360, angle_spacing):
                rad = np.radians(angle)
                end_x = center_x + int(max_radius * np.cos(rad))
                end_y = center_y + int(max_radius * np.sin(rad))
                
                cv2.line(overlay, (center_x, center_y), (end_x, end_y), 
                        (255, 255, 255), self.overlay_config.line_thickness)
                
                # 添加角度标签
                label = f"{angle}°"
                label_x = center_x + int(max_radius * 1.1 * np.cos(rad))
                label_y = center_y + int(max_radius * 1.1 * np.sin(rad))
                cv2.putText(overlay, label, (label_x, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 使用alpha混合
        alpha = self.overlay_config.line_alpha
        result = cv2.addWeighted(overlay, alpha, sonar_img, 1 - alpha, 0)
        
        return result