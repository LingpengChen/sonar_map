#!/usr/bin/env python3

import numpy as np

class SonarImageMsgInterface:
    """声纳图像消息接口，用于包装ROS消息"""
    
    def __init__(self, msg):
        """
        初始化接口
        
        参数:
            msg: ProjectedSonarImage消息
        """
        self.msg = msg
        
        # 提取基本信息
        self.bearings = np.array([b for b in msg.bearings])
        self.ranges = np.array([r for r in msg.ranges])
        
        # 提取强度数据
        width = len(self.bearings)
        height = len(self.ranges)
        self.intensities = np.array(msg.image.data).reshape(height, width)
        
        # 归一化强度数据到0-1范围
        if self.intensities.max() > 0:
            self.intensities = self.intensities / self.intensities.max()
    
    def do_log_scale(self, min_db, max_db):
        """
        应用对数刻度
        
        参数:
            min_db: 最小dB值
            max_db: 最大dB值
        """
        # 避免对数运算中的0值
        mask = self.intensities > 0
        log_intensities = np.zeros_like(self.intensities)
        
        # 计算对数值
        if np.any(mask):
            log_intensities[mask] = 20 * np.log10(self.intensities[mask])
            
            # 自动计算最小最大值（如果参数为0）
            if min_db == 0 and max_db == 0:
                min_db = np.min(log_intensities[mask])
                max_db = np.max(log_intensities[mask])
            
            # 归一化
            log_intensities = (log_intensities - min_db) / (max_db - min_db)
            log_intensities = np.clip(log_intensities, 0, 1)
        
        self.intensities = log_intensities