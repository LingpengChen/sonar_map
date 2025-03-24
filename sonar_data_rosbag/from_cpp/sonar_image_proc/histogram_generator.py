#!/usr/bin/env python3

import numpy as np

class HistogramGenerator:
    """声纳图像直方图生成器"""
    
    @staticmethod
    def generate(interface, bins=256):
        """
        生成强度直方图
        
        参数:
            interface: 声纳图像接口
            bins: 直方图的箱数
        
        返回:
            直方图数据（bin计数）
        """
        # 生成直方图
        hist, _ = np.histogram(interface.intensities.flatten(), bins=bins, range=(0, 1))
        return hist.tolist()