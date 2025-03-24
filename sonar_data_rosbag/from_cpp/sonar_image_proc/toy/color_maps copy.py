#!/usr/bin/env python3

import numpy as np
import cv2

class SonarColorMap:
    """基础声纳颜色映射类"""
    
    def apply(self, intensity, out_rgb):
        """
        将强度值映射到RGB颜色
        
        参数:
            intensity: 强度值 (0.0-1.0)
            out_rgb: 输出RGB数组，应当是长度为3的数组或者3通道numpy数组
        """
        raise NotImplementedError("Subclasses must implement apply()")

class InfernoColorMap(SonarColorMap):
    """Inferno颜色映射实现"""
    
    def __init__(self):
        # 初始化颜色映射表
        self.colormap = cv2.applyColorMap(
            np.arange(0, 256, dtype=np.uint8), 
            cv2.COLORMAP_INFERNO
        )
    
    def apply(self, intensity, out_rgb):
        # 将强度映射到颜色
        idx = int(np.clip(intensity * 255, 0, 255))
        color = self.colormap[0, idx]
        out_rgb[0] = color[2]  # R
        out_rgb[1] = color[1]  # G
        out_rgb[2] = color[0]  # B

class InfernoSaturationColorMap(InfernoColorMap):
    """带饱和度的Inferno颜色映射"""
    
    def apply(self, intensity, out_rgb):
        super().apply(intensity, out_rgb)
        # 增加饱和度处理
        # 这里可以添加额外的饱和度处理逻辑