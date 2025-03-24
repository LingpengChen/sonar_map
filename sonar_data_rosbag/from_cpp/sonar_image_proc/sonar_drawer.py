import numpy as np
import cv2
import math
from typing import Tuple, Optional, Dict, Any
from .color_maps import SonarColorMap
from .sonar_image_msg_interface import AbstractSonarInterface, SonarImageMsgInterface, AzimuthRangeIndices



class SonarDrawer:
    """声纳绘制器，复现原C++代码中的SonarDrawer类"""
    
    def __init__(self):
        """初始化声纳绘制器"""
        self._cached_map = None
        self._cached_ping = None
    
    def drawRectSonarImage(self, ping: SonarImageMsgInterface, 
                           color_map: SonarColorMap,
                           rect_in: Optional[np.ndarray] = None) -> np.ndarray:
        """
        绘制矩形声纳图像
        
        Args:
            ping: 声纳接口
            color_map: 颜色映射
            rect_in: 输入图像，可选
        
        Returns:
            矩形声纳图像
        """
        # 确定图像大小
        img_size = (ping.nRanges(), ping.nBearings())
        
        # 创建输出图像
        if rect_in is None or rect_in.shape[0] == 0 or rect_in.shape[1] == 0:
            rect = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
        else:
            rect = rect_in.copy()
            rect.resize((img_size[1], img_size[0], 3))
        
        # 遍历像素并着色
        for r in range(ping.nRanges()):
            for b in range(ping.nBearings()):
                loc_idx = AzimuthRangeIndices(b, r)
                rect[b, r] = color_map.lookup_cv8uc3(ping, loc_idx)
        
        return rect
    
    def _calculate_maps(self, ping: SonarImageMsgInterface) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算重映射的映射矩阵
        
        Args:
            ping: 声纳接口
            
        Returns:
            映射矩阵对
        """
        # 检查缓存
        if (self._cached_map is not None and self._cached_ping is not None and
            self._cached_ping.nRanges() == ping.nRanges() and
            self._cached_ping.nBearings() == ping.nBearings() and
            self._cached_ping.rangeBounds() == ping.rangeBounds() and
            self._cached_ping.azimuthBounds() == ping.azimuthBounds()):
            return self._cached_map
        
        # 计算新的映射
        n_ranges = ping.nRanges()
        azimuth_bounds = ping.azimuthBounds()
        
        minus_width = math.floor(n_ranges * math.sin(azimuth_bounds[0]))
        plus_width = math.ceil(n_ranges * math.sin(azimuth_bounds[1]))
        width = plus_width - minus_width
        
        origin_x = abs(minus_width)
        
        img_size = (width, n_ranges) # 354 352
        if width <= 0 or n_ranges <= 0:
            return None
        
        # 创建映射矩阵
        map_x = np.zeros(img_size, dtype=np.float32)
        map_y = np.zeros(img_size, dtype=np.float32)
        
        db = (azimuth_bounds[1] - azimuth_bounds[0]) / ping.nAzimuths()  # 2*0.5235987833701112 / 256
        
        for x in range(img_size[0]):
            for y in range(img_size[1]):
                dx = x - origin_x
                dy = img_size[1] - y
                
                range_val = math.sqrt(dx * dx + dy * dy)
                azimuth = math.atan2(dx, dy)
                
                xp = range_val
                # yp = (azimuth - azimuth_bounds[0]) / db
                yp = (azimuth_bounds[1] - azimuth) / db
                
                map_x[x, y] = xp
                map_y[x, y] = yp
        
        # 保存到缓存
        self._cached_map = (map_x, map_y)
        self._cached_ping = ping
        
        return (map_x, map_y)
    
    def remapRectSonarImage(self, ping: AbstractSonarInterface, 
                           rect_image: np.ndarray) -> np.ndarray:
        """
        将矩形声纳图像重映射为扇形图像
        
        Args:
            ping: 声纳接口
            rect_image: 矩形声纳图像
            
        Returns:
            扇形声纳图像
        """
        # 获取映射矩阵
        maps = self._calculate_maps(ping)
        if maps is None:
            return rect_image
        
        map_x, map_y = maps
        
        # 执行重映射
        dst = cv2.remap(rect_image, map_x, map_y, cv2.INTER_CUBIC, 
                       borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        
        return dst
    
    def drawSonar(self, ping: AbstractSonarInterface,
                 color_map: SonarColorMap,
                 img: Optional[np.ndarray] = None,
                 add_overlay: bool = False) -> np.ndarray:
        """
        绘制声纳图像
        
        Args:
            ping: 声纳接口
            color_map: 颜色映射
            img: 输入图像，可选
            add_overlay: 是否添加覆盖层，默认False
            
        Returns:
            声纳图像
        """
        # 绘制矩形图像
        rect = self.drawRectSonarImage(ping, color_map, img)
        
        # 重映射为扇形图像
        sonar = self.remapRectSonarImage(ping, rect)
        
        # 这里忽略overlay功能，因为你说它不重要
        return sonar

