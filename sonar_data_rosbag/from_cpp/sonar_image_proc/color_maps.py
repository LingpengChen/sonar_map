import numpy as np
import cv2
import matplotlib.pyplot as plt
from .ColorMaps import inferno_data_float, inferno_data_uint8
from .sonar_image_msg_interface import AbstractSonarInterface, SonarImageMsgInterface, AzimuthRangeIndices
# from sonar_image_proc.ColorMaps import inferno_data_float, inferno_data_uint8


# 从ColorMaps.cpp导入的Inferno色彩映射数据
# 这里只展示前几个和最后几个数据点，完整数据请从文件中复制

# 将上面的列表转换为NumPy数组
inferno_data_float = np.array(inferno_data_float, dtype=np.float32)
inferno_data_uint8 = np.array(inferno_data_uint8, dtype=np.uint8)


class SonarColorMap:
    """声纳色彩映射基类，定义了将声纳数据映射到不同颜色格式的接口"""
    def lookup_cv32fc1(self, ping:SonarImageMsgInterface, loc:AzimuthRangeIndices):
        """返回单通道浮点值（灰度）"""
        return ping.intensity_float(loc)
    
    def lookup_cv8uc3(self, ping, loc):
        """返回3通道8位BGR值（OpenCV默认格式）"""
        f = self.lookup_cv32fc1(ping, loc)
        return np.array([f * 255, f * 255, f * 255], dtype=np.uint8)
    
    def lookup_cv32fc3(self, ping, loc):
        """返回3通道浮点BGR值"""
        f = self.lookup_cv32fc1(ping, loc)
        return np.array([f, f, f], dtype=np.float32)

class MitchellColorMap(SonarColorMap):
    """Mitchell色彩映射，一种简单的自定义映射"""
    def lookup_cv8uc3(self, ping, loc):
        i = ping.intensity_float(loc)
        # 注意OpenCV使用BGR顺序，所以这里是[蓝,绿,红]
        return np.array([i * 255, i * 255, (1 - i) * 255], dtype=np.uint8)

class InfernoColorMap(SonarColorMap):
    """Inferno色彩映射，使用预定义的色彩映射表"""
    def lookup_cv8uc3(self, ping, loc):
        i = ping.intensity_uint8(loc)
        # 注意OpenCV使用BGR顺序，所以我们需要反转RGB->BGR
        return np.array([
            inferno_data_uint8[i][2],  # B
            inferno_data_uint8[i][1],  # G
            inferno_data_uint8[i][0]   # R
        ], dtype=np.uint8)
    
    def lookup_cv32fc3(self, ping, loc):
        i = ping.intensity_uint8(loc)
        # 同样，需要反转RGB->BGR
        return np.array([
            inferno_data_float[i][2],  # B
            inferno_data_float[i][1],  # G
            inferno_data_float[i][0]   # R
        ], dtype=np.float32)

class InfernoSaturationColorMap(InfernoColorMap):
    """带饱和度标记的Inferno色彩映射，将最大值（255）标记为绿色"""
    def lookup_cv8uc3(self, ping, loc):
        i = ping.intensity_uint8(loc)
        if i == 255:
            return np.array([0, 255, 0], dtype=np.uint8)  # BGR格式的绿色
        else:
            return super().lookup_cv8uc3(ping, loc)
    
    def lookup_cv32fc3(self, ping, loc):
        i = ping.intensity_uint8(loc)
        if i == 255:
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)  # BGR格式的绿色
        else:
            return super().lookup_cv32fc3(ping, loc)

# 示例实现类
class SampleSonarData(AbstractSonarInterface):
    """示例声纳数据类，用于演示"""
    def __init__(self, data):
        """
        初始化示例声纳数据
        
        参数:
            data: 二维NumPy数组，代表声纳强度数据
        """
        self.data = data
    
    def intensity_float(self, loc:AzimuthRangeIndices):
        """返回归一化的浮点强度值"""
        return self.data[loc.azimuth(), loc.range_idx()] / 255.0
    
    def intensity_uint8(self, loc):
        """返回原始的整数强度值"""
        return self.data[loc.azimuth(), loc.range_idx()]

def apply_colormap(sonar_data:SonarImageMsgInterface, colormap: SonarColorMap, height, width):
    """将色彩映射应用到声纳数据上生成彩色图像
    
    参数:
        sonar_data: AbstractSonarInterface的实例
        colormap: SonarColorMap的实例
        height: 输出图像高度
        width: 输出图像宽度
        
    返回:
        彩色图像的NumPy数组，形状为(height, width, 3)
    """
    result = np.zeros((height, width, 3), dtype=np.uint8)
    
    for y in range(height):
        for x in range(width):
            loc = AzimuthRangeIndices(y, x)
            result[y, x] = colormap.lookup_cv8uc3(sonar_data, loc)
    
    return result

# 测试代码
def generate_test_image():
    """生成测试图像并应用不同的色彩映射"""
    # 创建一个示例声纳数据（渐变测试图案）
    height, width = 200, 256
    data = np.zeros((height, width), dtype=np.uint8)
    
    # 水平渐变
    for x in range(width):
        data[:, x] = x
    
    # 创建声纳数据对象
    sonar = SampleSonarData(data)
    
    # 应用不同的色彩映射
    gray_image = apply_colormap(sonar, SonarColorMap(), height, width)
    mitchell_image = apply_colormap(sonar, MitchellColorMap(), height, width)
    inferno_image = apply_colormap(sonar, InfernoColorMap(), height, width)
    inferno_sat_image = apply_colormap(sonar, InfernoSaturationColorMap(), height, width)
    
    # 显示结果
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.title("灰度映射")
    plt.imshow(cv2.cvtColor(gray_image, cv2.COLOR_BGR2RGB))
    
    plt.subplot(2, 2, 2)
    plt.title("Mitchell映射")
    plt.imshow(cv2.cvtColor(mitchell_image, cv2.COLOR_BGR2RGB))
    
    plt.subplot(2, 2, 3)
    plt.title("Inferno映射")
    plt.imshow(cv2.cvtColor(inferno_image, cv2.COLOR_BGR2RGB))
    
    plt.subplot(2, 2, 4)
    plt.title("Inferno映射（带饱和标记）")
    plt.imshow(cv2.cvtColor(inferno_sat_image, cv2.COLOR_BGR2RGB))
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    generate_test_image()