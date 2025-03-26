import numpy as np

def reconstruct_depth_from_sonar_byprior(sonar_image, range_axis, N):
    depth_image = np.full(N, -1)
    
    values = []
    for count, value in zip(sonar_image, range_axis):
        values.extend([value] * count)
    
    # 排序并反转，确保从大到小
    values.sort(reverse=True)
    
    # 填充结果数组
    depth_image[N-len(values):] = values
    
    return depth_image

# 测试代码
sonar_image = np.array([0, 1, 2, 2, 1])
range_axis = np.array([0, 1, 2, 3, 4])
N = 10

result = reconstruct_depth_from_sonar_byprior(sonar_image, range_axis, N)
print(result)  # [-1 -1 -1 -1 -1 -1 -1  2  2  1]