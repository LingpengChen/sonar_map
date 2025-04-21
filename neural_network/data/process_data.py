import numpy as np
import glob
import os

# 处理所有三种类型的文件
file_types = ['obs_vectors', 'prior_matrices', 'true_matrices']
raw_data_dir = '/home/clp/catkin_ws/src/sonar_map/neural_network/data/raw_data'
dataset_dir = '/home/clp/catkin_ws/src/sonar_map/neural_network/data/dataset'
for file_type in file_types:
    # 获取所有符合模式的文件
    file_pattern = f'{file_type}_*.npy'
    matrix_files = glob.glob(os.path.join(raw_data_dir, file_pattern))
    
    # 用于存储所有加载的矩阵
    all_matrices = []
    
    # 加载每个文件并存储
    for file_name in matrix_files:
        matrices = np.load(file_name)
        all_matrices.append(matrices)
    
    # 沿着axis=0拼接所有矩阵
    combined_matrices = np.concatenate(all_matrices, axis=0)
    
    # 保存合并后的矩阵
    np.save(os.path.join(dataset_dir, f'{file_type}.npy'), combined_matrices)