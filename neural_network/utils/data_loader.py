"""加载或生成概率矩阵训练数据"""
import torch
import numpy as np
import os
from torch.utils.data import TensorDataset, DataLoader

class ProbabilityDataLoader:
    def __init__(self, M, N, batch_size=64, data_dir='./data/dataset/'):
        """
        初始化数据加载器
        Args:
            M: 矩阵行数
            N: 矩阵列数
            batch_size: 批处理大小
            data_dir: 数据目录
        """
        self.M = M
        self.N = N
        self.batch_size = batch_size
        self.data_dir = data_dir
        
    def load_data(self):
        """
        从文件加载数据
        Returns:
            prior_matrices, obs_vectors, true_matrices
        """
        # 检查数据文件是否存在
        prior_path = os.path.join(self.data_dir, 'prior_matrices.npy')
        obs_path = os.path.join(self.data_dir, 'obs_vectors.npy')
        true_path = os.path.join(self.data_dir, 'true_matrices.npy')
        
        # 验证文件是否存在
        for path in [prior_path, obs_path, true_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"数据文件不存在: {path}")
                
        # 加载数据
        prior_matrices = torch.from_numpy(np.load(prior_path)).float()
        obs_vectors = torch.from_numpy(np.load(obs_path)).float()
        true_matrices = torch.from_numpy(np.load(true_path)).float()
        
        # 验证数据形状
        assert prior_matrices.shape[1:] == (self.M, self.N), \
            f"先验矩阵形状 {prior_matrices.shape[1:]} 与配置不匹配 ({self.M}, {self.N})"
        assert obs_vectors.shape[1:] == (self.M,), \
            f"观测向量形状 {obs_vectors.shape[1:]} 与配置不匹配 ({self.M},)"
        assert true_matrices.shape[1:] == (self.M, self.N), \
            f"真实矩阵形状 {true_matrices.shape[1:]} 与配置不匹配 ({self.M}, {self.N})"
            
        print(f"成功加载数据: {prior_matrices.shape[0]} 个样本")
        print(f"先验矩阵形状: {prior_matrices.shape}")
        print(f"观测向量形状: {obs_vectors.shape}")
        print(f"真实矩阵形状: {true_matrices.shape}")
        
        return prior_matrices, obs_vectors, true_matrices
    
    def split_data(self, prior_matrices, obs_vectors, true_matrices, 
                 train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, shuffle=True):
        """
        将数据分割为训练集、验证集和测试集
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, \
            "分割比例之和必须为1"
        
        # 获取数据数量
        n_samples = prior_matrices.shape[0]
        indices = np.arange(n_samples)
        
        # 打乱索引
        if shuffle:
            np.random.shuffle(indices)
            
        # 计算分割点
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)
        
        # 分割数据
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        # 创建数据子集
        train_priors = prior_matrices[train_indices]
        train_obs = obs_vectors[train_indices]
        train_true = true_matrices[train_indices]
        
        val_priors = prior_matrices[val_indices]
        val_obs = obs_vectors[val_indices]
        val_true = true_matrices[val_indices]
        
        test_priors = prior_matrices[test_indices]
        test_obs = obs_vectors[test_indices]
        test_true = true_matrices[test_indices]
        
        print(f"数据分割完成:")
        print(f"  训练集: {train_priors.shape[0]} 个样本 ({train_ratio*100:.1f}%)")
        print(f"  验证集: {val_priors.shape[0]} 个样本 ({val_ratio*100:.1f}%)")
        print(f"  测试集: {test_priors.shape[0]} 个样本 ({test_ratio*100:.1f}%)")
        
        return (train_priors, train_obs, train_true), \
               (val_priors, val_obs, val_true), \
               (test_priors, test_obs, test_true)
    
    def get_data_loaders(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, shuffle=True):
        """
        加载数据并创建数据加载器
        """
        # 加载数据
        prior_matrices, obs_vectors, true_matrices = self.load_data()
        
        # 分割数据
        train_data, val_data, test_data = self.split_data(
            prior_matrices, obs_vectors, true_matrices,
            train_ratio, val_ratio, test_ratio, shuffle
        )
        
        # 创建数据集
        train_dataset = TensorDataset(*train_data)
        val_dataset = TensorDataset(*val_data)
        test_dataset = TensorDataset(*test_data)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        
        return train_loader, val_loader, test_loader
        
    def generate_batch(self, num_samples):
        """
        生成一批训练数据 (用于兼容旧代码)
        注意: 此方法在使用真实数据集时不应被调用
        """
        print("警告: 使用真实数据集时不应调用generate_batch方法")
        
        # 生成虚拟数据
        prior_matrices = torch.rand(num_samples, self.M, self.N)
        obs_vectors = torch.rand(num_samples, self.M)
        true_matrices = torch.rand(num_samples, self.M, self.N)
        
        # 归一化以确保每列和为1
        for i in range(num_samples):
            for j in range(self.N):
                prior_matrices[i, :, j] = prior_matrices[i, :, j] / prior_matrices[i, :, j].sum()
                true_matrices[i, :, j] = true_matrices[i, :, j] / true_matrices[i, :, j].sum()
                
        return prior_matrices, obs_vectors, true_matrices