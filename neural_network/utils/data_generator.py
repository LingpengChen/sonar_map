"""生成概率矩阵训练数据"""
import torch
import numpy as np

class ProbabilityDataGenerator:
    def __init__(self, M, N, batch_size=64):
        """
        初始化数据生成器
        Args:
            M: 矩阵行数
            N: 矩阵列数
            batch_size: 批处理大小
        """
        self.M = M
        self.N = N
        self.batch_size = batch_size
    
    def generate_batch(self, num_samples):
        """
        生成一批训练数据
        Args:
            num_samples: 样本数量
        
        Returns:
            prior_matrices: 先验矩阵 [num_samples, M, N]
            obs_vectors: 观测向量 [num_samples, M]
            true_matrices: 真实矩阵 [num_samples, M, N]
        """
        # 生成真实矩阵（每列是一个概率分布）
        true_matrices = torch.zeros(num_samples, self.M, self.N)
        for i in range(num_samples):
            for j in range(self.N):
                probs = torch.rand(self.M)
                true_matrices[i, :, j] = probs / probs.sum()  # 归一化为概率分布
        
        # 生成先验矩阵（对真实矩阵添加噪声）
        noise = torch.rand(num_samples, self.M, self.N) * 0.5
        prior_matrices_raw = true_matrices + noise
        
        # 归一化先验矩阵，确保每列和为1
        prior_matrices = torch.zeros_like(prior_matrices_raw)
        for i in range(num_samples):
            for j in range(self.N):
                col_sum = prior_matrices_raw[i, :, j].sum()
                prior_matrices[i, :, j] = prior_matrices_raw[i, :, j] / col_sum
        
        # 生成观测向量（基于真实矩阵的一些特征）
        obs_vectors = torch.zeros(num_samples, self.M)
        for i in range(num_samples):
            # 简单示例：观测向量是真实矩阵所有列的加权和
            weights = torch.rand(self.N)
            obs_vectors[i] = torch.matmul(true_matrices[i], weights)
            
            # 添加一些噪声
            noise = torch.rand(self.M) * 0.1
            obs_vectors[i] = obs_vectors[i] + noise
            
            # 归一化观测向量（可选）
            obs_vectors[i] = obs_vectors[i] / obs_vectors[i].sum()
        
        return prior_matrices, obs_vectors, true_matrices
    
    def get_data_loaders(self, train_samples, val_samples, test_samples):
        """
        创建训练、验证和测试数据加载器
        """
        from torch.utils.data import TensorDataset, DataLoader
        
        # 生成训练数据
        train_priors, train_obs, train_true = self.generate_batch(train_samples)
        train_dataset = TensorDataset(train_priors, train_obs, train_true)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # 生成验证数据
        val_priors, val_obs, val_true = self.generate_batch(val_samples)
        val_dataset = TensorDataset(val_priors, val_obs, val_true)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        # 生成测试数据
        test_priors, test_obs, test_true = self.generate_batch(test_samples)
        test_dataset = TensorDataset(test_priors, test_obs, test_true)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        
        return train_loader, val_loader, test_loader