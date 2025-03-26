"""神经网络基础类"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseNetwork(nn.Module):
    def __init__(self, M, N, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.M = M
        self.N = N
        self.hidden_dim = hidden_dim
        self.dropout = dropout
    
    def forward(self, prior_matrix, obs_vector):
        """
        Args:
            prior_matrix: 先验矩阵 [batch_size, M, N]
            obs_vector: 观测向量 [batch_size, M]
        
        Returns:
            posterior_matrix: 后验矩阵 [batch_size, M, N]
        """
        raise NotImplementedError
    
    def _normalize_columns(self, matrix):
        """对矩阵的每一列应用softmax，保证每列和为1"""
        return F.softmax(matrix, dim=1)