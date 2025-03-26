"""基于连接的概率更新网络"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_network import BaseNetwork

class ConcatNetwork(BaseNetwork):
    def __init__(self, M, N, hidden_dim=256, dropout=0.1):
        super().__init__(M, N, hidden_dim, dropout)
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(2*M*N, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, M*N),
        )
    
    def forward(self, prior_matrix, obs_vector):
        batch_size = prior_matrix.size(0)
        
        # 扩展观测向量为与prior_matrix相同形状
        expanded_obs = obs_vector.unsqueeze(-1).expand(-1, -1, self.N)
        
        # 连接先验矩阵和扩展后的观测向量
        concat_input = torch.cat([prior_matrix, expanded_obs], dim=1)
        
        # 展平为二维张量供全连接层处理
        concat_flat = concat_input.view(batch_size, -1)
        
        # 通过编码器和解码器
        features = self.encoder(concat_flat)
        output_flat = self.decoder(features)
        
        # 重塑为原始形状
        output = output_flat.view(batch_size, self.M, self.N)
        
        # 列方向归一化
        posterior_matrix = self._normalize_columns(output)
        
        return posterior_matrix