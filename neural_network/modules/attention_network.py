"""基于注意力机制的概率更新网络"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_network import BaseNetwork

class AttentionNetwork(BaseNetwork):
    def __init__(self, M, N, hidden_dim=256, dropout=0.1):
        super().__init__(M, N, hidden_dim, dropout)
        
        # 先验矩阵投影
        self.prior_to_query = nn.Linear(N, hidden_dim)
        self.prior_to_key = nn.Linear(N, hidden_dim)
        
        # 观测向量投影
        self.obs_to_value = nn.Linear(1, hidden_dim)
        
        # 注意力缩放因子
        self.scale = hidden_dim ** -0.5
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(N + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, N)
        )
    
    def forward(self, prior_matrix, obs_vector):
        batch_size = prior_matrix.size(0)
        
        # 先验矩阵转换为Query和Key
        # [batch_size, M, N] -> [batch_size, M, hidden_dim]
        query = self.prior_to_query(prior_matrix)
        key = self.prior_to_key(prior_matrix)
        
        # 观测向量转换为Value
        # [batch_size, M] -> [batch_size, M, 1] -> [batch_size, M, hidden_dim]
        value = self.obs_to_value(obs_vector.unsqueeze(-1))
        
        # 计算注意力权重
        # [batch_size, M, hidden_dim] @ [batch_size, hidden_dim, M] -> [batch_size, M, M]
        scores = torch.bmm(query, key.transpose(1, 2)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        
        # 应用注意力权重到Value
        # [batch_size, M, M] @ [batch_size, M, hidden_dim] -> [batch_size, M, hidden_dim]
        context = torch.bmm(attn_weights, value)
        
        # 特征融合 - 连接原始先验矩阵和注意力结果
        # [batch_size, M, N+hidden_dim]
        enhanced_features = torch.cat([prior_matrix, context], dim=-1)
        fused_features = self.fusion(enhanced_features)
        
        # 解码生成输出
        output = self.decoder(fused_features)
        
        # 列方向归一化
        posterior_matrix = self._normalize_columns(output)
        
        return posterior_matrix