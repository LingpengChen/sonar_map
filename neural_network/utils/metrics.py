"""评估指标计算"""
import torch
import torch.nn.functional as F
import numpy as np

def cross_entropy_loss(posterior, true_matrix):
    """
    计算交叉熵损失（按列计算）
    Args:
        posterior: 模型预测的后验矩阵 [batch_size, M, N]
        true_matrix: 真实矩阵 [batch_size, M, N]
    """
    batch_size, M, N = posterior.shape
    loss = 0
    
    for col in range(N):
        pred_col = posterior[:, :, col]
        true_col = true_matrix[:, :, col]
        
        # 对每列计算交叉熵
        col_loss = -torch.sum(true_col * torch.log(pred_col + 1e-10)) / batch_size
        loss += col_loss
    
    return loss / N  # 所有列的平均交叉熵

def kl_divergence(posterior, true_matrix):
    """
    计算KL散度（按列计算）
    """
    batch_size, M, N = posterior.shape
    kl_div = 0
    
    for col in range(N):
        pred_col = posterior[:, :, col]
        true_col = true_matrix[:, :, col]
        
        # 对每列计算KL散度
        col_kl = F.kl_div(
            torch.log(pred_col + 1e-10), 
            true_col, 
            reduction='batchmean'
        )
        kl_div += col_kl
    
    return kl_div / N  # 所有列的平均KL散度

def mse_loss(posterior, true_matrix):
    """
    计算均方误差
    """
    return F.mse_loss(posterior, true_matrix)

def column_sum_error(matrix):
    """
    检查矩阵每列和是否为1
    """
    col_sums = matrix.sum(dim=1)
    error = torch.abs(col_sums - 1.0).mean()
    return error