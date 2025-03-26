import torch
import torch.nn as nn
import torch.nn.functional as F

class SonarReconstructionLoss(nn.Module):
    def __init__(self, config):
        super(SonarReconstructionLoss, self).__init__()
        # 获取配置参数
        self.D = config['D']
        self.A = config['A']
        self.N = config['N']
        self.max_depth = config['MAX_DEPTH']
        self.lambda_phys = config['training']['lambda_phys']
        self.lambda_ent = config['training']['lambda_ent']
        
        # 预计算每个深度对应的声纳bin索引
        self.register_buffer('depth_to_bin', torch.zeros(self.D, self.N))
        depths = torch.linspace(0, self.max_depth, self.D)
        bins = torch.linspace(0, self.max_depth, self.N)
        
        # 构建映射矩阵
        for i, depth in enumerate(depths):
            # 找到最近的bin
            bin_idx = torch.argmin(torch.abs(bins - depth)).item()
            self.depth_to_bin[i, bin_idx] = 1.0
            
    def forward(self, R, P, s):
        """
        计算总损失
        
        输入:
            R: [batch_size, D, A] 重建的2D海床地图
            P: [batch_size, D, A] 物理先验
            s: [batch_size, N] 原始声纳向量
            
        输出:
            loss: 总损失
            losses: dict, 包含各部分损失
        """
        batch_size = R.shape[0]
        
        # 1. 重建损失 - 将2D地图压缩回1D声纳向量并比较
        # 沿角度维度求和，得到深度分布
        R_depth = R.sum(dim=2)  # [batch_size, D]
        
        # 映射到声纳bins
        R_sonar = torch.matmul(R_depth, self.depth_to_bin)  # [batch_size, N]
        
        # 计算与原始声纳的MSE
        recon_loss = F.mse_loss(R_sonar, s)
        
        # 2. 物理一致性损失
        # 创建置信度掩码 - 先验概率高的区域权重大
        mask = P.clone().detach()
        phys_loss = torch.mean(((R - P) ** 2) * mask)
        
        # 3. 熵正则化损失
        # 添加小值避免log(0)
        epsilon = 1e-10
        entropy_loss = -torch.mean(torch.sum(R * torch.log(R + epsilon), dim=1))
        
        # 总损失
        total_loss = recon_loss + self.lambda_phys * phys_loss + self.lambda_ent * entropy_loss
        
        # 返回总损失和各部分损失
        losses = {
            'total': total_loss,
            'recon': recon_loss,
            'physics': phys_loss,
            'entropy': entropy_loss
        }
        
        return total_loss, losses