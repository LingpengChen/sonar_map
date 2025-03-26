import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PhysicsPriorGenerator(nn.Module):
    def __init__(self, D, A, max_depth, angle_range):
        super(PhysicsPriorGenerator, self).__init__()
        self.D = D  # 深度维度
        self.A = A  # 角度维度
        self.max_depth = max_depth  # 最大深度(米)
        self.angle_range = angle_range  # 角度范围(度)
        self.sigma = 0.5  # 高斯分布的标准差，可调节
        
        # 预计算角度和深度离散化
        self.register_buffer('angles', torch.linspace(-angle_range/2, angle_range/2, A))  # [-15, 15]度
        self.register_buffer('depths', torch.linspace(0, max_depth, D))  # [0, 20]米
        
    def generate_prior(self, p):
        """
        生成物理先验
        
        输入:
            p: [batch_size, 3] 包含 [θₐ, d, θₚ]
            
        输出:
            P: [batch_size, D, A] 物理先验矩阵
        """
        batch_size = p.shape[0]
        
        # 提取参数
        theta_a = p[:, 0]  # 垂直孔径 (不直接使用，而是使用预设的angles)
        d = p[:, 1]  # 海床深度
        theta_p = p[:, 2]  # 声纳俯角
        
        # 创建输出矩阵
        P = torch.zeros(batch_size, self.D, self.A, device=p.device)
        
        # 对每个批次样本计算
        for b in range(batch_size):
            for i, alpha in enumerate(self.angles):
                # 计算该角度的理论深度
                angle_rad = (theta_p[b] + alpha) * np.pi / 180
                d_theory = d[b] / torch.cos(angle_rad)
                
                # 如果理论深度大于最大深度或为负，则跳过
                if d_theory <= 0 or d_theory >= self.max_depth:
                    continue
                
                # 计算高斯分布
                for j, depth in enumerate(self.depths):
                    P[b, j, i] = torch.exp(-((depth - d_theory) ** 2) / (2 * self.sigma ** 2))
            
            # 列归一化(对每个角度)
            for i in range(self.A):
                col_sum = P[b, :, i].sum()
                if col_sum > 0:  # 避免除以零
                    P[b, :, i] = P[b, :, i] / col_sum
                    
        return P
        
    def forward(self, p, p_calib):
        """
        生成两种物理先验
        
        输入:
            p: [batch_size, 3] 原始参数 [θₐ, d, θₚ]
            p_calib: [batch_size, 3] 校准参数 [θₐ', d', θₚ']
            
        输出:
            P_d: [batch_size, D, A] 直接物理先验
            P_c: [batch_size, D, A] 校准物理先验
        """
        P_d = self.generate_prior(p)  # 直接物理先验
        P_c = self.generate_prior(p_calib)  # 校准物理先验
        
        return P_d, P_c

class PriorFusionModule(nn.Module):
    def __init__(self, D, A, learnable=True):
        super(PriorFusionModule, self).__init__()
        self.D = D
        self.A = A
        self.learnable = learnable
        
        if learnable:
            # 可学习的融合参数(初始化为0.5)
            self.lambda_param = nn.Parameter(torch.tensor(0.5))
            # 动态融合层
            self.dynamic_fusion = nn.Sequential(
                nn.Linear(D*A*2, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
            
    def forward(self, P_d, P_c, dynamic=False):
        """
        融合两个物理先验
        
        输入:
            P_d: [batch_size, D, A] 直接物理先验
            P_c: [batch_size, D, A] 校准物理先验
            dynamic: 是否使用动态融合权重
            
        输出:
            P: [batch_size, D, A] 融合后的物理先验
        """
        if not self.learnable:
            # 使用固定的融合权重0.5
            P = 0.5 * P_d + 0.5 * P_c
        elif not dynamic:
            # 使用可学习但全局的融合权重
            lambda_val = torch.sigmoid(self.lambda_param)  # 确保在[0,1]范围内
            P = lambda_val * P_d + (1 - lambda_val) * P_c
        else:
            # 动态融合权重
            batch_size = P_d.shape[0]
            flat_priors = torch.cat([
                P_d.view(batch_size, -1),
                P_c.view(batch_size, -1)
            ], dim=1)
            lambda_dynamic = self.dynamic_fusion(flat_priors)
            P = lambda_dynamic.view(-1, 1, 1) * P_d + (1 - lambda_dynamic.view(-1, 1, 1)) * P_c
            
        return P