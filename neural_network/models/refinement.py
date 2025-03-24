import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvRefinementModule(nn.Module):
    def __init__(self, D, A, K, C_in):
        super(ConvRefinementModule, self).__init__()
        self.D = D
        self.A = A
        self.C_in = C_in
        
        # 将特征映射到2D形状
        self.fc_reshape = nn.Linear(K, D*A*C_in)
        
        # 2D卷积层
        self.conv1 = nn.Conv2d(C_in, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        
    def forward(self, h, P):
        """
        使用2D卷积精细化物理先验
        
        输入:
            h: [batch_size, K] 融合特征
            P: [batch_size, D, A] 物理先验(用于残差连接)
            
        输出:
            R': [batch_size, D, A] 精细化后的2D地图
        """
        batch_size = h.shape[0]
        
        # 映射并重塑为3D张量 [batch_size, D, A, C_in]
        h_2d = self.fc_reshape(h).view(batch_size, self.D, self.A, self.C_in)
        
        # 调整形状为卷积输入 [batch_size, C_in, D, A]
        h_2d = h_2d.permute(0, 3, 1, 2)
        
        # 应用卷积层
        c1 = F.relu(self.conv1(h_2d))
        c2 = F.relu(self.conv2(c1))
        c3 = self.conv3(c2)  # [batch_size, 1, D, A]
        
        # 调整形状 [batch_size, D, A]
        R_prime = c3.squeeze(1)
        
        # 添加残差连接
        R_prime = R_prime + P
        
        return R_prime

class ColumnSoftmax(nn.Module):
    def __init__(self):
        super(ColumnSoftmax, self).__init__()
        
    def forward(self, R_prime):
        """
        对每列应用Softmax
        
        输入:
            R_prime: [batch_size, D, A] 精细化后的2D地图
            
        输出:
            R: [batch_size, D, A] 归一化后的2D地图
        """
        # 对每个角度列应用softmax
        R = F.softmax(R_prime, dim=1)
        return R