import torch
import torch.nn as nn
import torch.nn.functional as F

class SonarFeatureExtractor(nn.Module):
    def __init__(self, input_size, output_size):
        super(SonarFeatureExtractor, self).__init__()
        
        # CNN + 残差连接
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2)
        self.conv5 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        
        # 特征汇总全连接层
        self.fc = nn.Linear(64 * input_size, output_size)
        
    def forward(self, s):
        """
        输入: s ∈ ℝᴺ (声纳向量)
        输出: f_s ∈ ℝᴹ (声纳特征向量)
        """
        # 调整输入形状 [batch, N] -> [batch, 1, N]
        s = s.unsqueeze(1)
        
        # 卷积层 + 残差连接
        c1 = F.relu(self.conv1(s))
        c2 = F.relu(self.conv2(c1))
        c3 = F.relu(self.conv3(c2))
        c4 = F.relu(self.conv4(c3))
        c5 = c2 + c4  # 残差连接
        
        # 最终卷积层
        c6 = F.relu(self.conv5(c5))
        
        # 扁平化并应用全连接层
        flat = c6.view(c6.size(0), -1)
        f_s = self.fc(flat)
        
        return f_s

class FeatureFusionModule(nn.Module):
    def __init__(self, D, A, M, K):
        super(FeatureFusionModule, self).__init__()
        self.D = D
        self.A = A
        self.M = M
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(D*A + M, K),
            nn.ReLU()
        )
        
    def forward(self, P, f_s):
        """
        融合物理先验和声纳特征
        
        输入:
            P: [batch_size, D, A] 物理先验
            f_s: [batch_size, M] 声纳特征
            
        输出:
            h: [batch_size, K] 融合特征
        """
        batch_size = P.shape[0]
        
        # 扁平化物理先验
        p_flat = P.view(batch_size, -1)  # [batch_size, D*A]
        
        # 连接特征
        h_concat = torch.cat([p_flat, f_s], dim=1)  # [batch_size, D*A+M]
        
        # 融合
        h = self.fusion(h_concat)  # [batch_size, K]
        
        return h