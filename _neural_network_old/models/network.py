import torch
import torch.nn as nn

from models.calibration import ParameterCalibrationModule
from models.feature import SonarFeatureExtractor, FeatureFusionModule
from models.physics import PhysicsPriorGenerator, PriorFusionModule
from models.refinement import ConvRefinementModule, ColumnSoftmax

class SonarReconstructionNetwork(nn.Module):
    def __init__(self, config):
        super(SonarReconstructionNetwork, self).__init__()
        
        # 从配置获取超参数
        self.A = config['A']
        self.D = config['D']
        self.N = config['N']
        self.M = config['M']
        self.K = config['K']
        self.H1 = config['H1']
        self.H2 = config['H2']
        self.C_in = config['C_in']
        self.max_depth = config['MAX_DEPTH']
        self.angle_range = config['ANGLE_RANGE']
        
        # 子模块
        self.param_calibration = ParameterCalibrationModule(self.H1, self.H2)
        self.sonar_feature_extractor = SonarFeatureExtractor(self.N, self.M)
        self.physics_prior_generator = PhysicsPriorGenerator(self.D, self.A, self.max_depth, self.angle_range)
        self.prior_fusion = PriorFusionModule(self.D, self.A)
        self.feature_fusion = FeatureFusionModule(self.D, self.A, self.M, self.K)
        self.conv_refinement = ConvRefinementModule(self.D, self.A, self.K, self.C_in)
        self.column_softmax = ColumnSoftmax()
        
    def forward(self, p, s):
        """
        完整的前向传播
        
        输入:
            p: [batch_size, 3] 参数向量 [θₐ, d, θₚ]
            s: [batch_size, N] 声纳向量
            
        输出:
            R: [batch_size, D, A] 重建的2D海床地图
            中间结果: dict, 包含中间层的输出用于可视化
        """
        # 存储中间结果
        intermediates = {}
        
        # 1. 参数校准
        p_calib = self.param_calibration(p)
        intermediates['p_calib'] = p_calib
        
        # 2. 声纳特征提取
        f_s = self.sonar_feature_extractor(s)
        intermediates['f_s'] = f_s
        
        # 3. 物理先验生成
        P_d, P_c = self.physics_prior_generator(p, p_calib)
        intermediates['P_d'] = P_d
        intermediates['P_c'] = P_c
        
        # 4. 先验融合
        P = self.prior_fusion(P_d, P_c)
        intermediates['P'] = P
        
        # 5. 特征融合
        h = self.feature_fusion(P, f_s)
        intermediates['h'] = h
        
        # 6. 卷积精细化
        R_prime = self.conv_refinement(h, P)
        intermediates['R_prime'] = R_prime
        
        # 7. 列归一化
        R = self.column_softmax(R_prime)
        
        return R, intermediates