import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class SonarDataset(Dataset):
    def __init__(self, param_data, sonar_data, ground_truth=None, transform=None):
        """
        声纳重建数据集
        
        参数:
            param_data: [N, 3] 参数矩阵，每行包含 [θₐ, d, θₚ]
                θₐ: 垂直孔径 (度)
                d: 海床深度 (米)
                θₚ: 声纳俯角 (度)
            sonar_data: [N, L] 声纳向量矩阵，每行是一个声纳数据向量
            ground_truth: [N, D, A] (可选) 用于监督学习的地面真值2D海床地图
            transform: (可选) 数据增强变换
        """
        self.param_data = torch.tensor(param_data, dtype=torch.float32)
        self.sonar_data = torch.tensor(sonar_data, dtype=torch.float32)
        
        if ground_truth is not None:
            self.ground_truth = torch.tensor(ground_truth, dtype=torch.float32)
        else:
            self.ground_truth = None
            
        self.transform = transform
        
    def __len__(self):
        return len(self.param_data)
    
    def __getitem__(self, idx):
        p = self.param_data[idx]
        s = self.sonar_data[idx]
        
        if self.transform:
            p, s = self.transform(p, s)
            
        if self.ground_truth is not None:
            gt = self.ground_truth[idx]
            return p, s, gt
        else:
            return p, s

class SonarDataAugmentation:
    def __init__(self, param_noise_std=0.05, sonar_noise_std=0.02):
        """
        声纳数据增强
        
        参数:
            param_noise_std: 参数噪声标准差
            sonar_noise_std: 声纳噪声标准差
        """
        self.param_noise_std = param_noise_std
        self.sonar_noise_std = sonar_noise_std
        
    def __call__(self, p, s):
        # 添加参数噪声
        p_noise = torch.randn_like(p) * self.param_noise_std
        p = p + p_noise
        
        # 添加声纳噪声
        s_noise = torch.randn_like(s) * self.sonar_noise_std
        s = s + s_noise
        s = torch.clamp(s, 0, 1)  # 确保在合理范围内
        
        return p, s

def generate_synthetic_dataset(num_samples, N, D, A, noise_level=0.1):
    """
    生成合成数据集用于训练和测试
    """
    param_data = np.zeros((num_samples, 3))
    sonar_data = np.zeros((num_samples, N))
    ground_truth = np.zeros((num_samples, D, A))
    
    # 预计算角度和深度
    angles = np.linspace(-15, 15, A)
    depths = np.linspace(0, 20, D)
    sonar_bins = np.linspace(0, 20, N)
    
    for i in range(num_samples):
        # 随机生成参数
        theta_a = 30.0  # 固定垂直孔径为30度
        d = np.random.uniform(5, 15)  # 随机海床深度5-15m
        theta_p = np.random.uniform(-20, -5)  # 随机声纳俯角 -20到-5度
        
        param_data[i] = [theta_a, d, theta_p]
        
        # 生成2D地图 - 模拟平坦海床
        for j, alpha in enumerate(angles):
            angle_rad = (theta_p + alpha) * np.pi / 180
            d_theory = d / np.cos(angle_rad) if np.cos(angle_rad) > 0.1 else 20
            
            # 在理论深度周围生成高斯分布
            for k, depth in enumerate(depths):
                ground_truth[i, k, j] = np.exp(-((depth - d_theory) ** 2) / (2 * 0.5 ** 2))
                
            # 归一化每列
            if ground_truth[i, :, j].sum() > 0:
                ground_truth[i, :, j] = ground_truth[i, :, j] / ground_truth[i, :, j].sum()
        
        # 生成对应的声纳向量 - 简化模型
        for k, bin_range in enumerate(sonar_bins):
            # 寻找此距离对应的深度概率
            for j, alpha in enumerate(angles):
                angle_rad = (theta_p + alpha) * np.pi / 180
                if np.cos(angle_rad) > 0.1:
                    # 找到距离对应的深度索引
                    depth_idx = np.argmin(np.abs(depths - bin_range))
                    # 添加到声纳向量
                    sonar_data[i, k] += ground_truth[i, depth_idx, j] / A
        
        # 添加噪声
        sonar_data[i] += np.random.normal(0, noise_level, N)
        sonar_data[i] = np.clip(sonar_data[i], 0, 1)
    
    return param_data, sonar_data, ground_truth

def create_data_loaders(config):
    """创建数据加载器"""
    # 获取配置参数
    N = config['model']['N']
    D = config['model']['D']
    A = config['model']['A']
    batch_size = config['training']['batch_size']
    train_ratio = config['dataset']['train_ratio']
    num_samples = config['dataset']['synthetic_samples']
    noise_level = config['dataset']['noise_level']
    param_noise_std = config['augmentation']['param_noise_std']
    sonar_noise_std = config['augmentation']['sonar_noise_std']
    
    # 生成合成数据集
    param_data, sonar_data, ground_truth = generate_synthetic_dataset(
        num_samples, N, D, A, noise_level
    )
    
    # 分割训练集和验证集
    split_idx = int(len(param_data) * train_ratio)
    
    train_params = param_data[:split_idx]
    train_sonar = sonar_data[:split_idx]
    val_params = param_data[split_idx:]
    val_sonar = sonar_data[split_idx:]
    
    # 创建数据增强
    transform = SonarDataAugmentation(param_noise_std, sonar_noise_std)
    
    # 创建数据集
    train_dataset = SonarDataset(train_params, train_sonar, transform=transform)
    val_dataset = SonarDataset(val_params, val_sonar)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader