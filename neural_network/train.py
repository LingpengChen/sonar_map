import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.network import SonarReconstructionNetwork
from utils.loss import SonarReconstructionLoss
from utils.dataset import create_data_loaders
from utils.visualization import plot_training_history, visualize_reconstruction

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    epoch_loss = 0
    
    for p, s in dataloader:
        p, s = p.to(device), s.to(device)
        
        # 前向传播
        R, intermediates = model(p, s)
        
        # 计算损失
        loss, _ = criterion(R, intermediates['P'], s)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    """验证模型"""
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for p, s in dataloader:
            p, s = p.to(device), s.to(device)
            
            # 前向传播
            R, intermediates = model(p, s)
            
            # 计算损失
            loss, _ = criterion(R, intermediates['P'], s)
            
            val_loss += loss.item()
            
    return val_loss / len(dataloader)

def train_model(config, train_loader, val_loader, device):
    """训练模型的主函数"""
    # 创建模型
    model = SonarReconstructionNetwork(config['model']).to(device)
    
    # 创建损失函数
    criterion = SonarReconstructionLoss(config).to(device)
    
    # 创建优化器
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['training']['learning_rate']
    )
    
    # 创建学习率调度器
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=config['training']['lr_decay_factor'], 
        patience=config['training']['lr_patience']
    )
    
    # 训练参数
    num_epochs = config['training']['num_epochs']
    
    # 存储训练和验证损失
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # 确保保存模型的目录存在
    os.makedirs('checkpoints', exist_ok=True)
    
    print("开始训练...")
    for epoch in range(num_epochs):
        # 训练一个epoch
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # 验证
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # 学习率调整
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'checkpoints/best_sonar_model.pth')
            
        print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}')
    
    # 加载最佳模型
    model.load_state_dict(torch.load('checkpoints/best_sonar_model.pth'))
    
    # 绘制训练历史
    plot_training_history(train_losses, val_losses)
    
    return model, train_losses, val_losses

def main():
    """主函数"""
    # 加载配置
    config = load_config('config/config.yaml')
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据加载器
    train_loader, val_loader = create_data_loaders(config)
    
    # 训练模型
    model, train_losses, val_losses = train_model(config, train_loader, val_loader, device)
    
    # 测试和可视化一个样本
    for p, s in val_loader:
        sample_p = p[0]
        sample_s = s[0]
        break
    
    visualize_reconstruction(model, sample_p, sample_s, device)
    
    print("训练完成!")

if __name__ == "__main__":
    main()