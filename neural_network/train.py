"""训练脚本"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np

from model_config import Config
from modules import get_network
from utils import SonarMapDataLoader, SonarMapDataGenerator, cross_entropy_loss, plot_matrices, plot_training_curves

def train(config: Config):
    # 创建保存目录
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # 创建TensorBoard日志
    writer = SummaryWriter(config.log_dir)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    data_loader = SonarMapDataLoader(config.M, config.N, config.batch_size, data_dir=config.data_dir)
    train_loader, val_loader, test_loader = data_loader.get_data_loaders(
        train_ratio=config.train_samples_percentage, val_ratio=config.val_samples_percentage, test_ratio=config.test_samples_percentage, shuffle=True
    )
    
    # 初始化模型
    model = get_network(config).to(device)
    print(f"模型类型: {config.model_type}")
    print(f"模型总参数: {sum(p.numel() for p in model.parameters())}")
    
    
    from torchinfo import summary
    # 获取一个批次数据的形状用于模型摘要
    sample_input_shapes = [(1, config.M, config.N), (1, config.M)]
    sample_inputs = [torch.rand(*shape).to(device) for shape in sample_input_shapes]
    print("模型摘要 (使用torchinfo):")
    summary(model, input_data=sample_inputs, device=device)
    print("="*50 + "\n")
    
    # return

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 记录训练指标
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # 训练循环
    print(f"开始训练，总共 {config.epochs} 轮...")
    start_time = time.time()
    
    for epoch in range(config.epochs):
        # 训练阶段
        model.train()
        epoch_loss = 0
        
        for batch_idx, (prior_matrices, obs_vectors, true_matrices) in enumerate(train_loader):
            prior_matrices = prior_matrices.to(device)
            obs_vectors = obs_vectors.to(device)
            true_matrices = true_matrices.to(device)
            
            # 前向传播
            posterior_matrices = model(prior_matrices, obs_vectors)
            
            # 计算损失
            loss = cross_entropy_loss(posterior_matrices, true_matrices)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 记录损失
            epoch_loss += loss.item()
            
            # # 打印进度 2448个样本/64每一批次=39批次，每个epoch只有39批次，所以%50不会显示
            # if (batch_idx + 1) % 50 == 0:
            #     print(f"Epoch [{epoch+1}/{config.epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        # 计算平均训练损失
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for prior_matrices, obs_vectors, true_matrices in val_loader:
                prior_matrices = prior_matrices.to(device)
                obs_vectors = obs_vectors.to(device)
                true_matrices = true_matrices.to(device)
                
                # 前向传播
                posterior_matrices = model(prior_matrices, obs_vectors)
                
                # 计算损失
                loss = cross_entropy_loss(posterior_matrices, true_matrices)
                val_loss += loss.item()
                
        # 计算平均验证损失
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        
        # 更新学习率
        scheduler.step(avg_val_loss)
        
        # 打印结果
        print(f"Epoch [{epoch+1}/{config.epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # 可视化一个批次的结果
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                sample_prior = prior_matrices[0].cpu()
                sample_obs = obs_vectors[0].cpu()
                sample_true = true_matrices[0].cpu()
                sample_posterior = posterior_matrices[0].cpu()
                
                plot_matrices(
                    sample_prior, sample_obs, sample_posterior, sample_true,
                    save_path=os.path.join(config.log_dir, f'matrices_epoch_{epoch+1}.png')
                )
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, os.path.join(config.save_dir, 'best_model.pth'))
            print(f"保存最佳模型，验证损失: {best_val_loss:.4f}")
    
    # 训练完成，绘制损失曲线
    plot_training_curves(
        train_losses, val_losses, 
        save_path=os.path.join(config.log_dir, 'training_curves.png')
    )
    
    # 保存最终模型
    torch.save({
        'epoch': config.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1],
    }, os.path.join(config.save_dir, 'final_model.pth'))
    
    # 计算训练时间
    total_time = time.time() - start_time
    print(f"训练完成! 总时间: {total_time/60:.2f} 分钟")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    
    return model

if __name__ == "__main__":
    config = Config()
    print("配置信息:")
    print(config)
    train(config)