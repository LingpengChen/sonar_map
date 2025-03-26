"""推理脚本"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from model_config import Config
from modules import get_network
from utils import ProbabilityDataGenerator, plot_matrices, cross_entropy_loss, kl_divergence, mse_loss

def inference(config, model_path, num_samples=10):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = get_network(config).to(device)
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"加载模型: {model_path}, 验证损失: {checkpoint.get('val_loss', 'N/A')}")
    
    # 创建输出目录
    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)
    
    # 生成测试数据
    data_generator = ProbabilityDataGenerator(config.M, config.N)
    prior_matrices, obs_vectors, true_matrices = data_generator.generate_batch(num_samples)
    
    # 评估模型
    model.eval()
    losses = []
    kl_divs = []
    mse_losses = []
    
    with torch.no_grad():
        for i in range(num_samples):
            prior = prior_matrices[i].unsqueeze(0).to(device)
            obs = obs_vectors[i].unsqueeze(0).to(device)
            true = true_matrices[i].unsqueeze(0).to(device)
            
            # 前向传播
            posterior = model(prior, obs)
            
            # 计算指标
            loss = cross_entropy_loss(posterior, true).item()
            kl_div = kl_divergence(posterior, true).item()
            mse = mse_loss(posterior, true).item()
            
            losses.append(loss)
            kl_divs.append(kl_div)
            mse_losses.append(mse)
            
            # 可视化结果
            plot_matrices(
                prior[0], obs[0], posterior[0], true[0],
                save_path=os.path.join(results_dir, f'sample_{i+1}.png')
            )
            
            # 打印结果
            print(f"样本 {i+1}:")
            print(f"  交叉熵损失: {loss:.4f}")
            print(f"  KL散度: {kl_div:.4f}")
            print(f"  MSE: {mse:.4f}")
    
    # 打印总体结果
    print("\n总体结果:")
    print(f"平均交叉熵损失: {np.mean(losses):.4f} ± {np.std(losses):.4f}")
    print(f"平均KL散度: {np.mean(kl_divs):.4f} ± {np.std(kl_divs):.4f}")
    print(f"平均MSE: {np.mean(mse_losses):.4f} ± {np.std(mse_losses):.4f}")
    
    # 绘制结果直方图
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(losses, bins=10)
    plt.xlabel('Cross Entropy Loss')
    plt.ylabel('Frequency')
    plt.title(f'Avg: {np.mean(losses):.4f}')
    
    plt.subplot(1, 3, 2)
    plt.hist(kl_divs, bins=10)
    plt.xlabel('KL Divergence')
    plt.ylabel('Frequency')
    plt.title(f'Avg: {np.mean(kl_divs):.4f}')
    
    plt.subplot(1, 3, 3)
    plt.hist(mse_losses, bins=10)
    plt.xlabel('MSE')
    plt.ylabel('Frequency')
    plt.title(f'Avg: {np.mean(mse_losses):.4f}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'metrics_histogram.png'))
    
    return np.mean(losses), np.mean(kl_divs), np.mean(mse_losses)

if __name__ == "__main__":
    config = Config()
    model_path = os.path.join(config.save_dir, 'best_model.pth')
    inference(config, model_path)