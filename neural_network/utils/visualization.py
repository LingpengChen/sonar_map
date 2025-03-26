"""可视化工具"""
import matplotlib.pyplot as plt
import torch
import numpy as np
import seaborn as sns

def plot_matrices(prior, observation, posterior, true_matrix, save_path=None):
    """
    可视化概率矩阵和观测向量
    Args:
        prior: 先验矩阵
        observation: 观测向量
        posterior: 后验矩阵
        true_matrix: 真实矩阵
        save_path: 保存路径，如果为None则显示图像
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 先验矩阵
    sns.heatmap(prior.detach().cpu().numpy(), annot=True, fmt='.2f', 
                ax=axes[0], cmap='Blues', vmin=0, vmax=1)
    axes[0].set_title('Prior Matrix')
    
    # 观测向量
    obs_display = observation.detach().cpu().numpy().reshape(-1, 1)
    sns.heatmap(obs_display, annot=True, fmt='.2f', 
                ax=axes[1], cmap='Greens', vmin=0, vmax=1)
    axes[1].set_title('Observation Vector')
    
    # 后验矩阵
    sns.heatmap(posterior.detach().cpu().numpy(), annot=True, fmt='.2f', 
                ax=axes[2], cmap='Reds', vmin=0, vmax=1)
    axes[2].set_title('Posterior Matrix')
    
    # 真实矩阵
    sns.heatmap(true_matrix.detach().cpu().numpy(), annot=True, fmt='.2f', 
                ax=axes[3], cmap='Purples', vmin=0, vmax=1)
    axes[3].set_title('True Matrix')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_training_curves(train_losses, val_losses, save_path=None):
    """
    绘制训练和验证损失曲线
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()