import numpy as np
import matplotlib.pyplot as plt
import torch

def visualize_reconstruction(model, p, s, device='cuda'):
    """
    可视化重建结果
    
    参数:
        model: 训练好的模型
        p: [3] 参数向量 [θₐ, d, θₚ]
        s: [N] 声纳向量
    """
    model.eval()
    
    # 准备输入
    p = p.unsqueeze(0).to(device)  # [1, 3]
    s = s.unsqueeze(0).to(device)  # [1, N]
    
    # 前向传播
    with torch.no_grad():
        R, intermediates = model(p, s)
    
    # 转换为numpy用于可视化
    R_np = R.squeeze(0).cpu().numpy()
    P_d_np = intermediates['P_d'].squeeze(0).cpu().numpy()
    P_c_np = intermediates['P_c'].squeeze(0).cpu().numpy()
    P_np = intermediates['P'].squeeze(0).cpu().numpy()
    s_np = s.squeeze(0).cpu().numpy()
    
    # 可视化
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # 直接物理先验
    im0 = axs[0, 0].imshow(P_d_np, aspect='auto', 
                          extent=[-15, 15, 20, 0])
    axs[0, 0].set_title('Direct Physics Prior')
    axs[0, 0].set_xlabel('Angle (degrees)')
    axs[0, 0].set_ylabel('Depth (m)')
    plt.colorbar(im0, ax=axs[0, 0])
    
    # 校准物理先验
    im1 = axs[0, 1].imshow(P_c_np, aspect='auto', 
                          extent=[-15, 15, 20, 0])
    axs[0, 1].set_title('Calibrated Physics Prior')
    axs[0, 1].set_xlabel('Angle (degrees)')
    axs[0, 1].set_ylabel('Depth (m)')
    plt.colorbar(im1, ax=axs[0, 1])
    
    # 最终重建结果
    im2 = axs[1, 0].imshow(R_np, aspect='auto', 
                          extent=[-15, 15, 20, 0])
    axs[1, 0].set_title('Reconstructed Seafloor Map')
    axs[1, 0].set_xlabel('Angle (degrees)')
    axs[1, 0].set_ylabel('Depth (m)')
    plt.colorbar(im2, ax=axs[1, 0])
    
    # 声纳向量
    axs[1, 1].plot(np.linspace(0, 20, len(s_np)), s_np)
    axs[1, 1].set_title('Input Sonar Vector')
    axs[1, 1].set_xlabel('Range (m)')
    axs[1, 1].set_ylabel('Intensity')
    
    plt.tight_layout()
    plt.show()
    
    return R_np, P_d_np, P_c_np, P_np

def plot_training_history(train_losses, val_losses):
    """绘制训练历史"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('training_history.png')
    plt.show()