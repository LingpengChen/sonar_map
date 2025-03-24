import torch
import yaml
import argparse
import numpy as np
from models.network import SonarReconstructionNetwork
from utils.visualization import visualize_reconstruction

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_model(config, model_path, device):
    """加载训练好的模型"""
    model = SonarReconstructionNetwork(config['model']).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def reconstruct_seafloor(model, sonar_data, params, device):
    """使用模型重建海床地图"""
    # 确保数据格式正确并转移到正确设备
    if not isinstance(sonar_data, torch.Tensor):
        sonar_data = torch.tensor(sonar_data, dtype=torch.float32)
    if not isinstance(params, torch.Tensor):
        params = torch.tensor(params, dtype=torch.float32)
    
    # 添加批次维度(如果需要)
    if sonar_data.dim() == 1:
        sonar_data = sonar_data.unsqueeze(0)
    if params.dim() == 1:
        params = params.unsqueeze(0)
    
    sonar_data = sonar_data.to(device)
    params = params.to(device)
    
    # 执行推理
    with torch.no_grad():
        reconstruction, intermediates = model(params, sonar_data)
    
    # 返回结果
    return reconstruction, intermediates

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Sonar Reconstruction Inference')
    parser.add_argument('--config', type=str, default='config/config.yaml', 
                        help='Path to config file')
    parser.add_argument('--model', type=str, default='checkpoints/best_sonar_model.pth', 
                        help='Path to model checkpoint')
    parser.add_argument('--sonar_file', type=str, help='Path to sonar data file')
    parser.add_argument('--params_file', type=str, help='Path to parameters file')
    parser.add_argument('--output', type=str, default='reconstruction.npy', 
                        help='Output file for reconstruction')
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载配置和模型
    config = load_config(args.config)
    model = load_model(config, args.model, device)
    
    # 读取输入数据
    if args.sonar_file and args.params_file:
        # 从文件加载
        sonar_data = np.load(args.sonar_file)
        params = np.load(args.params_file)
    else:
        # 使用示例数据
        print("未提供输入文件，使用示例数据...")
        N = config['model']['N']
        sonar_data = np.random.rand(N)
        params = np.array([30.0, 10.0, -15.0])  # [θₐ, d, θₚ]
    
    # 执行推理
    reconstruction, intermediates = reconstruct_seafloor(model, sonar_data, params, device)
    
    # 保存结果
    reconstruction_np = reconstruction.cpu().numpy()
    np.save(args.output, reconstruction_np)
    print(f"重建结果已保存到 {args.output}")
    
    # 可视化结果
    p_tensor = torch.tensor(params, dtype=torch.float32)
    s_tensor = torch.tensor(sonar_data, dtype=torch.float32)
    visualize_reconstruction(model, p_tensor, s_tensor, device)

if __name__ == "__main__":
    main()