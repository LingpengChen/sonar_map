"""主入口文件"""
import os
import argparse

from model_config import Config
from train import train
from inference import inference

def main():
    parser = argparse.ArgumentParser(description='概率矩阵更新网络')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'both'], 
                        help='运行模式: train, test, 或 both')
    parser.add_argument('--model', type=str, default='attention', choices=['concat', 'attention'], 
                        help='模型类型: concat 或 attention')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批处理大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    
    args = parser.parse_args()
    
    # 配置
    config = Config()
    config.model_type = args.model
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    
    print(f"运行模式: {args.mode}")
    print(f"模型类型: {args.model}")
    
    if args.mode in ['train', 'both']:
        # 训练模型
        model = train(config)
        
    if args.mode in ['test', 'both']:
        # 测试模型
        model_path = os.path.join(config.save_dir, 'best_model.pth')
        if not os.path.exists(model_path):
            print(f"模型文件 {model_path} 不存在")
            return
        
        inference(config, model_path)

if __name__ == "__main__":
    main()