"""配置文件，存储模型参数和训练参数"""

class Config:
    # 数据参数
    M = 6  # 矩阵行数
    N = 4  # 矩阵列数
    
    # 模型参数
    model_type = 'concat'  # 'concat' 或 'attention'
    hidden_dim = 256
    dropout = 0.1
    
    # 训练参数
    batch_size = 64
    epochs = 100
    learning_rate = 0.001
    weight_decay = 1e-5
    
    # 数据生成参数
    train_samples = 10000
    val_samples = 1000
    test_samples = 1000
    
    # 路径
    save_dir = './checkpoints'
    log_dir = './logs'
    
    def __str__(self):
        return str(self.__dict__)