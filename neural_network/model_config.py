"""配置文件，存储模型参数和训练参数"""

class Config:
    # 数据参数
    # prior_matrices.shape = torch.Size([data_num, M, N])
    # true_matrices.shape = torch.Size([data_num, M, N])
    # obs_vectors.shape = torch.Size([data_num, M])
    M = 100  # 矩阵行数 range/range_resolution
    N = 31  # 矩阵列数 vertical_aperture/aperture_resolution
    
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
    train_samples_percentage = 0.7
    val_samples_percentage = 0.15
    test_samples_percentage = 0.15
    
    # 路径
    data_dir = './data/dataset'
    save_dir = './checkpoints/' + model_type
    log_dir = './logs/' + model_type
    
    def __str__(self):
        return str(self.__dict__)