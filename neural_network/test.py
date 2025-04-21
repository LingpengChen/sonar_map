from model_config import Config
from utils import SonarMapDataLoader, SonarMapDataGenerator, cross_entropy_loss, plot_matrices, plot_training_curves

if __name__ == '__main__':
    config = Config()
    
    data_loader = SonarMapDataLoader(config.M, config.N, config.batch_size, data_dir=config.data_dir)
    train_loader, val_loader, test_loader = data_loader.get_data_loaders(
        train_ratio=config.train_samples_percentage, val_ratio=config.val_samples_percentage, test_ratio=config.test_samples_percentage, shuffle=True
    )