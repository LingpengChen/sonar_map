from .concat_network import ConcatNetwork
from .attention_network import AttentionNetwork

def get_network(config):
    if config.model_type == 'concat':
        return ConcatNetwork(config.M, config.N, config.hidden_dim, config.dropout)
    elif config.model_type == 'attention':
        return AttentionNetwork(config.M, config.N, config.hidden_dim, config.dropout)
    else:
        raise ValueError(f"不支持的模型类型: {config.model_type}")