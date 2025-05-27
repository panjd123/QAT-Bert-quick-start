import torch

class ModelConfig:
    def __init__(self):
        # 文档长度配置
        self.doc1_max_len = 64
        self.doc2_max_len = 128
        
        # 模型维度配置
        self.doc1_dim = 256    # 第一个文档的维度
        self.doc2_dim = 256    # 第二个文档的维度
        self.total_dim = self.doc1_dim + self.doc2_dim  # 总维度
        
        # Transformer配置
        self.num_heads = 16
        self.num_layers = 6
        self.dropout = 0.1
        self.ff_dim = 4 * self.total_dim     # 前馈网络维度
        
        # 相似度计算层配置
        self.similarity_hidden_dims = [256, 64]  # 相似度计算层的隐藏层维度
        self.activation = 'gelu'  # 可选: 'relu', 'gelu', 'tanh'
        
        # 训练配置
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.num_epochs = 100
        
        # 词表配置
        self.min_word_freq = 2  # 词表最小词频
        
    def get_activation(self):
        """获取激活函数"""
        if self.activation == 'relu':
            return torch.nn.ReLU()
        elif self.activation == 'gelu':
            return torch.nn.GELU()
        elif self.activation == 'tanh':
            return torch.nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}") 