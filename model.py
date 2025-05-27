import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import ModelConfig

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # 线性变换
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力
        # scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # if mask is not None:
        #     scores = scores.masked_fill(mask == 0, -1e9)
        # attn = torch.softmax(scores, dim=-1)
        # 应用注意力
        # out = torch.matmul(attn, v)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_linear(out)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, activation):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            activation,
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 自注意力
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class FirstColumn(nn.Module):
    def forward(self, x):
        return x[:, 0]

class DocumentSimilarityModel(nn.Module):
    def __init__(self, vocab_size, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, config.total_dim)
        self.pos_encoding = PositionalEncoding(config.total_dim, config.doc1_max_len + config.doc2_max_len)
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer编码器层
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                config.total_dim, 
                config.num_heads, 
                config.ff_dim, 
                config.dropout,
                config.get_activation()
            )
            for _ in range(config.num_layers)
        ])
        
        if config.similarity_hidden_dims is None:
            self.similarity = FirstColumn()
        else:
            # 相似度计算层
            similarity_layers = []
            prev_dim = config.total_dim
            for hidden_dim in config.similarity_hidden_dims:
                similarity_layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    config.get_activation(),
                    nn.Dropout(config.dropout)
                ])
                prev_dim = hidden_dim
            
            similarity_layers.append(nn.Linear(prev_dim, 1))
            similarity_layers.append(nn.Sigmoid())
            
            self.similarity = nn.Sequential(*similarity_layers)
        
    def forward(self, doc1_ids, doc2_ids):
        # 将两个文档的ID连接在一起
        combined_ids = torch.cat([doc1_ids, doc2_ids], dim=1)
        
        # 词嵌入
        x = self.embedding(combined_ids)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # 编码器层
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        
        # 使用[CLS]标记的输出作为文档对的表示
        x = x[:, 0, :]
        
        # 计算相似度
        similarity = self.similarity(x)
        
        return similarity.squeeze() 