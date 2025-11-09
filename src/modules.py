# src/modules.py
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    实现固定（非学习）的正弦位置编码
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # PE 矩阵 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        
        # 位置索引 (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # (d_model/2)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 广播机制 (max_len, d_model/2)
        pe[:, 0::2] = torch.sin(position * div_term) # 偶数维度
        pe[:, 1::2] = torch.cos(position * div_term) # 奇数维度
        
        # 增加 batch 维度 (1, max_len, d_model)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len, d_model)
        """
        # 截取所需长度的 PE 并加到 x 上
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class RelativePositionEmbedding(nn.Module):
    """
    相对位置编码模块 - 实现Shaw等人的相对位置编码方法
    用于在注意力计算中加入相对位置信息
    """
    def __init__(self, d_model: int, max_relative_position: int = 128):
        super().__init__()
        self.d_model = d_model
        self.max_relative_position = max_relative_position
        
        # 存储相对位置的嵌入向量，范围从 -max_relative_position 到 max_relative_position
        # 实际存储为 [0, 2*max_relative_position]，使用时需要偏移
        self.relative_embeddings = nn.Embedding(2 * max_relative_position + 1, d_model)
        
        # 初始化权重
        nn.init.xavier_uniform_(self.relative_embeddings.weight)
    
    def forward(self, seq_len_q: int, seq_len_k: int) -> torch.Tensor:
        """
        生成相对位置编码矩阵
        
        Args:
            seq_len_q: 查询序列长度
            seq_len_k: 键序列长度
            
        Returns:
            relative_positions: (seq_len_q, seq_len_k) 的相对位置索引矩阵
        """
        # 生成位置索引 [0, 1, 2, ..., seq_len_q-1]
        q_positions = torch.arange(seq_len_q, device=self.relative_embeddings.weight.device)
        # 生成位置索引 [0, 1, 2, ..., seq_len_k-1]
        k_positions = torch.arange(seq_len_k, device=self.relative_embeddings.weight.device)
        
        # 计算相对位置: pos_q - pos_k
        # (seq_len_q, 1) - (1, seq_len_k) = (seq_len_q, seq_len_k)
        relative_positions = q_positions.unsqueeze(1) - k_positions.unsqueeze(0)
        
        # 限制相对位置范围，并加上偏移量使其非负
        # 将 [-max, max] 映射到 [0, 2*max]
        relative_positions = torch.clamp(
            relative_positions, 
            -self.max_relative_position, 
            self.max_relative_position
        )
        relative_positions += self.max_relative_position
        
        return relative_positions


def scaled_dot_product_attention(Q: torch.Tensor, 
                                 K: torch.Tensor, 
                                 V: torch.Tensor, 
                                 mask: torch.Tensor = None, 
                                 relative_pos_embedding: nn.Module = None) -> (torch.Tensor, torch.Tensor):
    """
    实现缩放点积注意力，支持相对位置编码
    
    Args:
        Q (B, n_heads, seq_len_q, d_k)
        K (B, n_heads, seq_len_k, d_k)
        V (B, n_heads, seq_len_v, d_v) (seq_len_k == seq_len_v)
        mask (B, 1, 1, seq_len_k) 或 (B, 1, seq_len_q, seq_len_k)
        relative_pos_embedding: 相对位置编码模块，为None时使用标准注意力
    
    Returns:
        output (B, n_heads, seq_len_q, d_v)
        attention_weights (B, n_heads, seq_len_q, seq_len_k)
    """
    batch_size, n_heads, seq_len_q, d_k = Q.size()
    seq_len_k = K.size(2)
    
    # 标准注意力分数计算: Q*K^T/sqrt(d_k)
    # (B, n_heads, seq_len_q, seq_len_k)
    attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 如果提供了相对位置编码，则添加相对位置注意力分数
    if relative_pos_embedding is not None:
        # 获取相对位置索引矩阵 (seq_len_q, seq_len_k)
        relative_positions = relative_pos_embedding(seq_len_q, seq_len_k)
        
        # 获取相对位置嵌入向量 (seq_len_q, seq_len_k, d_k)
        pos_embeddings = relative_pos_embedding.relative_embeddings(relative_positions)
        
        # 计算相对位置贡献: Q * pos_embeddings^T / sqrt(d_k)
        # 扩展 Q 以匹配形状 (B, n_heads, seq_len_q, d_k) -> (B, n_heads, seq_len_q, 1, d_k)
        q_reshaped = Q.unsqueeze(-2)  # (B, n_heads, seq_len_q, 1, d_k)
        # 扩展 pos_embeddings 以匹配形状 (1, 1, seq_len_q, seq_len_k, d_k)
        pos_embeddings = pos_embeddings.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len_q, seq_len_k, d_k)
        # 计算贡献: (B, n_heads, seq_len_q, seq_len_k)
        pos_contribution = torch.sum(q_reshaped * pos_embeddings, dim=-1) / math.sqrt(d_k)
        
        # 添加相对位置贡献到注意力分数
        attention_scores += pos_contribution
    
    # 应用掩码
    if mask is not None:
        attention_scores = attention_scores.masked_fill(mask, -1e9)
    
    # Softmax 归一化
    attention_weights = torch.softmax(attention_scores, dim=-1)
    
    # 注意力加权和
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights