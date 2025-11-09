# src/layers.py
import torch
import torch.nn as nn
from src.modules import scaled_dot_product_attention
from src import config

from src.modules import RelativePositionEmbedding

class MultiHeadAttention(nn.Module):
    """
    实现多头注意力，支持相对位置编码
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, use_relative_pos: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        self.use_relative_pos = use_relative_pos
        
        # 定义线性层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # 如果使用相对位置编码，创建相对位置嵌入模块
        self.relative_pos_embedding = RelativePositionEmbedding(self.d_k) if use_relative_pos else None
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, 
                Q: torch.Tensor, 
                K: torch.Tensor, 
                V: torch.Tensor, 
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播
        """
        batch_size = Q.size(0)
        
        q_proj = self.W_q(Q)
        k_proj = self.W_k(K)
        v_proj = self.W_v(V)
        
        # 重塑并转置，为多头注意力做准备
        q_proj = q_proj.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_proj = k_proj.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_proj = v_proj.view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        attn_output, _ = scaled_dot_product_attention(q_proj, k_proj, v_proj, mask, self.relative_pos_embedding)
        
        # 合并多头结果
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, self.d_model)
        
        output = self.W_o(attn_output)
        return output

class PositionwiseFeedForward(nn.Module):
    """
    实现逐点前馈网络
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.gelu = nn.GELU()  # 替换ReLU为GeLU
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x (B, L, d_model) -> (B, L, d_ff) -> (B, L, d_model)
        return self.w_2(self.dropout(self.gelu(self.w_1(x))))


class EncoderLayer(nn.Module):
    """
    实现单个 Encoder 层
    采用 Post-Norm 结构: x + Sublayer(x) -> LayerNorm
    支持相对位置编码
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, use_relative_pos: bool = False):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # Add & Norm 1
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # Add & Norm 2
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x (B, L_src, d_model)
        # mask (B, 1, 1, L_src)
        
        # 1. Multi-Head Attention (Sublayer 1)
        _x = x
        attn_output = self.self_attn(Q=x, K=x, V=x, mask=mask)
        
        # 2. Add & Norm 1
        if config.USE_RESIDUALS:
            x = _x + self.dropout1(attn_output)
        else:
            x = self.dropout1(attn_output)
        x = self.norm1(x)
        
        # 3. Feed Forward Network (Sublayer 2)
        _x = x
        ffn_output = self.ffn(x)
        
        # 4. Add & Norm 2
        if config.USE_RESIDUALS:
            x = _x + self.dropout2(ffn_output)
        else:
            x = self.dropout2(ffn_output)
        x = self.norm2(x)
        
        return x


class DecoderLayer(nn.Module):
    """
    实现单个 Decoder 层
    采用 Post-Norm 结构: x + Sublayer(x) -> LayerNorm
    支持自回归掩码和编码器-解码器注意力
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, use_relative_pos: bool = False):
        super().__init__()
        # 自注意力机制 (带掩码)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout, use_relative_pos)
        # 编码器-解码器注意力机制
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout, use_relative_pos)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # Add & Norm 1 (自注意力)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # Add & Norm 2 (编码器-解码器注意力)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
        # Add & Norm 3 (前馈网络)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, 
                x: torch.Tensor, 
                enc_output: torch.Tensor, 
                src_mask: torch.Tensor = None, 
                tgt_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: 解码器输入 (B, L_tgt, d_model)
            enc_output: 编码器输出 (B, L_src, d_model)
            src_mask: 源序列掩码 (B, 1, 1, L_src)
            tgt_mask: 目标序列掩码 (B, 1, L_tgt, L_tgt)
        """
        # 1. 掩码自注意力 (Sublayer 1)
        _x = x
        attn_output = self.self_attn(Q=x, K=x, V=x, mask=tgt_mask)
        
        # 2. Add & Norm 1
        if config.USE_RESIDUALS:
            x = _x + self.dropout1(attn_output)
        else:
            x = self.dropout1(attn_output)
        x = self.norm1(x)
        
        # 3. 编码器-解码器交叉注意力 (Sublayer 2)
        _x = x
        cross_attn_output = self.cross_attn(Q=x, K=enc_output, V=enc_output, mask=src_mask)
        
        # 4. Add & Norm 2
        if config.USE_RESIDUALS:
            x = _x + self.dropout2(cross_attn_output)
        else:
            x = self.dropout2(cross_attn_output)
        x = self.norm2(x)
        
        # 5. 前馈网络 (Sublayer 3)
        _x = x
        ffn_output = self.ffn(x)
        
        # 6. Add & Norm 3
        if config.USE_RESIDUALS:
            x = _x + self.dropout3(ffn_output)
        else:
            x = self.dropout3(ffn_output)
        x = self.norm3(x)
        
        return x