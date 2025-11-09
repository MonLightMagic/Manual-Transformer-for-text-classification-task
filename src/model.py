# src/model.py
import torch
import torch.nn as nn
import math
from src.layers import EncoderLayer
from src.modules import PositionalEncoding
from src import config

class TextClassifier(nn.Module):
    """
    基于Transformer编码器的文本分类模型
    """
    def __init__(self, 
                 vocab_size: int, 
                 d_model: int, 
                 n_layers: int, 
                 n_heads: int, 
                 d_ff: int, 
                 dropout: float, 
                 max_seq_len: int, 
                 num_classes: int):
        super().__init__()
        self.d_model = d_model
        
        # 词嵌入层
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码层
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_seq_len)
        
        # 编码器层堆叠
        # 根据配置决定是否使用相对位置编码
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                d_model, 
                n_heads, 
                d_ff, 
                dropout,
                use_relative_pos=config.USE_RELATIVE_POSITION_ENCODING
            ) for _ in range(n_layers)
        ])
        
        # 最终的 LayerNorm
        self.norm = nn.LayerNorm(d_model)
        
        # 分类头
        self.classifier = nn.Linear(d_model, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """
        初始化模型权重
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: (B, L) 输入序列的token id
            attention_mask: (B, L) 注意力掩码，1表示有效token，0表示填充token
        
        Returns:
            logits: (B, num_classes) 分类的logits
        """
        # 1. 词嵌入 + 位置编码
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        
        # 如果使用相对位置编码，则不添加绝对位置编码
        # 相对位置编码已经在注意力计算中处理了位置信息
        if config.USE_POSITIONAL_ENCODING and not config.USE_RELATIVE_POSITION_ENCODING:
            x = self.pos_encoding(x)
        
        # 2. 构建注意力掩码
        # (B, L) -> (B, 1, 1, L)
        src_mask = attention_mask.unsqueeze(1).unsqueeze(2).bool()
        src_mask = ~src_mask  # 转换为True表示需要mask的位置
        
        # 3. 通过编码器层
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        
        # 4. 最终的LayerNorm
        x = self.norm(x)
        
        # 5. 使用[CLS]标记的输出进行分类（这里使用序列的第一个token作为表示）
        # 也可以使用平均池化或最大池化
        cls_output = x[:, 0, :]  # (B, d_model)
        
        # 6. 分类头
        logits = self.classifier(cls_output)  # (B, num_classes)
        
        return logits