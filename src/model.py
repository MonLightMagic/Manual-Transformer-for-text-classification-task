# src/model.py
import torch
import torch.nn as nn
import math
from src.layers import EncoderLayer, DecoderLayer
from src.modules import PositionalEncoding
from src import config

class TextClassifier(nn.Module):
    """
    基于Transformer架构的文本分类器
    支持编码器-解码器结构和相对位置编码
    """
    def __init__(self, 
                 vocab_size: int, 
                 d_model: int, 
                 n_layers: int, 
                 n_heads: int, 
                 d_ff: int, 
                 dropout: float, 
                 max_seq_len: int, 
                 num_classes: int,
                 use_decoder: bool = None,
                 n_decoder_layers: int = None):
        super().__init__()
        self.d_model = d_model
        
        # 从配置中读取解码器相关参数，如果参数未提供
        self.use_decoder = config.USE_DECODER if use_decoder is None else use_decoder
        
        # 如果未指定解码器层数，则从配置读取或使用与编码器相同的层数
        if n_decoder_layers is not None:
            self.n_decoder_layers = n_decoder_layers
        elif hasattr(config, 'N_DECODER_LAYERS') and config.N_DECODER_LAYERS is not None:
            self.n_decoder_layers = config.N_DECODER_LAYERS
        else:
            self.n_decoder_layers = n_layers
        
        # 词嵌入层
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码层
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_seq_len)
        
        # 编码器层堆叠
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                d_model, 
                n_heads, 
                d_ff, 
                dropout,
                use_relative_pos=config.USE_RELATIVE_POSITION_ENCODING
            ) for _ in range(n_layers)
        ])
        
        # 解码器层堆叠 (如果启用)
        if use_decoder:
            self.decoder_layers = nn.ModuleList([
                DecoderLayer(
                    d_model,
                    n_heads,
                    d_ff,
                    dropout,
                    use_relative_pos=config.USE_RELATIVE_POSITION_ENCODING
                ) for _ in range(self.n_decoder_layers)
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
    
    def _create_autoregressive_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        创建自回归掩码，确保位置i只能关注i及之前的位置
        """
        # 创建上三角矩阵 (seq_len, seq_len)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        # 转换形状为 (1, 1, seq_len, seq_len)
        return mask.unsqueeze(0).unsqueeze(0)

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
        if config.USE_POSITIONAL_ENCODING and not config.USE_RELATIVE_POSITION_ENCODING:
            x = self.pos_encoding(x)
        
        # 2. 构建注意力掩码
        # (B, L) -> (B, 1, 1, L)
        src_mask = attention_mask.unsqueeze(1).unsqueeze(2).bool()
        src_mask = ~src_mask  # 转换为True表示需要mask的位置
        
        # 3. 通过编码器层
        enc_output = x
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)
        
        # 4. 通过解码器层 (如果启用)
        if self.use_decoder:
            # 对于文本分类任务，我们使用编码器的输出作为解码器的输入
            dec_output = enc_output
            
            # 创建自回归掩码
            batch_size, seq_len, _ = dec_output.size()
            tgt_mask = self._create_autoregressive_mask(seq_len, dec_output.device)
            
            # 解码器层处理
            for layer in self.decoder_layers:
                dec_output = layer(dec_output, enc_output, tgt_mask, src_mask)
            
            # 使用解码器输出进行分类
            x = dec_output
        else:
            # 使用编码器输出进行分类
            x = enc_output
        
        # 5. 最终的LayerNorm
        x = self.norm(x)
        
        # 6. 使用[CLS]标记的输出进行分类
        cls_output = x[:, 0, :]  # (B, d_model)
        
        # 7. 分类头
        logits = self.classifier(cls_output)  # (B, num_classes)
        
        return logits