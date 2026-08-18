import torch
import torch.nn as nn
import torch.nn.functional as F
from core.shr_attention import SHRAttention


class SHREncoderLayer(nn.Module):
    """
    SHR编码器层，兼容标准TransformerEncoderLayer接口
    """
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.relu,
        layer_norm_eps=1e-5,
        nest_depth=4,
        crop_threshold=0.15,
        batch_first=True
    ):
        super().__init__()
        self.self_attn = SHRAttention(
            embed_dim=d_model,
            num_heads=nhead,
            nest_depth=nest_depth,
            crop_threshold=crop_threshold,
            dropout=dropout,
            batch_first=batch_first
        )
        
        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = activation

    def forward(self, src, src_mask=None):
        """前向传播"""
        # 自注意力子层
        src2 = self.self_attn(src, src, src, attn_mask=src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # 前馈网络子层
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src
