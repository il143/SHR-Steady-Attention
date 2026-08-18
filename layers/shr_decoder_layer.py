import torch
import torch.nn as nn
import torch.nn.functional as F
from core.shr_attention import SHRAttention


class SHRDecoderLayer(nn.Module):
    """
    SHR解码器层，兼容标准TransformerDecoderLayer接口
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
        # 自注意力（因果掩码）
        self.self_attn = SHRAttention(
            embed_dim=d_model,
            num_heads=nhead,
            nest_depth=nest_depth,
            crop_threshold=crop_threshold,
            dropout=dropout,
            batch_first=batch_first
        )
        # 交叉注意力
        self.multihead_attn = SHRAttention(
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
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = activation

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """前向传播"""
        # 自注意力子层
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # 交叉注意力子层
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # 前馈网络子层
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt
