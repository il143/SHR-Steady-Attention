import torch
import torch.nn as nn
from core.shr_attention import SHRAttention


class SHRCrossAttention(nn.Module):
    """
    独立封装的SHR交叉注意力模块，用于编码器-解码器架构
    """
    def __init__(
        self,
        d_model,
        nhead,
        nest_depth=4,
        crop_threshold=0.15,
        dropout=0.0,
        batch_first=True
    ):
        super().__init__()
        self.attn = SHRAttention(
            embed_dim=d_model,
            num_heads=nhead,
            nest_depth=nest_depth,
            crop_threshold=crop_threshold,
            dropout=dropout,
            batch_first=batch_first
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value, attn_mask=None):
        """
        query: 查询端张量
        key_value: 键值端张量（编码器输出等）
        """
        residual = query
        output = self.attn(query, key_value, key_value, attn_mask=attn_mask)
        output = self.dropout(output)
        output = self.norm(output + residual)
        return output
