import torch.nn as nn
from core import SHRAttention


class SHRCrossAttention(nn.Module):
    """独立交叉注意力模块，用于多模态、扩散模型等场景"""
    def __init__(
        self,
        d_model,
        nhead,
        mode="balanced",
        rigid_R=True,
        batch_first=True,
        dropout=0.1,
    ):
        super().__init__()
        self.attn = SHRAttention(
            d_model, nhead, mode=mode, rigid_R=rigid_R,
            batch_first=batch_first, dropout=dropout
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value, attn_mask=None, key_padding_mask=None):
        out, _ = self.attn(query, key_value, key_value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        out = query + self.dropout(out)
        out = self.norm(out)
        return out
