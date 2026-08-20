import torch.nn as nn
from core import SHRAttention


class SHREncoderLayer(nn.Module):
    """SHR 编码器层，接口兼容 PyTorch 原生 TransformerEncoderLayer"""
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        mode="balanced",
        rigid_R=True,
        batch_first=True,
    ):
        super().__init__()
        self.self_attn = SHRAttention(
            d_model, nhead, mode=mode, rigid_R=rigid_R,
            batch_first=batch_first, dropout=dropout
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
