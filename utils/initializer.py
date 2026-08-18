import torch
import torch.nn as nn


def init_shr_weights(module):
    """
    SHR模块参数初始化，沿用Xavier初始化保证初始稳定性
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def generate_causal_mask(seq_len, device='cpu'):
    """生成因果注意力掩码（下三角掩码）"""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    mask = mask.masked_fill(mask == 0, float(0.0))
    return mask
