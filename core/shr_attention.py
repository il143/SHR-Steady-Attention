import torch
import torch.nn as nn
import torch.nn.functional as F
from .nested_calibrator import NestedCalibrator
from .steady_coupler import SteadyCoupler
from .rigid_constraint import clamp_weights


class SHRAttention(nn.Module):
    """
    SHR稳态注意力算子，一对一兼容PyTorch MultiheadAttention接口
    """
    def __init__(
        self,
        embed_dim,
        num_heads=8,
        nest_depth=4,
        crop_threshold=0.15,
        rigid_R=True,
        dropout=0.0,
        bias=True,
        batch_first=True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.nest_depth = nest_depth
        self.crop_threshold = crop_threshold
        self.rigid_R = rigid_R
        self.dropout = dropout
        self.batch_first = batch_first
        
        assert embed_dim % num_heads == 0, "embed_dim必须能被num_heads整除"
        
        # QKV投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # 输出投影
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # 嵌套校准器
        self.calibrator = NestedCalibrator(
            max_depth=nest_depth,
            crop_threshold=crop_threshold
        )
        
        # 多分支融合器（多头复用）
        self.coupler = SteadyCoupler()
        
        # Dropout层
        self.attn_dropout = nn.Dropout(dropout)
        
        # 缓存注意力权重
        self._attention_weights = None

    def _reshape_to_heads(self, x):
        """将最后一维拆分为多头"""
        batch, seq, _ = x.shape
        x = x.view(batch, seq, self.num_heads, self.head_dim)
        return x.transpose(1, 2)  # [batch, heads, seq, head_dim]

    def _reshape_from_heads(self, x):
        """多头合并回原始维度"""
        batch, heads, seq, head_dim = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq, self.embed_dim)

    def forward(self, query, key, value, attn_mask=None):
        """
        前向传播，接口与nn.MultiheadAttention对齐
        query/key/value形状: [batch, seq, embed_dim] (batch_first=True)
        """
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        
        batch, seq_q, _ = query.shape
        seq_k = key.shape[1]
        
        # 1. QKV投影
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # 2. 拆分为多头
        q = self._reshape_to_heads(q)
        k = self._reshape_to_heads(k)
        v = self._reshape_to_heads(v)
        
        # 3. 计算初始注意力分数
        scale = self.head_dim ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # 4. 应用注意力掩码（如因果掩码）
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask
        
        # 5. 初始归一化
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 6. SHR嵌套校准
        attn_weights = self.calibrator(attn_weights)
        
        # 7. 刚性约束
        if self.rigid_R:
            attn_weights = clamp_weights(attn_weights)
        
        # 8. Dropout
        attn_weights = self.attn_dropout(attn_weights)
        
        # 9. 加权求和
        output = torch.matmul(attn_weights, v)
        
        # 10. 合并多头
        output = self._reshape_from_heads(output)
        
        # 11. 输出投影
        output = self.out_proj(output)
        
        # 缓存权重
        self._attention_weights = attn_weights.detach()
        
        if not self.batch_first:
            output = output.transpose(0, 1)
        
        return output

    def get_attention_weights(self):
        """获取当前计算的注意力权重分布"""
        return self._attention_weights

    def set_nest_depth(self, depth):
        """动态修改嵌套深度"""
        self.nest_depth = depth
        self.calibrator.max_depth = depth
