import torch
import torch.nn as nn
from .rigid_constraint import compute_R_from_weights


class SteadyCoupler(nn.Module):
    """
    多分支稳态融合器：按各分支的R值加权融合，保证输出量级稳定
    """
    def __init__(self):
        super().__init__()

    def forward(self, branch_outputs, branch_weights):
        """
        branch_outputs: 各分支输出value张量，形状 [batch, heads, seq, dim] * N
        branch_weights: 各分支的注意力权重，用于计算R值
        返回融合后的输出，形状与单分支一致
        """
        num_branches = len(branch_outputs)
        if num_branches == 1:
            return branch_outputs[0]
        
        # 计算每个分支的稳态强度R
        R_list = []
        for w in branch_weights:
            R = compute_R_from_weights(w)
            R_list.append(R.mean(dim=(-1, -2), keepdim=True))  # 全局R标量
        
        # 拼接并归一化融合权重
        R_tensor = torch.cat(R_list, dim=-1)  # [batch, 1, 1, N]
        fusion_weights = R_tensor / R_tensor.sum(dim=-1, keepdim=True)
        
        # 加权融合
        outputs_stack = torch.stack(branch_outputs, dim=-1)
        fused = torch.sum(outputs_stack * fusion_weights.unsqueeze(-2), dim=-1)
        
        return fused
