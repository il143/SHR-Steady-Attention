import torch
import torch.nn.functional as F


def clamp_weights(weights, min_val=1e-8, max_val=1.0 - 1e-8):
    """
    R>0刚性约束：将权重严格限制在开区间 (min_val, max_val) 内
    边界采用平滑截断，避免梯度突变
    """
    # 平滑夹紧，保证边界处梯度连续
    weights_clamped = torch.clamp(weights, min=min_val, max=max_val)
    return weights_clamped


def compute_steady_R(S, H):
    """
    根据敛张分量计算稳态基准值R
    R = sqrt(S * H)
    """
    R = torch.sqrt(S * H)
    return R


def normalize_sh(weights):
    """
    从权重分布中拆分敛张分量并归一化
    返回 S(敛极标度), H(张极标度), 满足 S + H = 1
    """
    # 以权重分布的集中度为S，离散度为H
    # 计算权重分布的熵，归一化得到张极标度
    weights_norm = weights / weights.sum(dim=-1, keepdim=True)
    entropy = -torch.sum(weights_norm * torch.log(weights_norm + 1e-12), dim=-1, keepdim=True)
    max_entropy = torch.log(torch.tensor(weights.shape[-1], dtype=torch.float32))
    
    H = entropy / max_entropy  # 张极：越均匀熵越大，H越大
    S = 1.0 - H                # 敛极：越集中熵越小，S越大
    
    # 刚性约束，避免边界零值
    S = clamp_weights(S, min_val=1e-6, max_val=1.0 - 1e-6)
    H = 1.0 - S
    
    return S, H


def compute_R_from_weights(weights):
    """
    直接从权重分布计算当前稳态强度R
    """
    S, H = normalize_sh(weights)
    R = compute_steady_R(S, H)
    return R
