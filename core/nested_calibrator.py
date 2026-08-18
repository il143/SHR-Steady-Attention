import torch
import torch.nn as nn
from .rigid_constraint import clamp_weights, compute_R_from_weights


class NestedCalibrator(nn.Module):
    """
    纵向嵌套校准器：对注意力权重进行迭代收敛校准，同步执行分支裁剪
    """
    def __init__(self, max_depth=4, crop_threshold=0.15, step_alpha=0.3):
        super().__init__()
        self.max_depth = max_depth
        self.crop_threshold = crop_threshold
        self.step_alpha = step_alpha

    def forward(self, weights):
        """
        输入初始权重分布，输出校准后的稳态权重
        weights形状: [batch, heads, seq_q, seq_k]
        """
        batch, heads, seq_q, seq_k = weights.shape
        device = weights.device
        
        # 初始归一化
        weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = clamp_weights(weights)
        
        # 计算全局稳态基准R
        R = compute_R_from_weights(weights)  # [batch, heads, seq_q, 1]
        
        # 初始化未收敛掩码：True表示未收敛，继续迭代
        active_mask = torch.ones_like(weights, dtype=torch.bool, device=device)
        
        current_weights = weights.clone()
        
        for _ in range(self.max_depth):
            # 仅对未收敛分支计算偏差
            deviation = torch.abs(current_weights - 0.5)
            drive = R - deviation  # 校准驱动力
            
            # 校准增量，仅作用于未收敛分支
            delta = self.step_alpha * drive * active_mask.float()
            current_weights = current_weights + delta
            
            # 重新归一化并施加刚性约束
            current_weights = current_weights / current_weights.sum(dim=-1, keepdim=True)
            current_weights = clamp_weights(current_weights)
            
            # 更新收敛掩码：偏差小于阈值则标记为已收敛
            new_deviation = torch.abs(current_weights - R)
            converged = new_deviation < self.crop_threshold
            active_mask = active_mask & (~converged)
            
            # 全部收敛则提前终止
            if not active_mask.any():
                break
        
        return current_weights
