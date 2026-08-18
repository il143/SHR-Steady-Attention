import torch
import time


def compute_relative_error(pred, target):
    """计算相对误差"""
    diff = torch.abs(pred - target)
    norm = torch.abs(target) + 1e-12
    return (diff / norm).mean().item()


def measure_inference_time(model, input_tensor, rounds=100):
    """测量平均推理耗时"""
    # 预热
    for _ in range(10):
        _ = model(input_tensor)
    
    # 正式计时
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(rounds):
        _ = model(input_tensor)
    torch.cuda.synchronize()
    total = time.time() - start
    
    return total / rounds * 1000  # 返回毫秒


def compute_output_consistency(model, input_tensor, rounds=1000):
    """计算输出一致性（千次推理最大相对误差）"""
    outputs = []
    for _ in range(rounds):
        out = model(input_tensor)
        outputs.append(out.detach())
    
    outputs = torch.stack(outputs, dim=0)
    mean_out = outputs.mean(dim=0)
    max_diff = torch.max(torch.abs(outputs - mean_out))
    max_rel_error = (max_diff / (torch.abs(mean_out).mean() + 1e-12)).item()
    
    return max_rel_error


def compute_efficiency_coefficient(error_0, error_1, time_0, time_1):
    """
    计算效率系数
    效率系数 = (精度提升倍数) / (算力增长倍数)
    """
    precision_gain = error_0 / error_1  # 误差越小精度越高
    time_gain = time_1 / time_0
    return precision_gain / time_gain
