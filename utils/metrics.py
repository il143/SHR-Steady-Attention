import torch
import time


def compute_relative_error(pred, target):
    """计算预测结果与基准真值的平均相对误差"""
    diff = torch.abs(pred - target)
    norm = torch.abs(target) + 1e-12
    return (diff / norm).mean().item()


def measure_inference_time(model, input_tensor, rounds=100):
    """
    测量模型单步平均推理耗时
    Args:
        model: 待测试模型
        input_tensor: 输入张量
        rounds: 测试重复次数
    Returns:
        float: 单步推理耗时，单位毫秒
    """
    # 预热，消除冷启动误差
    for _ in range(10):
        _ = model(input_tensor)
    
    # CUDA 同步后正式计时
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(rounds):
        _ = model(input_tensor)
    torch.cuda.synchronize()
    total = time.time() - start
    
    return total / rounds * 1000


def compute_output_consistency(model, input_tensor, rounds=1000):
    """
    计算输出一致性，统计多次推理结果的最大相对误差
    Args:
        model: 待测试模型
        input_tensor: 固定输入张量
        rounds: 重复推理次数
    Returns:
        float: 多次推理的最大相对误差
    """
    outputs = []
    for _ in range(rounds):
        out = model(input_tensor)
        outputs.append(out.detach())
    
    outputs = torch.stack(outputs, dim=0)
    mean_out = outputs.mean(dim=0)
    max_diff = torch.max(torch.abs(outputs - mean_out))
    max_rel_error = (max_diff / (torch.abs(mean_out).mean() + 1e-12)).item()
    
    return max_rel_error


def compute_efficiency_coefficient(error_base, error_test, time_base, time_test):
    """
    计算精度收敛效率系数
    效率系数 = 精度提升倍数 / 算力增长倍数
    数值越高，代表单位算力投入带来的精度收益越高
    Args:
        error_base: 基准方案的相对误差
        error_test: 测试方案的相对误差
        time_base: 基准方案的单步耗时
        time_test: 测试方案的单步耗时
    Returns:
        float: 效率系数
    """
    precision_gain = error_base / error_test  # 误差越小精度越高，对应精度提升倍数越大
    time_gain = time_test / time_base
    return precision_gain / time_gain
