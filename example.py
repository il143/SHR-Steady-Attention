import torch
from core.shr_attention import SHRAttention
from layers.shr_encoder_layer import SHREncoderLayer


def basic_usage():
    """
    基础调用示例：最常用的单注意力算子调用方式
    演示：初始化算子 → 生成随机输入 → 执行前向计算 → 查看输出
    适用场景：快速验证算子可用性、替换项目中原生注意力
    """
    print("=== 示例1：基础SHR注意力调用 ===")

    # ========== 1. 定义输入维度参数 ==========
    # batch：批次大小，一次处理多少条样本
    # seq_len：序列长度，即输入文本的token数量
    # embed_dim：嵌入维度，每个token的特征向量长度
    batch = 2
    seq_len = 32
    embed_dim = 128
    # 注意力头数，并行计算的分支数量，需能整除 embed_dim
    num_heads = 4

    # ========== 2. 初始化SHR注意力算子 ==========
    # mode 为公开版固定精度档位，三选一：
    #   basic    : 轻量档，速度最快，精度基础，适合短文本高并发场景
    #   balanced : 均衡档（默认），速度与效果平衡，通用场景推荐
    #   long_seq : 长序列优档，长文本下稳定性更好，算力开销略高
    attn = SHRAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        mode="balanced",   # 公开版仅支持档位切换，细粒度参数需商业授权
        batch_first=True   # 输入格式为 [batch, seq_len, embed_dim]
    )

    # ========== 3. 生成模拟输入并执行计算 ==========
    # 生成形状匹配的随机张量，模拟真实模型输入
    x = torch.randn(batch, seq_len, embed_dim)
    # 执行注意力前向计算，接口与原生对齐，返回 (输出张量, 注意力权重)
    output, _ = attn(x, x, x)  # 三个参数分别对应 query、key、value

    # ========== 4. 打印结果验证 ==========
    print(f"输入张量形状: {x.shape}  [批次, 序列长度, 嵌入维度]")
    print(f"输出张量形状: {output.shape}  与输入完全对齐，可无缝替换原生注意力")
    print("✅ 基础调用验证通过\n")


def encoder_usage():
    """
    编码器层调用示例：完整Transformer编码器层的替换方式
    适用场景：直接替换BERT等编码器类模型的原生编码器层
    """
    print("=== 示例2：SHR编码器层使用 ===")

    # d_model：模型整体特征维度
    # dim_feedforward：前馈网络的隐藏层维度
    batch = 2
    seq_len = 32
    d_model = 128
    dim_feedforward = 256

    # 初始化完整编码器层，内部已封装好注意力+前馈网络+层归一化
    encoder_layer = SHREncoderLayer(
        d_model=d_model,
        nhead=4,
        dim_feedforward=dim_feedforward,
        mode="balanced"
    )

    # 生成输入并执行前向计算
    x = torch.randn(batch, seq_len, d_model)
    output = encoder_layer(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print("✅ 编码器层调用验证通过\n")


def mode_switch_demo():
    """
    精度档位切换示例：演示运行中切换不同效果档位
    适用场景：对比不同档位的速度与效果差异，选择最适合业务的档位
    """
    print("=== 示例3：精度档位切换演示 ===")

    # 初始化，默认使用轻量档
    attn = SHRAttention(embed_dim=64, num_heads=2, mode="basic")
    x = torch.randn(1, 16, 64)

    # 基础档计算
    out_basic, _ = attn(x, x, x)
    print(f"【basic 轻量档】输出形状: {out_basic.shape}，速度优先")

    # 切换为长序列优化档，无需重新初始化
    attn.set_mode("long_seq")
    out_long, _ = attn(x, x, x)
    print(f"【long_seq 长序列优档】输出形状: {out_long.shape}，稳定性优先")

    # 切换回均衡档
    attn.set_mode("balanced")
    out_balanced, _ = attn(x, x, x)
    print(f"【balanced 均衡档】输出形状: {out_balanced.shape}，通用推荐")

    print("✅ 三档切换验证通过\n")


if __name__ == "__main__":
    # 依次执行三个示例，全部通过即代表环境与算子功能正常
    basic_usage()
    encoder_usage()
    mode_switch_demo()
    print("=" * 40)
    print("🎉 所有公开版示例运行完成！")
    print("💡 如需自定义嵌套深度、裁剪阈值等细粒度参数，请申请商业授权版本")
    print("=" * 40)
