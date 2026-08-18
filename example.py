import torch
from core.shr_attention import SHRAttention
from layers.shr_encoder_layer import SHREncoderLayer


def basic_usage():
    """基础SHR注意力调用示例"""
    print("=== 基础SHR注意力调用 ===")
    batch, seq_len, embed_dim = 2, 32, 128
    num_heads = 4
    
    # 初始化
    attn = SHRAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        nest_depth=4,
        crop_threshold=0.15,
        batch_first=True
    )
    
    # 前向传播
    x = torch.randn(batch, seq_len, embed_dim)
    output = attn(x, x, x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print("✅ 基础调用完成\n")


def encoder_usage():
    """SHR编码器层使用示例"""
    print("=== SHR编码器层使用 ===")
    batch, seq_len, d_model = 2, 32, 128
    
    encoder_layer = SHREncoderLayer(
        d_model=d_model,
        nhead=4,
        dim_feedforward=256,
        nest_depth=3
    )
    
    x = torch.randn(batch, seq_len, d_model)
    output = encoder_layer(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print("✅ 编码器层调用完成\n")


def dynamic_depth():
    """动态调整嵌套深度示例"""
    print("=== 动态调整嵌套深度 ===")
    attn = SHRAttention(embed_dim=64, num_heads=2, nest_depth=2)
    x = torch.randn(1, 16, 64)
    
    out1 = attn(x, x, x)
    print(f"深度2输出形状: {out1.shape}")
    
    attn.set_nest_depth(5)
    out2 = attn(x, x, x)
    print(f"深度5输出形状: {out2.shape}")
    print("✅ 动态深度调整完成\n")


if __name__ == "__main__":
    basic_usage()
    encoder_usage()
    dynamic_depth()
    print("🎉 所有示例运行完成")
