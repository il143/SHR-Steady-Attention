```
# SHR 稳态注意力算子 V1.0
**R>0 刚性约束的低波动 Transformer 注意力算子，原生无缝替换 Softmax 注意力，从机制上规避数值奇点、显著降低推理输出波动**

## 项目简介
SHR（Steady-Hold-Rigid）是一套数值稳定型 Transformer 注意力算子实现。它通过刚性非零约束重构注意力归一化逻辑，配合嵌套校准机制，从根源削弱极端输入下的 NaN、数值溢出与输出随机抖动，在不改变原有网络接口的前提下，显著提升模型推理的确定性与边界鲁棒性。

## 核心特性
- ✅ **高确定性推理**：输出结果高度可复现，显著降低随机数值误差与浮点抖动
- ✅ **边界鲁棒**：从机制上规避零值奇点，极端输入下不易出现数值发散与 NaN 崩溃
- ✅ **无缝替换**：接口兼容 PyTorch 原生 MultiheadAttention，原有代码零修改即可替换
- ✅ **弹性精度**：支持多档位精度配置，按需平衡计算效果与算力开销
- ✅ **高效收敛**：嵌套校准机制的精度收敛效率优于原生层数堆叠

## 环境要求
- Python >= 3.10
- PyTorch >= 2.0.0
- CUDA >= 11.8（可选，用于 GPU 加速）

## 快速开始
### 1. 安装依赖
```

pip install -r requirements.txt

```

### 2. 运行示例
```

python example.py

```

### 3. 基础调用（公开基础版，固定档位）
```python
import torch
from core.shr_attention import SHRAttention

# 初始化算子，mode 支持 3 档：basic / balanced / long_seq
attn = SHRAttention(embed_dim=512, num_heads=8, mode="balanced")

# 前向计算
x = torch.randn(2, 128, 512)
output = attn(x, x, x)
```

### 4. 替换原生注意力

仅需修改导入语句，原有业务代码完全不变：

```
# 原导入
# from torch.nn import MultiheadAttention
# 替换为
from core.shr_attention import SHRAttention as MultiheadAttention
```

> 
> 注：公开基础版仅提供 3 档固定精度模式；完整嵌套深度、裁剪阈值等参数自定义能力，需申请商业授权版本。

## 目录结构

```
SHR-Steady-Attention/
├── core/                  # 核心算子模块
│   ├── shr_attention.py   # SHR 稳态注意力核心实现
│   ├── nested_calibrator.py  # 嵌套校准与分支裁剪单元
│   ├── steady_coupler.py  # 多分支稳态融合单元
│   └── rigid_constraint.py # R>0 刚性约束模块
├── layers/                # 网络层封装
│   ├── shr_encoder_layer.py
│   ├── shr_decoder_layer.py
│   └── shr_cross_attention.py
├── utils/                 # 工具函数
│   ├── metrics.py
│   └── initializer.py
├── test/                  # 测试套件
│   ├── test_attention_base.py
│   ├── test_performance.py
│   └── test_convergence.py
├── example.py             # 快速运行示例
├── requirements.txt       # 依赖声明
├── LICENSE
└── README.md
```

## 核心参数说明（公开基础版）

表格

| 参数名 | 默认值 | 说明 |
| --- | --- | --- |
| embed_dim | - | 输入嵌入维度 |
| num_heads | 8 | 注意力并行头数 |
| mode | "balanced" | 精度档位：basic（轻量）/balanced（均衡）/long_seq（长序列优） |
| rigid_R | True | 是否开启 R>0 刚性约束 |
| batch_first | True | 输入是否 batch 维度在前 |

> 
> 完整细粒度参数（嵌套校准深度、收敛裁剪阈值等）仅在商业授权版开放。

## 运行测试

```
# 基础功能测试
python test/test_attention_base.py

# 性能对标测试
python test/test_performance.py

# 收敛性验证测试
python test/test_convergence.py
```

## 版本说明

### V1.0

- 完整实现 SHR 稳态注意力核心机制
- 支持编码器、解码器、交叉注意力全场景封装
- 提供功能、性能、收敛性三类完整测试套件
- 100% 兼容标准 Transformer 生态接口

## 版权说明

本技术为原创成果，已完成可信时间戳存证。
学术研究可自由使用，商用请联系授权。