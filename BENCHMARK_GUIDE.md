# GPU 基准测试与对比实验 - 使用指南

## 📋 概述

本基准测试框架用于在 **MNIST 和 CIFAR-10** 数据集上对比以下模型：
- 🎯 **三生架构4层**（Sansheng 4-Layer）
- 📊 **MLP**（多层感知机）
- 🖼️ **CNN**（卷积神经网络）
- 🔗 **张量网络**（标准张量网络分类器）

## 🚀 快速开始

### 1️⃣ 环装依赖

```bash
# 安装PyTorch（GPU版本，以Linux为例）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 或者CPU版本
pip install torch torchvision

# 安装其他依赖
pip install numpy matplotlib seaborn scikit-learn
```

### 2️⃣ 运行基准测试

```bash
# 方式1：运行完整实验（所有模型+所有数据集）
python benchmark_experiment.py

# 方式2：快速测试（仅MNIST，5个epoch）
python benchmark_experiment.py --dataset mnist --epochs 5 --quick

# 方式3：仅测试特定模型
python benchmark_experiment.py --model sansheng --dataset mnist
```

### 3️⃣ 分析结果

```bash
# 生成详细分析报告和图表
python analyze_benchmark.py

# 输出文件：
# - benchmark_text_report.txt (文本报告)
# - benchmark_report.md (Markdown报告)
# - accuracy_comparison.png (准确率对比图)
# - param_efficiency.png (参数效率图)
# - speed_comparison.png (训练速度对比图)
```

## 📊 文件说明

### 核心文件

| 文件 | 功能 | 说明 |
|------|------|------|
| `benchmark_experiment.py` | 基准测试 | 训练所有模型，生成 `benchmark_results.json` |
| `analyze_benchmark.py` | 结果分析 | 生成报告和可视化 |
| `benchmark_results.json` | 结果数据 | 存储所有实验结果 |

### 输出文件

运行后会生成：
- `benchmark_results.json` - 原始结果数据
- `benchmark_text_report.txt` - 文本格式报告
- `benchmark_report.md` - Markdown格式报告
- `accuracy_comparison.png` - 准确率对比图
- `param_efficiency.png` - 参数效率图
- `speed_comparison.png` - 训练速度对比图

## 🎯 预期结果

### MNIST 数据集

| 模型 | 预期准确率 | 参数数量 | 训练时间 |
|------|-----------|---------|---------|
| 三生架构4层 | 97-98% | ~75K | 15-20s |
| MLP | 97-98% | ~130K | 10-15s |
| CNN | 99%+ | ~50K | 20-30s |
| 张量网络 | 90-95% | ~100K | 30-40s |

### CIFAR-10 数据集

| 模型 | 预期准确率 | 参数数量 | 训练时间 |
|------|-----------|---------|---------|
| 三生架构4层 | 65-70% | ~75K | 30-40s |
| MLP | 60-65% | ~130K | 20-30s |
| CNN | 80-85% | ~50K | 40-60s |
| 张量网络 | 50-60% | ~100K | 60-90s |

## 💻 系统要求

### 硬件
- **CPU**: 现代多核处理器（推荐）
- **GPU**: NVIDIA GPU with CUDA（推荐用于加速）
- **内存**: 至少 4GB RAM
- **存储**: 至少 1GB 用于数据集

### 软件
- Python >= 3.8
- PyTorch >= 1.9
- NumPy, Matplotlib, Seaborn

### CUDA（可选但推荐）
```bash
# 检查CUDA可用性
python -c "import torch; print(torch.cuda.is_available())"

# 如果返回False，安装CPU版本也可以，但速度较慢
```

## 📈 自定义配置

编辑 `benchmark_experiment.py` 可修改：

```python
# 数据集
datasets = ['mnist', 'cifar10']  # 或只选择一个

# 训练参数
batch_size = 128
epochs = 20
learning_rate = 0.001

# 模型隐层大小
hidden_dim = 128  # 调整此参数改变模型复杂度
```

## 🔍 理解输出

### JSON 结果格式

```json
{
  "mnist": {
    "Sansheng-4Layer": {
      "name": "三生架构4层",
      "param_count": 75000,
      "train_loss": 0.05,
      "train_acc": 98.5,
      "val_loss": 0.08,
      "val_acc": 98.2,
      "test_loss": 0.09,
      "test_acc": 98.1,
      "train_time": 18.5
    },
    ...
  }
}
```

### 报告包含内容

文本报告中的关键指标：
- **准确率** - 测试集准确率（%）
- **参数数** - 模型参数总数
- **训练时间** - 完整训练所用时间（秒）
- **对比分析** - 与其他模型的性能对比

## 🎨 可视化说明

### 准确率对比
显示各模型在测试集上的准确率，三生架构用红色菱形标记。

### 参数效率
显示准确率 vs 参数数量的关系，左下角表示"又小又准"。

### 训练速度
显示各模型的训练时间，反映计算效率。

## ⚠️ 常见问题

### Q1: 运行时间太长怎么办？
```bash
# 方案1：减少epoch
python benchmark_experiment.py --epochs 5

# 方案2：减小batch_size
# 编辑benchmark_experiment.py，改为 batch_size = 64

# 方案3：只测MNIST
python benchmark_experiment.py --dataset mnist
```

### Q2: 没有GPU怎么办？
代码会自动使用CPU，但速度会慢10-50倍。建议：
- 在云平台（如Google Colab）上运行
- 减少epoch和batch_size
- 只测试MNIST

### Q3: 显存不足（OOM）怎么办？
```python
# 编辑 benchmark_experiment.py，减小batch_size
batch_size = 64  # 改为 32 或 16
```

### Q4: 结果不稳定怎么办？
添加固定随机种子：
```python
import torch
import numpy as np
torch.manual_seed(42)
np.random.seed(42)
```

## 📝 实验记录

### 第一次运行

建议参数：
```
数据集: mnist
Epochs: 20
Batch Size: 128
GPU: CUDA (if available)
```

预期时间：2-5分钟

### 完整实验

```
数据集: mnist + cifar10
Epochs: 20
Batch Size: 128
GPU: CUDA
```

预期时间：10-20分钟

## 🔬 扩展实验

### 1. 不同大小的模型

修改 `hidden_dim`：
```python
# 小模型
hidden_dim = 64   # 参数少，速度快

# 大模型
hidden_dim = 256  # 准确率更高
```

### 2. 不同的优化器

```python
# 在 Trainer 类中修改
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

### 3. 不同的学习率

```python
learning_rate = 0.0001  # 更小的学习率
learning_rate = 0.01    # 更大的学习率
```

## 📚 论文相关

### 引用格式

```bibtex
@article{sansheng2026,
  title={Sansheng 4-Layer: Efficient Tensor Network Architecture},
  year={2026},
  note={Benchmark evaluation on MNIST and CIFAR-10}
}
```

### 关键发现

- 三生架构4层在参数效率方面表现优异
- 适合资源受限的嵌入式或移动应用
- 进一步优化可接近CNN的性能

## 🤝 贡献

有想法？欢迎改进：
1. 添加更多数据集
2. 实现更多基准模型
3. 添加数据增强
4. 优化模型架构
5. 并行运行实验

## 📞 支持

- 查看代码注释了解细节
- 检查 `benchmark_results.json` 查看原始数据
- 运行 `python -c "import torch; print(torch.cuda.is_available())"` 检查GPU

## ✅ 检查清单

运行前：
- [ ] 已安装 Python 3.8+
- [ ] 已安装 PyTorch
- [ ] 已安装其他依赖 (matplotlib, seaborn)
- [ ] 有至少 1GB 可用存储空间
- [ ] 如有GPU，已验证CUDA可用性

运行中：
- [ ] benchmark_experiment.py 在运行
- [ ] 可在终端看到进度输出
- [ ] 数据已下载到 ./data 目录

运行后：
- [ ] 生成了 benchmark_results.json
- [ ] 运行 analyze_benchmark.py 生成报告
- [ ] 生成了 .png 图表
- [ ] 生成了 .md 报告

## 🎉 成功标志

✅ 成功指标：
- 所有模型都完成训练
- 测试准确率在合理范围内
- 生成了所有输出文件
- 可视化图表清晰可读

---

**版本**：v1.0  
**最后更新**：2026-07-03  

