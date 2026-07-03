# 三生（Three‑Life / DaoSheng）代码 Wiki

本文档面向“接手代码的人”，聚焦仓库代码形态、关键实现、依赖关系与运行方式；理论叙事与实验结论类材料请优先阅读仓库内的交接/报告 Markdown（例如 [三生架构_完整交接文档.md](file:///d:/三生/三生架构_完整交接文档.md)、[README_final.md](file:///d:/三生/README_final.md)）。

## 目录

- [1. 仓库概览](#1-仓库概览)
- [2. 运行与复现](#2-运行与复现)
- [3. 架构总览（代码视角）](#3-架构总览代码视角)
- [4. 代码结构（按文件/模块）](#4-代码结构按文件模块)
- [5. 关键类与函数索引](#5-关键类与函数索引)
- [6. 依赖与模块关系](#6-依赖与模块关系)
- [7. 已知坑与整理建议](#7-已知坑与整理建议)
- [8. 术语表（最小集合）](#8-术语表最小集合)

---

## 1. 仓库概览

### 1.1 当前仓库形态（现实 vs README 规划）

- 现实：根目录扁平化，主要由“可运行 Python 脚本 + 实验报告/图片 + zip 打包件”构成（见 [d:\三生](file:///d:/三生/)）。
- README 规划：在 [README_final.md](file:///d:/三生/README_final.md#L61-L84) 中描述了 `core/ learning/ engineering/ experiments/` 的工程化目录，但该目录树当前未落地；相近内容可能在 [sansheng_engineering.zip](file:///d:/三生/sansheng_engineering.zip)、[sansheng_experiments.zip](file:///d:/三生/sansheng_experiments.zip) 内。

### 1.2 “从代码出发”的核心对象

仓库中“真正可复用/可复现”的核心实现主要集中在三条线：

- NumPy 端到端最小闭环（用于概念验证/可快速跑通）
  - [demo_final.py](file:///d:/三生/demo_final.py)
  - 关键点：单纯形状态、耦合算子、五行动力学、去噪、简单分类任务
- PyTorch 可微分实现（用于训练/梯度验证/优化器）
  - [sansheng_full_impl.py](file:///d:/三生/sansheng_full_impl.py)
  - 关键点：`SanshengLayer`（可微分耦合）、`SimplexAdam`（单纯形优化）、分类器与损失
- 道生多尺度（11 层）纯 NumPy 管道（用于“古典映射”的结构化实现）
  - [sansheng_dao_sheng.py](file:///d:/三生/sansheng_dao_sheng.py)
  - 关键点：`DaoShengPipeline` 串联 0~10 层（编码→路由→耦合→法则→拓扑→正则→变换→收缩→自组织→不确定性→收敛）

---

## 2. 运行与复现

### 2.1 Python 环境建议

- Python：建议 `>=3.9`
- 最小依赖：`numpy`
- 实验脚本常用依赖：`matplotlib`、`scipy`
- PyTorch 实现依赖：`torch`（见 [sansheng_full_impl.py](file:///d:/三生/sansheng_full_impl.py#L11-L16)）

可选安装示例（按需选择）：

```bash
pip install numpy
pip install matplotlib scipy
pip install torch
```

### 2.2 推荐的“从零跑通”路径

1) 最小 Demo（纯 NumPy，含分类与去噪对比）

```bash
python demo_final.py
```

- 入口：`main()`（见 [demo_final.py](file:///d:/三生/demo_final.py#L251-L334)）

2) 兼容性验证实验（“二→三→万物”三组实验 + 可视化）

```bash
python sansheng_compatibility_test.py
```

- 入口：`__main__`（见 [sansheng_compatibility_test.py](file:///d:/三生/sansheng_compatibility_test.py#L385-L402)）
- 注意：该脚本默认把图保存到 Linux 风格绝对路径（见 [sansheng_compatibility_test.py](file:///d:/三生/sansheng_compatibility_test.py#L372-L379)），Windows 下需要改为相对路径或本地路径。

3) 道生 11 层管道自测（纯 NumPy）

```bash
python sansheng_dao_sheng.py
```

- 入口：`__main__`（见 [sansheng_dao_sheng.py](file:///d:/三生/sansheng_dao_sheng.py#L1897-L1939)）

4) PyTorch 完整实现自测（如果你本地可用 torch）

```bash
python sansheng_full_impl.py
```

- 入口：`__main__`（见 [sansheng_full_impl.py](file:///d:/三生/sansheng_full_impl.py#L675-L741)）

### 2.3 输出路径注意事项（重要）

部分脚本把输出写死到 `/app/data/...`，在 Windows/本地环境通常不存在：

- 仿真可视化输出路径：[sansheng_simulation_v2.py](file:///d:/三生/sansheng_simulation_v2.py#L485-L501)
- 兼容性验证可视化输出路径：[sansheng_compatibility_test.py](file:///d:/三生/sansheng_compatibility_test.py#L372-L379)
- 学习算法报告输出路径：[sansheng_learning_numpy.py](file:///d:/三生/sansheng_learning_numpy.py#L957-L982)
- 去噪对比脚本通过 `sys.path.insert` 依赖该路径：[sansheng_denoising.py](file:///d:/三生/sansheng_denoising.py#L325-L333)

建议做一次“环境归一化”改造：统一使用相对输出目录（例如 `./outputs/`）并通过命令行参数配置，但这属于仓库整理工作，不在本文档修改范围内。

---

## 3. 架构总览（代码视角）

### 3.1 数据结构：三元单纯形状态

大部分实现共享一个核心约束：每个元胞/位置的状态是三元概率向量，满足：

- `ψ = (α, γ, β)`
- `α, γ, β >= 0`
- `α + γ + β = 1`（单纯形约束）

NumPy demo 侧通过归一化维持约束（例如 [demo_final.py](file:///d:/三生/demo_final.py#L53-L67)），PyTorch 侧通过 `softmax` 投影维持约束（例如 [SanshengLayer.forward](file:///d:/三生/sansheng_full_impl.py#L164-L214)）。

### 3.2 计算主循环：邻域耦合 + 动力学/正则

从代码抽象出一个“最小闭环”：

1) 邻域收集（8 邻域）
2) 对每个邻居应用耦合算子 `C(ψ_u, ψ_v)`
3) 聚合邻域贡献（平均/加权）
4) 残差式更新（由 `ε` 控制）
5) 投影回单纯形（归一化/softmax）
6) 可选：五行循环/去噪/正则化/多尺度层等

对应实现：

- NumPy demo：`get_8_neighbors` → `coupling_operator` → `evolve_step` → `wuxing_transition`（见 [demo_final.py](file:///d:/三生/demo_final.py#L45-L118)）
- PyTorch：`_get_neighbors` → `_coupling_operator` → `forward`（见 [SanshengLayer](file:///d:/三生/sansheng_full_impl.py#L23-L214)）

### 3.3 代码层面的“架构图”

```mermaid
flowchart TD
  A[输入 X] --> B[编码为三生状态 ψ=(α,γ,β)]
  B --> C[邻域收集]
  C --> D[耦合算子 C(ψu,ψv)]
  D --> E[邻域聚合]
  E --> F[残差更新 ε]
  F --> G[投影回单纯形 / softmax]
  G --> H{分支: 任务/实验}
  H --> I[分类: γ pooling + head]
  H --> J[仿真: 可视化与指标]
  H --> K[去噪: 预滤波/一致性/退火]
  H --> L[道生: 多层管道]
```

---

## 4. 代码结构（按文件/模块）

### 4.1 NumPy 最小 Demo：demo_final.py

- 文件： [demo_final.py](file:///d:/三生/demo_final.py)
- 目标：提供一个端到端可运行“闭环”：初始化→演化→分类→去噪对比
- 关键能力块
  - 状态初始化与约束：`initialize_sansheng_state`（见 [demo_final.py](file:///d:/三生/demo_final.py#L24-L35)）
  - 8 邻域：`get_8_neighbors`（见 [demo_final.py](file:///d:/三生/demo_final.py#L45-L51)）
  - 耦合算子：`coupling_operator`（见 [demo_final.py](file:///d:/三生/demo_final.py#L53-L67)）
  - 五行动力学：`wuxing_transition`（见 [demo_final.py](file:///d:/三生/demo_final.py#L69-L94)）
  - 去噪：`denoise`（见 [demo_final.py](file:///d:/三生/demo_final.py#L125-L172)）
  - 分类：`run_classification`（见 [demo_final.py](file:///d:/三生/demo_final.py#L213-L245)）

### 4.2 PyTorch 完整实现：sansheng_full_impl.py

- 文件： [sansheng_full_impl.py](file:///d:/三生/sansheng_full_impl.py)
- 目标：提供可微分版本与训练相关组件，便于在真实训练框架中集成
- 关键组件
  - `SanshengLayer`：三生耦合层（邻域提取、耦合算子、多步演化）
    - 邻域提取：[SanshengLayer._get_neighbors](file:///d:/三生/sansheng_full_impl.py#L94-L123)
    - 耦合算子：[SanshengLayer._coupling_operator](file:///d:/三生/sansheng_full_impl.py#L125-L163)
    - 演化主循环：[SanshengLayer.forward](file:///d:/三生/sansheng_full_impl.py#L164-L214)
  - `SimplexAdam`：在单纯形上维持约束的 Adam 变体（见 [SimplexAdam](file:///d:/三生/sansheng_full_impl.py#L230-L315)）
  - `SanshengClassifier`：编码→演化→γ 池化→分类头（见 [SanshengClassifier](file:///d:/三生/sansheng_full_impl.py#L359-L459)）
  - `EmergenceLoss`：任务损失 + 涌现项 + 平衡正则（见 [EmergenceLoss](file:///d:/三生/sansheng_full_impl.py#L465-L519)）
  - 梯度验证：`verify_gradients_torch`（见 [verify_gradients_torch](file:///d:/三生/sansheng_full_impl.py#L525-L589)）
  - 训练循环：`train_model`（见 [train_model](file:///d:/三生/sansheng_full_impl.py#L595-L669)）

### 4.3 学习算法（NumPy 验证版）：sansheng_learning_numpy.py

- 文件： [sansheng_learning_numpy.py](file:///d:/三生/sansheng_learning_numpy.py)
- 目标：提供 NumPy 版本的耦合层/优化器，实现数学验证与报告生成
- 关键组件
  - `SanshengLayer_numpy`（见 [SanshengLayer_numpy](file:///d:/三生/sansheng_learning_numpy.py#L23-L150)）
  - `SimplexAdam_numpy`（见 [SimplexAdam_numpy](file:///d:/三生/sansheng_learning_numpy.py#L152-L215)）
- 入口：`__main__`（见 [sansheng_learning_numpy.py](file:///d:/三生/sansheng_learning_numpy.py#L957-L985)）
- 注意：默认把报告写死到 `/app/data/...`（见 [sansheng_learning_numpy.py](file:///d:/三生/sansheng_learning_numpy.py#L973-L982)）

### 4.4 去噪增强版：sansheng_denoising.py

- 文件： [sansheng_denoising.py](file:///d:/三生/sansheng_denoising.py)
- 目标：针对“耦合非线性会放大噪声”的问题，提供去噪增强版耦合层，并做鲁棒性对比实验
- 关键组件
  - `SanshengDenoisingLayer`：三重去噪（双边滤波、异常抑制、温度退火） + 耦合演化（见 [SanshengDenoisingLayer](file:///d:/三生/sansheng_denoising.py#L21-L260)）
  - 噪声鲁棒性对比实验入口：`run_noise_robustness_comparison`（见 [sansheng_denoising.py](file:///d:/三生/sansheng_denoising.py#L325-L333)）
- 依赖关系：脚本内部通过 `sys.path.insert` 导入 NumPy 学习实现（见同上链接），因此路径在本地环境通常需要修改。

### 4.5 道生多尺度（11 层）实现：sansheng_dao_sheng.py

- 文件： [sansheng_dao_sheng.py](file:///d:/三生/sansheng_dao_sheng.py)
- 目标：提供“十一层古典融合设计”的可执行管道实现
- 管道类：`DaoShengPipeline`（见 [DaoShengPipeline](file:///d:/三生/sansheng_dao_sheng.py#L1592-L1880)）
- 关键层（按层号）
  - 0：`LiushuEncoder`（见 [LiushuEncoder](file:///d:/三生/sansheng_dao_sheng.py#L71-L208)）
  - 1：`SunziRouter`（见 [SunziRouter](file:///d:/三生/sansheng_dao_sheng.py#L216-L323)）
  - 2：`SanshengCoupler`（核心耦合层，含虚实耦合与五行注入）（见 [SanshengCoupler](file:///d:/三生/sansheng_dao_sheng.py#L333-L496)）
  - 3：`HuangdiLawLayer`（见 [HuangdiLawLayer](file:///d:/三生/sansheng_dao_sheng.py#L506-L608)）
  - 4：`JingluoTopology`（见 [JingluoTopology](file:///d:/三生/sansheng_dao_sheng.py#L618-L733)）
  - 5：`ZhongyongRegularizer`（见 [ZhongyongRegularizer](file:///d:/三生/sansheng_dao_sheng.py#L744-L846)）
  - 6：`YijingTransform`（见 [YijingTransform](file:///d:/三生/sansheng_dao_sheng.py#L857-L1009)）
  - 7：`YinfuContractor`（见 [YinfuContractor](file:///d:/三生/sansheng_dao_sheng.py#L1020-L1179)）
  - 8：`DaodejingSelfOrg`（见 [DaodejingSelfOrg](file:///d:/三生/sansheng_dao_sheng.py#L1193-L1308)）
  - 9：`ZhuangziUncertainty`（见 [ZhuangziUncertainty](file:///d:/三生/sansheng_dao_sheng.py#L1318-L1449)）
  - 10：`DaxueConvergence`（见 [DaxueConvergence](file:///d:/三生/sansheng_dao_sheng.py#L1463-L1588)）

### 4.6 仿真/可视化/兼容性实验脚本

- 兼容性验证（含“二→三涌现 / Transformer 嵌入 / 三生→量子映射”）
  - [sansheng_compatibility_test.py](file:///d:/三生/sansheng_compatibility_test.py)
  - 二→三耦合算子（原始 F1 版本）：`couple_cells`（见 [sansheng_compatibility_test.py](file:///d:/三生/sansheng_compatibility_test.py#L25-L55)）
- 最小可运行仿真（含多子图可视化）
  - [sansheng_simulation_v2.py](file:///d:/三生/sansheng_simulation_v2.py)
  - 输出路径注意：[sansheng_simulation_v2.py](file:///d:/三生/sansheng_simulation_v2.py#L485-L501)

---

## 5. 关键类与函数索引

### 5.1 单纯形约束与投影

- NumPy：通过 `raw / raw.sum(...)` 或 `max + normalize` 保持在 Δ² 上（例如 [demo_final.py](file:///d:/三生/demo_final.py#L64-L67)）
- PyTorch：通过 `F.softmax(..., dim=-1)` 强制投影（例如 [SanshengLayer.forward](file:///d:/三生/sansheng_full_impl.py#L181-L205)）

### 5.2 邻域提取（8 邻域）

- NumPy：`get_8_neighbors`（周期边界取模）（见 [demo_final.py](file:///d:/三生/demo_final.py#L45-L51)）
- PyTorch：`_get_neighbors`（零填充 + 切片）（见 [SanshengLayer._get_neighbors](file:///d:/三生/sansheng_full_impl.py#L94-L123)）

两者边界条件不同，导致数值行为可能不同：Demo 更像环面网格，PyTorch 更像零边界。

### 5.3 耦合算子（核心非线性）

- Demo 版耦合算子（增强版）：[demo_final.py](file:///d:/三生/demo_final.py#L53-L67)
- PyTorch 版耦合算子（增强版）：[SanshengLayer._coupling_operator](file:///d:/三生/sansheng_full_impl.py#L125-L163)
- 兼容性实验中用于“二→三必然涌现”的原始 F1 形式：`couple_cells`（见 [sansheng_compatibility_test.py](file:///d:/三生/sansheng_compatibility_test.py#L25-L55)）

### 5.4 五行动力学（演化调控）

- Demo 中的五行循环：`wuxing_transition`（见 [demo_final.py](file:///d:/三生/demo_final.py#L69-L94)）
- 道生核心耦合层中也引入了“固定五行相生矩阵”注入（见 [SanshengCoupler.__init__](file:///d:/三生/sansheng_dao_sheng.py#L344-L382) 与 [SanshengCoupler.forward](file:///d:/三生/sansheng_dao_sheng.py#L436-L457)）

### 5.5 去噪与鲁棒性

- Demo 的“三重去噪”：`bilateral_filter` + `anomaly_suppression` + `temperature_annealing`（见 [demo_final.py](file:///d:/三生/demo_final.py#L125-L172)）
- 完整去噪层：`SanshengDenoisingLayer`（见 [sansheng_denoising.py](file:///d:/三生/sansheng_denoising.py#L21-L260)）

### 5.6 学习与优化

- PyTorch 优化器：`SimplexAdam`（更新后强制 softmax 投影）（见 [SimplexAdam.step](file:///d:/三生/sansheng_full_impl.py#L261-L314)）
- NumPy 优化器：`SimplexAdam_numpy`（见 [SimplexAdam_numpy.step](file:///d:/三生/sansheng_learning_numpy.py#L176-L214)）
- 梯度验证：
  - PyTorch：`verify_gradients_torch`（见 [verify_gradients_torch](file:///d:/三生/sansheng_full_impl.py#L525-L589)）
  - NumPy：脚本中提供 `verify_gradients_numpy()` 等验证入口（见 [sansheng_learning_numpy.py](file:///d:/三生/sansheng_learning_numpy.py#L249-L260)）

---

## 6. 依赖与模块关系

### 6.1 外部依赖（按重要度）

- 必需：`numpy`
- 常用：`matplotlib`（可视化脚本）、`scipy`（例如 `scipy.linalg.expm` 用于量子映射实验）
  - 例：`expm` 引用见 [sansheng_compatibility_test.py](file:///d:/三生/sansheng_compatibility_test.py#L12-L18)
- 训练相关：`torch`（PyTorch 版本）

### 6.2 模块之间的“真实耦合”

多数脚本是“自包含”的（复制粘贴式复用），因此仓库内模块依赖较少；已识别的显式依赖包括：

- 去噪脚本依赖学习脚本（通过 `sys.path.insert` 引入）：[sansheng_denoising.py](file:///d:/三生/sansheng_denoising.py#L325-L333)
- PyTorch 版本的说明文档与真实文件存在“路径/文件名不一致”的历史痕迹（文档提到 `sansheng_pytorch_version.py`，而仓库内对应实现为 [sansheng_full_impl.py](file:///d:/三生/sansheng_full_impl.py)）

---

## 7. 已知坑与整理建议

### 7.1 已知坑（会影响复现）

- 硬编码输出目录 `/app/data/...`：导致多数可视化/报告写入失败（见 [2.3 输出路径注意事项](#23-输出路径注意事项重要)）
- 边界条件不一致：NumPy demo 8 邻域用周期边界，PyTorch `_get_neighbors` 用零填充（见 [5.2 邻域提取](#52-邻域提取8-邻域)）
- 目录结构与 README 描述不一致：README 中的模块化目录未落地（见 [1.1 当前仓库形态](#11-当前仓库形态现实-vs-readme-规划)）

### 7.2 建议的最小整理方向（不改变算法，只改善可维护性）

- 引入统一的 `outputs/` 目录与 `--out` 参数（所有脚本统一写入）
- 抽出 `sansheng_core.py`（或 `core/` 目录）：统一 `softmax / neighbors / coupling_operator / simplex_project` 等基础函数，减少脚本间复制
- 抽出 `requirements.txt`（按“最小可跑 demo”与“全量实验”分两套）
- 解包 zip 并对齐 README 目录结构（必要时把现有文件移动/重命名并补链接）

---

## 8. 术语表（最小集合）

- Δ²（2‑simplex）：三元概率单纯形，`α+γ+β=1`
- α / β / γ：分别对应“阴 / 阳 / 和合”分量（不同文件里有时顺序不同，阅读代码时以实际索引为准）
- 耦合算子 `C(ψ_u, ψ_v)`：邻域交互的核心非线性映射
- ε：残差式更新的耦合强度/步长（见 [SanshengLayer.epsilon](file:///d:/三生/sansheng_full_impl.py#L79-L83)）
- λ_h：和合项系数（见 [SanshengLayer.lambda_h](file:///d:/三生/sansheng_full_impl.py#L84-L88)）
- λ_bal：平衡力系数（见 [SanshengLayer.lambda_bal](file:///d:/三生/sansheng_full_impl.py#L89-L92)）

