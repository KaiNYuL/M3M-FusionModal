# TechFlow v2.0：DEAP 单被试多通道 Mask-Only 重建技术说明

## 1. 任务定义与目标

### 1.1 任务类型
本版本任务为 `mask-only` 信号重建，不是未来预测：
- 输入：完整时间序列（多通道 EEG 片段）+ 主观标签辅助向量。
- 训练行为：对输入序列随机进行 patch 级掩码（mask）。
- 输出目标：重建原始完整序列。

### 1.2 研究目标
在小样本单被试设置（DEAP s01，40 trials）下，通过结构化状态空间模型（Mamba3）完成稳定的多通道缺失重建，核心指标为：
- 标准化空间：MSE / MAE / R2
- 原始空间：raw_MSE / raw_MAE / raw_R2

## 2. 数据与预处理流程

### 2.1 数据来源
- 数据文件：`data/s01.dat`
- 被试数量：1（单被试）
- trial 数：40
- 原始信号长度：8064
- 通道选择（1-based）：`[8,10,15,21,22,23,24,26,27,28,31,35,36,37,38,39,40]`，共 17 通道。

### 2.2 数据组织
从 `s01.dat` 读取后，信号形状组织为：
- `x_full`: `[N, T, C] = [40, 8064, 17]`
- `y_aux`: `[N, 4]`（主观标签四维）

### 2.3 划分与标准化
- 划分：`train/val/test = 22/8/10`
- 划分参数：`test_size=0.25`, `val_size=0.25`, `group_split=false`
- 归一化：`subject_norm=false`，采用训练集统计量做 z-score 标准化（输入与目标分开缩放器）。
- 噪声增强：训练阶段在输入上加入高斯噪声，`noise_std=0.001`。

## 3. 模型架构（Mask-Only 版本）

### 3.1 总体结构
模型为 `MultiModalMambaKANEncoderMaskOnly`，流程如下：
1. 时域预卷积：`Conv1d(in_channels=17 -> d_model=32, kernel=5)`
2. Patch 嵌入：`Conv1d(kernel=stride=patch_size=96)`
3. 掩码注入：随机 patch mask，mask 位置替换为可学习 `mask_token`
4. 辅助标签融合：将 `y_aux` 投影到 `d_model` 后作为偏置加到 token 序列
5. 双向 Mamba3 堆叠：`n_bi_layers=2`（前向流 + 反向流）
6. Patch 解码：线性头输出每个 token 对应的 `C * patch_size`
7. Fold 回时域：恢复为 `[B, T, C]`

### 3.2 关键参数
- `d_model=32`
- `d_state=64`
- `headdim=16`
- `n_bi_layers=2`
- `chunk_size=32`
- `patch_size=96`
- `dropout=0.15`
- `use_mimo=auto`（当前显存下实际为 `false`）

### 3.3 核心改进机制（v2.0）

#### 3.3.1 Mask 区域优先损失
训练损失支持仅强调被 mask 区域：
- `mask_loss_on_masked_only=true`
- `mask_visible_loss_weight=0.05`

含义：
- mask 区域权重约为 1
- 可见区域权重约为 0.05
- 让模型重点学习“缺失补全”而不是“复制可见区”

#### 3.3.2 可见区残差直通
- `mask_observed_residual=true`
- 在输出端将可见位置直接回填原输入，网络主要负责 mask 区域重建。

#### 3.3.3 评估防退化保护
当 `mask_observed_residual=true` 且 `encoder_eval_mask_ratio<=0` 时，代码自动将评估 mask 比例提升到非零（通常不小于训练 mask 比例），避免出现“无 mask 导致虚高分”的退化评估。

## 4. 训练目标与优化策略

### 4.1 损失函数
- 基础逐元素损失：`HuberLoss(delta=1.0, reduction='none')`
- 聚合方式：加权归一化聚合（支持通道权重 + mask 权重）
- 当前设置：`use_channel_weight=false`（禁用通道方差加权）

### 4.2 优化器与学习率策略
- 优化器：`AdamW`
- 参数：`lr=1e-4`, `weight_decay=0.01`
- 调度器：`ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6)`
- 早停：`patience=12`
- 选模指标：`selection_metric='val_loss'`

### 4.3 掩码策略
- 训练 mask 比例：`encoder_random_mask_ratio=0.15`
- 配置中评估比例：`encoder_eval_mask_ratio=0.0`
- 实际运行：因防退化保护，评估会自动使用非零比例（本次记录为 0.15）。

## 5. 本次实验配置快照

来自 `mamba3-minimal/configs/deap_multimodal_mask_only_s01.yaml`：
- `prediction_mode='encoder_mask_only'`
- `epochs=240`, `batch_size=8`, `seed=42`
- `lr=1e-4`, `loss_type='huber'`
- `mask_loss_on_masked_only=true`
- `mask_visible_loss_weight=0.05`
- `mask_observed_residual=true`

## 6. 结果摘要（当前最佳记录）

来自 `outputs/deap_multimodal_mask_only_s01/metrics_report.json`：
- 标准化空间：
  - `mse = 0.2257`
  - `mae = 0.1362`
  - `r2 = 0.7511`
- 原始空间：
  - `raw_mse = 1.1415e7`
  - `raw_mae = 345.10`
  - `raw_r2 = 0.6769`

解释：
- 说明模型在当前 mask-only 重建任务上已具备较强拟合和重建能力。
- `r2` 明显转正且达到较高水平，表明改进机制有效。

阶段性成果说明：
- 本次结果为阶段性实验成果，用于验证当前方法在单被试 mask-only 任务上的有效性。
- 该结果可作为论文阶段性主结果之一，但正式定稿建议补充多随机种子复现、对比基线与消融实验。

## 7. 可视化产物

当前已生成图像（用于论文插图/附录）：
- `outputs/deap_multimodal_mask_only_s01/figures/loss_curves.png`
- `outputs/deap_multimodal_mask_only_s01/figures/val_r2_curve.png`
- `outputs/deap_multimodal_mask_only_s01/figures/metrics_bar.png`
- `outputs/deap_multimodal_mask_only_s01/figures/per_channel_mse.png`

图 1：训练与验证损失曲线

![loss_curves](../results/figures/loss_curves.png)

图 2：验证集 R2 曲线

![val_r2_curve](../results/figures/val_r2_curve.png)

图 3：总指标柱状图（best_val_loss / test_mse / test_mae / test_r2）

![metrics_bar](../results/figures/metrics_bar.png)

图 4：各通道 MSE 柱状图

![per_channel_mse](../results/figures/per_channel_mse.png)

## 8. 复现实验命令

### 8.1 训练
```powershell
C:/Users/KN/AppData/Local/Programs/Python/Python311/python.exe mamba3-minimal/deap_mamba3_multimodal_decoder.py --config mamba3-minimal/configs/deap_multimodal_mask_only_s01.yaml
```

### 8.2 指标可视化
```powershell
C:/Users/KN/AppData/Local/Programs/Python/Python311/python.exe mamba3-minimal/plot_training_metrics.py --report outputs/deap_multimodal_mask_only_s01/metrics_report.json
```

## 9. v2.0 相对前版本的关键变化

1. 任务明确切换为纯 mask 重建（去除未来外推目标）。
2. 引入 mask 区域优先损失（masked-only weighted loss）。
3. 引入可见区残差直通（observed residual passthrough）。
4. 引入评估防退化保护，避免 0-mask 评估导致虚高结果。
5. 增加结果可视化自动化脚本与每通道误差分析图。

## 10. 论文写作建议（对应本技术流）

建议在论文中将本方法命名为：
- `Bi-Mamba3 Masked Reconstruction with Observed Residual`（可中文译名：双向 Mamba3 观测残差增强的掩码重建）

方法章节可按以下结构展开：
1. 问题定义（多通道 patch mask 重建）
2. 模型主干（双向 Mamba3 编码与 patch 解码）
3. 损失设计（mask 优先 + 可见区弱约束）
4. 推理策略（可见区残差直通 + 非零评估 mask）
5. 实验设置与结果（单被试、固定通道、可复现实验）

## 11. 致谢

本项目在实现与实验过程中，参考并受益于以下工作与作者：

1. Albert Gu 与 Tri Dao：Mamba 架构系列作者，为状态空间建模方向提供了核心理论与方法基础。
2. Tommy Ip：`mamba2-minimal` 作者，其 SSD chunking 的最小实现思路对本项目代码组织与实现路径有重要启发。
3. John Ma：`mamba-minimal` 作者，其“可读、可教学”的极简实现风格为本项目工程实践提供了参考。

同时感谢 Mamba3 相关社区贡献者与开源维护者，提供了高质量的讨论、实现样例与验证思路，使本项目能够在可复现、可解释的方向上持续迭代。

