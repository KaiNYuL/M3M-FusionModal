# Fusion 主模型技术报告

## 0. Fusion 架构与参数配置

### 0.1 架构原理

主模型采用 `encoder_mask_only + fusion`，即双分支双向状态空间编码器 + token 级门控融合 + patch 级重建头。

#### 0.1.1 输入、分段映射与 token 化

给定标准化后的多通道输入序列：

$$
X \in \mathbb{R}^{B \times T \times C}
$$

其中 $B$ 为 batch，$T$ 为时间长度，$C$ 为通道数。主流程先做通道投影与分段映射：

1. 线性/卷积前端：`disable_preconv=true` 时使用 `Linear(C -> d_model)`，否则使用 `Conv1d(C -> d_model)`。
2. 激活函数：前端投影后统一使用 SiLU。
3. Patch 映射：`Conv1d(kernel=patch_size, stride=patch_size)` 将时间轴切分为 patch token。

得到 token 表征：

$$
Z \in \mathbb{R}^{B \times N \times d_{model}}, \quad N \approx T / P
$$

其中 $P$ 为 `patch_size`。在本配置中 $P=96$。

#### 0.1.2 Mask 建模与重建目标

训练/评估阶段分别按 `encoder_random_mask_ratio` 与 `encoder_eval_mask_ratio` 随机采样 patch mask，并用可学习 `mask_token` 替换被遮挡 token。

同时，输出端启用 `mask_observed_residual=true`：

1. 对被 mask 的时间位置使用模型预测值。
2. 对未 mask 的时间位置保留原输入残差直通。

这使模型把容量集中在“缺失片段恢复”而非“已观测片段复制”。

#### 0.1.3 双分支编码与门控融合

同一份 masked token 序列并行送入两路双向堆栈：

1. Bi-Mamba3 分支（更强动态建模能力）。
2. Bi-Mamba 分支（更稳健的状态更新）。

两路输出均为：

$$
H_{mamba3}, H_{mamba} \in \mathbb{R}^{B \times N \times 2d_{model}}
$$

随后按 token 计算门值：

$$
g = \sigma\left(W_g[H_{mamba3};H_{mamba}] + b_g\right), \quad g \in \mathbb{R}^{B \times N \times 1}
$$

并执行加权融合：

$$
H_{fusion} = g \odot H_{mamba3} + (1-g) \odot H_{mamba}
$$

其中 `gate_bias` 提供可学习先验，控制训练早期更偏向哪一路分支。

#### 0.1.4 情绪评分偏置（aux bias）在模型中的体现

主观情绪评分向量（valence/arousal 等）记为：

$$
y_{aux} \in \mathbb{R}^{B \times d_{aux}}
$$

通过线性层 + SiLU 变换为条件偏置：

$$
b_{aux} = \mathrm{SiLU}(W_{aux} y_{aux}) \in \mathbb{R}^{B \times d_{model}}
$$

再广播到 token 维后加到输入表征：

$$
Z'_{b,t,:} = Z_{b,t,:} + b_{aux,b,:}
$$

这等价于“按样本级情绪上下文平移整段 token 特征分布”，让状态空间分支在同一生理模式下根据情绪标签选择不同重建轨迹。`no_aux_bias` 消融即关闭该通路（不注入该偏置）。

#### 0.1.5 输出头与最终输出

融合后的 token 经过线性头：

$$
\mathrm{Linear}(2d_{model} \rightarrow C \cdot P)
$$

将每个 token 还原为长度为 $P$ 的多通道片段，再按时间折叠回原序列长度，得到：

$$
\hat{X} \in \mathbb{R}^{B \times T \times C}
$$

最终训练目标在标准化空间上计算 MSE/MAE，并在报告中给出 R2 作为主判据。

### 0.2 主要参数配置（m3m-fusion-mask-restructruing）

本轮主模型采用 mask-only 重建，核心参数如下：

| 参数 | 取值 |
|---|---:|
| prediction_mode | encoder_mask_only |
| ssm_variant | fusion |
| d_model | 32 |
| d_state | 64 |
| headdim | 16 |
| n_bi_layers | 2 |
| patch_size | 96 |
| chunk_size | 32 |
| dropout | 0.15 |
| disable_preconv | true |
| preconv_kernel | 5 |
| encoder_random_mask_ratio | 0.15 |
| encoder_eval_mask_ratio | 0.15 |
| mask_observed_residual | true |
| lr | 1e-4 |
| weight_decay | 0.01 |
| batch_size | 8 |
| epochs | 120 |
| patience | 12 |

其中 `encoder_eval_mask_ratio` 固定为 0.15 以与训练掩码比例一致，避免评估阶段退化为无掩码重建。

## 1. 报告范围与本次修正

本报告基于 Fusion 主模型重建结果，采用 clean24（28 被试过滤后再剔除 4 个异常值）作为核心统计集合。

## 2. 数据与评估口径

- filtered 被试：28
- 异常值剔除：s15, s18, s30, s32
- clean 集合：24
- 指标口径：标准化空间 MSE / MAE / R2

文件：

- ../results/ablation/no_aux_bias_filtered_r2_ge0/fusion_filtered28_clean24_metrics.csv
- ../results/ablation/no_aux_bias_filtered_r2_ge0/fusion_filtered28_outliers_iqr.csv
- ../results/ablation/no_aux_bias_filtered_r2_ge0/fusion_filtered28_clean_summary_iqr.json

## 3. Baseline 对比（Fusion 为唯一主模型）

说明：baseline 区域仅包含 Fusion 与外部基线模型（深度/经典），不包含 mamba3。

表格：

- ../results/baselines_all/filtered_r2_ge0/all_baselines_comparison_table_fusion_primary_clean24.csv

核心均值（clean24）：

| model | n_subjects | mse_mean | mae_mean | r2_mean |
|---|---:|---:|---:|---:|
| fusion_primary_clean24 | 24 | 0.185862 | 0.113841 | 0.841564 |
| tcn_ae | 24 | 136909.849604 | 2.031063 | 0.517947 |
| ridge | 24 | 136903.216844 | 2.414709 | 0.141708 |
| timesnet_ae | 24 | 136918.362876 | 2.226615 | 0.093907 |
| pls | 24 | 136918.078642 | 2.271070 | 0.072014 |
| patch_transformer_ae | 24 | 147583.707846 | 4.250145 | 0.062920 |
| masked_transformer_ae | 24 | 147661.623497 | 4.263997 | 0.054804 |
| random_forest | 24 | 136920.226563 | 2.223584 | 0.046078 |

图像：

![all_models_metric](../results/baselines_all/filtered_r2_ge0/figures/all_models_metric_comparison_fusion_primary_clean24.png)

![r2_gap_vs_fusion](../results/baselines_all/filtered_r2_ge0/figures/all_models_r2_gap_vs_fusion_primary_clean24.png)

### 3.1 Baseline 各模型实现与主模型差异

本节按项目脚本中的真实实现给出基线模型定义，并与 Fusion 主模型（`encoder_mask_only + ssm_variant=fusion`）逐一比较。

#### 3.1.1 深度 baseline（`run_deep_baselines_filtered_compare.py`）

1. `patch_transformer_ae`
	- 实现：`Conv1d` patch embedding + 可学习位置编码 + `TransformerEncoder` + `Linear` patch 重建头。
	- 重建方式：直接重建整段序列，不使用双分支状态空间。
	- 与主模型差异：无 Mamba 状态空间、无 Bi-Mamba3/Bi-Mamba 门控融合、无情绪评分偏置注入。

2. `masked_transformer_ae`
	- 实现：结构与 `patch_transformer_ae` 相同，但加入 `mask_token`，训练时对 token 级随机掩码后再编码重建。
	- 重建方式：Transformer 掩码自编码。
	- 与主模型差异：虽有 token mask，但编码器骨干是 Transformer 而非状态空间双向堆栈；无双分支门控融合；无 aux bias 条件偏置。

3. `tcn_ae`
	- 实现：`1x1 Conv` 输入投影 + 多层空洞卷积残差块（dilation 递增）+ `1x1 Conv` 输出。
	- 重建方式：纯卷积时序重建。
	- 与主模型差异：无 patch token 机制、无状态空间递推、无双向分支与门控、无 aux bias。

4. `timesnet_ae`
	- 实现：`Linear` 投影 + `TimesBlock` 堆叠；`TimesBlock` 通过 FFT 提取主周期并做多尺度 1D inception 卷积。
	- 重建方式：周期建模驱动的自编码重建。
	- 与主模型差异：无 Mamba 状态更新、无双分支融合与门控、无 aux bias 条件调制。

深度 baseline 的公共训练口径：同样读取 `encoder_random_mask_ratio / encoder_eval_mask_ratio`，并在输入侧做随机置零掩码；优化器为 AdamW，损失为 Huber，早停策略与主流程保持可比。

#### 3.1.2 经典 baseline（`run_classic_ml_baselines_filtered_compare.py`）

1. `ridge`
	- 实现：`PCA(x)` + `PCA(y)` 降维后，用 `Ridge` 做多输出线性回归，再逆变换回原特征空间。

2. `pls`
	- 实现：`PCA(x)` + `PCA(y)` 后，使用 `PLSRegression` 做潜变量回归，再逆变换。

3. `random_forest`
	- 实现：`PCA(x)` + `PCA(y)` 后，使用 `RandomForestRegressor`（200 棵树、限定深度）回归。

经典 baseline 与主模型的关键差异：

1. 输入展平为二维向量后建模（样本级回归），不保留 token 级时序状态演化。
2. 无 patch 化、无双向状态空间堆栈、无门控融合。
3. 无情绪评分辅助偏置分支。

因此，baseline 区域本质上是在“统一数据切分与统一评价口径”下，对比不同建模范式相对于 Fusion 主模型的上限差距。

## 4. 消融实验（四路）

本节统一以 Fusion 为基准，比较 3 个消融对象：

1. mamba3
2. mamba
3. no_aux_bias

结果文件：

- ../results/ablation/no_aux_bias_filtered_r2_ge0/ablation_compare_fusion_primary_vs_mamba3_vs_mamba_vs_no_aux_bias_clean24.csv
- ../results/ablation/no_aux_bias_filtered_r2_ge0/ablation_compare_fusion_primary_vs_mamba3_vs_mamba_vs_no_aux_bias_clean24_summary.json

关键均值（clean24）：

- Fusion R2 mean = 0.841564
- mamba3 R2 mean = 0.719090
- mamba R2 mean = 0.805093
- no_aux_bias R2 mean = 0.528771

相对 Fusion 的 R2 差值均值：

- mamba3 - fusion = -0.122474
- mamba - fusion = -0.036471
- no_aux_bias - fusion = -0.312793

图像：

![ablation_grouped](../results/ablation/no_aux_bias_filtered_r2_ge0/figures/r2_grouped_fusion_mamba3_mamba_no_aux_bias_clean24.png)

![ablation_delta](../results/ablation/no_aux_bias_filtered_r2_ge0/figures/delta_r2_mamba3_mamba_no_aux_minus_fusion_clean24.png)

消融结论：

1. mamba3 相比 Fusion 有明显退化。
2. mamba 相比 Fusion 有中等退化，但优于 mamba3。
3. no_aux_bias 退化最大，说明情绪评分偏置对当前任务有显著正向作用。

### 4.1 消融模型具体实现与主模型差异

本节对应项目主训练脚本中 `prediction_mode=encoder_mask_only` 分支的实现。

1. Fusion（主模型）
	- 参数组合：`ssm_variant=fusion`，`disable_aux_bias=false`。
	- 结构：`MultiModalMambaKANEncoderMaskOnlyFusion`。
	- 机制：并行构建 Bi-Mamba3 与 Bi-Mamba 两路双向堆栈，得到 `h_m3` 与 `h_m` 后，以
	  $$
	  g=\sigma(W_g[h_{m3};h_m]+b_g),\quad h=g\odot h_{m3}+(1-g)\odot h_m
	  $$
	  做 token 级门控融合，再通过 patch 重建头还原序列。

2. mamba3 消融
	- 参数组合：`ssm_variant=mamba3`，`disable_aux_bias=false`。
	- 结构：`MultiModalMambaKANEncoderMaskOnly`（单一路双向堆栈）。
	- 与主模型差异：移除 Bi-Mamba 分支与门控融合，仅保留 Bi-Mamba3 主干。

3. mamba 消融
	- 参数组合：`ssm_variant=mamba`，`disable_aux_bias=false`。
	- 结构：同为 `MultiModalMambaKANEncoderMaskOnly`，但底层状态方程切换为 mamba 变体。
	- 与主模型差异：既移除融合门控，又将状态更新从 mamba3 退化为 mamba。

4. no_aux_bias 消融
	- 参数组合：`disable_aux_bias=true`（其余结构与对应主干保持一致，实验脚本中通常基于 Fusion 主干加载基线权重后仅关闭 aux bias 通路评估）。
	- 实现方式：训练脚本中的 aux 注入分支被跳过，即不执行 `x = x + SiLU(W_aux y_aux)`。
	- 与主模型差异：去除情绪评分条件偏置，仅依赖生理信号自身进行掩码重建。

5. 实验执行入口（用于复现差异）
	- `run_bimamba_ablation_filtered.py`：调用主训练脚本并传入 `--ssm_variant mamba` 形成 mamba 消融。
	- `run_no_aux_bias_ablation_filtered.py`：调用主训练脚本并传入 `--disable_aux_bias true` 形成 no_aux_bias 消融。
	- mamba3 对照来自同一训练脚本的 `ssm_variant=mamba3` 结果，随后由汇总脚本统一并入四路对比表。

## 5. 通道级结果（Fusion, clean24）

文件：

- ../results/summary/channel_test_fusion_primary_clean24/fusion_primary_clean24_per_channel_test_mse.csv
- ../results/summary/channel_test_fusion_primary_clean24/fusion_primary_clean24_per_channel_test_summary.csv
- ../results/summary/channel_test_fusion_primary_clean24/fusion_primary_clean24_channel_mapping.json

关键统计：

1. 最优通道：ch_16（原始 #40），mse_mean = 0.1440
2. 最差通道：ch_13（原始 #37），mse_mean = 0.2586

图像：

![channel_bar](../results/summary/channel_test_fusion_primary_clean24/figures/fusion_primary_clean24_channel_test_mse_mean_std.png)

![channel_heatmap](../results/summary/channel_test_fusion_primary_clean24/figures/fusion_primary_clean24_channel_test_mse_subject_heatmap.png)

## 6. 最终结论

1. 在 clean24 上，Fusion 是当前最优主模型。
4. no_aux_bias 进一步验证了情绪评分偏置模块的必要性。

## 7. 资产索引

- ../results/summary/fusion_primary_clean24_report_assets.json

## 7. 各个多模态通道简述
| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
| 8 | 10 | 15 | 21 | 22 | 23 | 24 | 26 | 27 | 28 | 31 | 35 | 36 | 37 | 38 | 39 | 40 |
| T7 | CP1 | Oz | F8 | FC6 | FC2 | Cz | T8 | CP6 | CP2 | PO4 | zEMG | tEMG | GSR | Respiration Belt | PPG | Temp |




