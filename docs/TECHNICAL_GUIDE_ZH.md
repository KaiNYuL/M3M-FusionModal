# Mamba3-Minimal 技术文档（架构、使用、调参）

## 1. 项目定位

`mamba3-minimal` 是一个纯 PyTorch 的单文件参考实现，目标是把 Mamba-3 论文里的关键数学设计，映射成可读、可运行的代码。

核心文件：
- `mamba3.py`
- `demo.py`
- `tests/test_parity.py`
- `tests/test_mimo.py`
- `tests/test_text.py`

## 2. 代码架构总览

从语言模型角度，执行路径是：

1. `Embedding`
2. 重复 `n_layer` 次 block：
3. `RMSNorm -> Mamba3 SSM -> Residual`
4. `RMSNorm -> SwiGLU MLP -> Residual`
5. `RMSNorm -> LM Head(weight tying)`

关键代码位置：
- 配置定义：`mamba3.py:53`
- 推理缓存结构：`mamba3.py:85`
- 顶层语言模型：`mamba3.py:111`
- Mamba-3 核心模块：`mamba3.py:236`
- 单步推理（O(1) decode）：`mamba3.py:485`
- RoPE 旋转实现：`mamba3.py:653`
- SSD 并行算法：`mamba3.py:711`
- MIMO-SSD：`mamba3.py:777`

## 3. 论文设计与代码对应关系

1. Trapezoidal Discretization（Eq.4, Prop.1）
- 论文思想：状态更新拆成 `alpha/beta/gamma` 三个系数。
- 代码位置：`mamba3.py:364` 附近定义 `alpha/beta/gamma`，并在 `forward` 与 `step` 两条路径中使用。

2. Complex SSM + Data-Dependent RoPE（Eq.9）
- 论文思想：通过数据相关旋转角，增强状态追踪能力。
- 代码位置：
- `mamba3.py:345` 附近构造 `raw_angles/cum_angles`
- `mamba3.py:653` 的 `apply_rope()` 完成偶奇维旋转
- `mamba3.py:485` 的 `step()` 更新 `cum_angle`

3. QK-Norm on B/C（Section 3.4）
- 论文思想：把规范化放在 B、C 投影之后，提升稳定性。
- 代码位置：`mamba3.py:332` 附近，`self.B_norm(B)` 和 `self.C_norm(C)`。

4. Learnable BC Bias（Section 3.4, App.G）
- 论文思想：每头、每通道的可学习偏置，初始化为 1。
- 代码位置：`mamba3.py:302` 到 `mamba3.py:321`，在 forward/step 中加到 B、C 上。

5. Two-SSD decomposition（工程实现关键）
- 论文递推里有上一时刻输入项，直接分块有边界依赖。
- 代码实现：拆成 `gamma SSD` + `beta SSD` 两次调用并相加。
- 代码位置：SISO 在 `mamba3.py:426` 附近，MIMO 在 `mamba3.py:383` 附近。

6. MIMO（Appendix D）
- 论文思想：在 rank-R 空间扩展表达能力。
- 代码位置：
- 参数与形状分支：`mamba3.py:256` 到 `mamba3.py:321`
- 并行算法：`mamba3.py:777`
- 单步推理分支：`mamba3.py:543` 附近

## 4. 如何使用这个框架

### 4.1 快速验证

在仓库根目录运行：

```bash
python demo.py
python tests/test_mimo.py
python tests/test_parity.py
python tests/test_text.py
```

用途：
- `demo.py`：看完整流程（forward、step、训练、生成）
- `test_mimo.py`：验证 MIMO/SISO 的 forward-step 一致性
- `test_parity.py`：验证状态追踪能力
- `test_text.py`：验证真实词表生成流程

### 4.2 作为库集成

最小流程：

1. 建 `Mamba3Config`
2. 建 `Mamba3LMHeadModel`
3. 训练时调用 `model(input_ids)`
4. 生成时用 `model.generate(prompt_ids, ...)`

注意：训练路径内部使用 chunked SSD，输入序列长度建议是 `chunk_size` 的整数倍。

## 5. 参数调优指南

### 5.1 模型结构参数

1. `d_model`
- 作用：主通道宽度，决定总体容量。
- 调大效果：精度上限更高，但显存和计算明显增加。

2. `n_layer`
- 作用：网络深度。
- 调大效果：更强表达能力，训练时间线性增长。

3. `d_state`
- 作用：SSM 状态维度。
- 约束：必须是偶数（RoPE 成对旋转）。
- 调大效果：长程状态建模增强，但状态计算更重。

4. `headdim`
- 作用：每头维度，`nheads = d_inner / headdim`。
- 调参建议：与 `d_model`、`expand` 联动，确保整除。

5. `expand`
- 作用：`d_inner = expand * d_model`。
- 调大效果：增加 SSM 分支容量，参数量与算力同时增加。

6. `chunk_size`
- 作用：SSD 分块大小，影响速度/显存折中。
- 调大：块内计算更重，通常吞吐更高但峰值显存上升。
- 调小：更省显存，但可能降低吞吐。

### 5.2 MIMO 参数

1. `use_mimo`
- `false`：SISO 基线
- `true`：启用 rank-R MIMO

2. `mimo_rank`
- 作用：MIMO 的 rank 维度 R。
- 调大效果：表达能力增加，参数量和计算上升。

### 5.3 推理采样参数

1. `temperature`
- 小于 1：更保守
- 大于 1：更发散

2. `top_k`
- 限制候选集合大小，提升稳定性。

3. `top_p`
- 按累计概率截断，常用于控制随机性。

### 5.4 训练稳定性参数

1. `A_log` 初始化区间
- 影响状态衰减快慢。

2. `dt_bias` 初始化区间
- 影响离散步长 `dt` 分布。

3. 梯度裁剪
- 示例里使用 `clip_grad_norm_=1.0`，建议保留。

## 6. 配置化调参（JSON/YAML）

仓库已提供：
- `configs/tuning.template.json`
- `configs/tuning.template.yaml`
- `run_config.py`

### 6.1 使用 JSON 运行

```bash
python run_config.py --config configs/tuning.template.json
```

### 6.2 使用 YAML 运行

```bash
pip install pyyaml
python run_config.py --config configs/tuning.template.yaml
```

脚本会执行：
1. 读取配置
2. 构建模型并初始化参数
3. 跑一次 forward（随机输入）
4. 可选执行 generate（输出 token id）

## 7. 调参实验建议流程

1. 固定数据与随机种子，先跑 SISO 基线。
2. 只改一个维度做扫参：例如先扫 `chunk_size`。
3. 记录吞吐、显存、loss 曲线与一致性测试结果。
4. 再启用 MIMO，扫 `mimo_rank` 对比收益/成本。
5. 最后联合调 `d_model/n_layer/d_state` 做容量规划。

## 8. 常见坑

1. 训练输入长度与 `chunk_size` 不对齐。
- 结果：触发断言。
- 处理：padding 或按块裁剪。

2. `d_state` 设成奇数。
- 结果：RoPE 配对失败。

3. `d_inner % headdim != 0`。
- 结果：头维重排失败。

4. 只看生成效果判断模型质量。
- 建议先看 loss 与任务指标，再看采样文本。
