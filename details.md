# Diffusion 统一过渡模型 — 训练与推理改进总结

> 时间窗：2026/04 阶段性迭代  
> 主体：基于 UWM + CompDiffuser 的 Talos 4-effector 多接触过渡扩散模型  
> 数据：SEIKO 生成的 (init, final) 接触翻转对，单 effector 翻转先验  
> 配置维度：16 维（4×3 位置 + 4 接触），无姿态，归一化到 $[-1,1]$

本文以 STAR 法则（Situation / Task / Action / Result）记录本阶段所有架构、损失、推理、数据层面的改动。

---

## 1. Change-Mask 分类头 + 加权 MSE（架构重构）

### Situation
原始架构对 16 维的 ε 预测做无差别 MSE，未利用 SEIKO 数据"每对 (init, final) **恰好** 1 个 effector 翻转接触并仅其位置变化"的强先验。模型在非翻转 effector 上输出明显漂移，破坏物理一致性。

### Task
让模型显式学到"哪个 effector 在翻转"，并对翻转/非翻转维度给出不同的损失权重。

### Action
**A1. 新增分类头**（[`models/denoiser/mlp.py`](models/denoiser/mlp.py)）
```python
self.change_cls_head = nn.Sequential(
    nn.Linear(h, h//2), nn.SiLU(), nn.Linear(h//2, 4)
)
```
forward 返回 `(eps_init, eps_final, change_logits)` 三元组。Transformer denoiser 提供占位实现保持接口对齐。

**A2. 加权 MSE**（[`models/diffusion.py`](models/diffusion.py) `compute_loss`）

按真实 contact 差分构造分类标签：
$$\hat{i}^* = \arg\max_i |c^{\text{final}}_i - c^{\text{init}}_i|$$

并构造 per-dim 权重：
$$w_d = \begin{cases} w_{\text{changed}} & \text{if dim } d \in \text{slice}(\hat{i}^*) \cup \{c_{\hat{i}^*}\} \\ 1 & \text{otherwise} \end{cases}$$

总损失：
$$\mathcal{L} = \mathcal{L}_{\text{init}}^{\text{wMSE}} + \mathcal{L}_{\text{final}}^{\text{wMSE}} + w_{\text{cls}} \cdot \text{CE}(\text{change\_logits}, \hat{i}^*)$$

**A3. 推理后处理**（`sample_transition`）
- 接触维 clamp 到 $[-1,1]$ 并按 sign 二值化
- 仅允许 `argmax(|Δcontact|)` 对应 effector 的位置和接触维变化，其他维强制等于 init

### Result
- 端到端 smoke test 通过（forward / backward / 单 effector 约束 / contact 二值化）
- 训练日志新增 `loss_change_cls`、`cls_acc` 及按 t-bucket 拆分的 4 路指标，可逐桶诊断模型在低/高噪声段的学习状态
- 后续被发现 `w_changed_dim=5.0` 反而抑制了非翻转维度的学习信号 → 后续调回 `1.0`（见 §4）

---

## 2. 推理评估全景指标 (5 项)

### Situation
单一的 `loss` 数值无法判断模型是否学到了"非翻转 effector 不变"的物理先验，也无法看到后处理是否在掩盖原始模型缺陷。

### Task
在 evaluate 中并行测量 raw 模型输出与 post-process 后输出的差距，量化模型本身学到了多少物理先验。

### Action（[`scripts/train.py`](scripts/train.py) `evaluate()`）
| 指标 | 含义 | 计算 |
|---|---|---|
| `cls_acc` | Change-mask 分类准确率 | $\mathbb{E}[\hat{i} = i^*]$ |
| `loss_init_bucket_{0..3}` | 按 $t$ 分桶的 init MSE | t 均匀分 4 段 |
| `cls_acc_bucket_{0..3}` | 按 $t$ 分桶的分类准确率 | 同上 |
| `flip_rate_raw / postproc` | 单 effector 翻转合规率 | $\mathbb{E}[n_{\text{flip}} = 1]$ |
| `pos_consistency_raw` | 非翻转 effector 位置漂移 ≤ thresh 的比例 | 阈值由 `pos_thresh` 控制 |
| `gap_pos / gap_flip` | post-process 与 raw 的差距 | 越大说明模型越依赖后处理 hack |

新增 `sample_transition(apply_postprocess=True/False)` 开关以同时获取两类输出。

### Result
该指标系统直接暴露了核心 bug：`pos_consistency_raw ≈ 0.000`，`gap_pos ≈ 0.977` —— **模型几乎完全靠后处理满足约束，自身没学到先验**。这一观测驱动了后续数据清洗与权重调整。

---

## 3. 数据清洗 — 强制非翻转 effector 字节同等

### Situation
SEIKO 生成数据存在 ~mm 级数值噪声：非翻转 effector 在 init/final 间漂移 $\bar{d}=0.5$mm、$p_{99}=5.7$mm、$\max=15$mm，约 15–20% 样本在归一化空间漂移 $\geq 0.01$。该噪声与"位置严格不变"的物理先验冲突，模型被迫在两个矛盾目标间权衡。

### Task
在 dataset 层面消除噪声，让训练目标与物理先验完全一致。

### Action（[`data/dataset.py`](data/dataset.py)）

新增 `clean_non_flip=True` 标志与 `_clean_non_flip_effectors()` 方法：
1. 对每对 (init, final) 计算 contact 差分，定位翻转 effector $i^*$
2. **强制** `final[s_j:e_j] = init[s_j:e_j]` for all $j \neq i^*$（位置 6 / 3 维）
3. 同时 `final.contacts[j] = init.contacts[j]` for all $j \neq i^*$

保留反向增强（`_augment_data`）：3000 → 6000 对，物理上完全合法，提升对称性。

### Result
- 3000/3000 对全部被清洗，清洗后非翻转漂移恒为 0
- 训练目标变为单 effector 翻转的 clean 信号

---

## 4. 关键超参修正 — 反思 `w_changed_dim`

### Situation
`w_changed_dim=5.0` 的设计初衷是"加权关注翻转 effector"，但事实上翻转 effector 的 ε 预测本就更易拟合（信号大）；该 5× boost 让模型把容量集中在已学好的维度，反而**降低了非翻转 effector ε≈0 的梯度密度**。

经探针实验：训练后模型 raw 输出的非翻转 effector 漂移 $\approx 0.74$（归一化），而数据本身 drift $\approx 0.08$，**模型严重欠拟合非翻转维度**。

### Task
让权重恢复均衡，并在评估端放宽过严的阈值。

### Action
- [`configs/unified_2080ti.yaml`](configs/unified_2080ti.yaml)：`w_changed_dim: 5.0 → 1.0`
- [`scripts/train.py`](scripts/train.py)：`pos_thresh: 0.01 → 0.05`（归一化阈值，对应 ~3.5–7 cm 物理误差，更符合实际容差）

### Result
权重均衡后非翻转维度获得对等的训练信号；阈值放宽使指标更能反映可用质量而非数值微差。

---

## 5. 数据集合并与覆盖分析

### Situation
新生成两批高质量数据：`seiko_talos_0428_5000.json` (5000 对)、`seiko_talos_0428_7000.json` (7000 对)，加上原 `seiko_talos_merged.json` (3000 对)。需要确认质量后合并。

### Task
合并 + 提供可复用的数据集结构/覆盖分析工具。

### Action
- [`scripts/merge_datasets.py`](scripts/merge_datasets.py)：通用合并工具，校验 schema 一致并记录 `source_files` 元信息
- [`scripts/analyze_dataset.py`](scripts/analyze_dataset.py)：9 子图覆盖分析（XY 散点、Z 直方图、四元数模长、接触模式频率、per-EE flip 计数、成对 EE 距离、非翻转漂移、翻转位移、文本元信息）+ 文本摘要

输出：[`dataset/seiko_talos_0428.json`](dataset/seiko_talos_0428.json)（**15000 对**）

### Result（合并后）
| 指标 | 值 | 评价 |
|---|---|---|
| Single-flip rate | 100.000% | 完美先验 |
| Quaternion norms | 全部 ≈ 1.0 | 输入合法 |
| Non-flip drift mean / p99 / max | 0.47 / 5.4 / 20.9 mm | 仍需 `clean_non_flip=True` 兜底 |
| Flip 位移 mean / range | 76mm / 0.4–137mm | 覆盖广 |
| Per-EE flip 计数 LF/RF/LH/RH | 3181/3237/4245/4337 | 手部略偏多 ~33% |
| 工作空间 (脚 z, 手 z) | $[-0.07, 0.32]$, $[0.62, 1.33]$ | 与物理一致 |

---

## 6. 链规划后处理 Bug 修复

### Situation
推理 chain mode 输出的 best chain 末端 contact / 位置不等于用户输入的 goal。例：goal `[1,1,1,1]` 被改写为 `[1,1,1,0]`，goal 上 RH 位置被回退到前一 waypoint。

### Task
诊断并修复硬约束被破坏的根因。

### Action（[`models/diffusion.py`](models/diffusion.py) `plan_chain` 后处理段）

**根因**：原后处理循环
```python
for k in range(K):              # ← 包含 k=K-1
    all_chains[:, k+1, :] = wp_a + masked_delta
```
当 `k = K-1` 时 `wp_b == goal`，被 `wp_a + (单 effector delta)` 覆盖；若 wp_{K-1}→goal 涉及 ≥2 个 contact 翻转，多余的翻转被 `argmax` 抹掉，goal 被改写。

**修复**：
1. 循环改为 `range(K-1)`，**永不修改 wp_K (goal) 与 wp_0 (start)**
2. 语义改为"前向钉锚"：`wp_{k+1}` 的非翻转 effector **强制等于 `wp_k`**（而非 `wp_a + delta`），保留中间 waypoint 的物理一致性

### Result
goal contact / 位置在所有 case 下严格保持输入值。这次修复也"诚实地"暴露了深层问题：模型本身在末段无法对齐 goal —— 进而引出 §7。

---

## 7. 链规划进度引导（Progress Anchor, B）

### Situation
即使 §6 的硬约束修好，模型 raw 输出的 waypoint 序列**距 goal 不单调**：常出现 wp1 反向远离 goal、末段 wp_{K-1}→goal 一步跨越 80% 距离的情况。这违反单 effector 翻转的物理预算（每段位移 mean 76mm），导致末段强制后处理的"大跳"。

### Task
在去噪过程中对中间 waypoint 的位置施加温和的"线性进度先验"，让序列几何上单调趋近 goal，但不破坏 contact 维（避免 OOD，因为模型未训练 contact-conditioned 生成）。

### Action（[`models/diffusion.py`](models/diffusion.py) `_chain_parallel`）

每个去噪步合并连接点后，对位置维做时间相关的 anchor：
$$
p^{\text{target}}_{k} = \tfrac{k}{K}\, p^{\text{goal}} + \tfrac{K-k}{K}\, p^{\text{start}}, \quad k = 1,\dots,K-1
$$

$$
\text{wp}_k^{\text{pos}} \leftarrow (1-\lambda_t)\cdot \text{avg}^{\text{pos}} + \lambda_t \cdot p^{\text{target}}_{k}
$$

其中
$$\lambda_t = \lambda_{\max} \cdot \frac{t}{T-1}$$

- $t$ 大（噪声大）时强引导，$t \to 0$ 时 $\lambda_t \to 0$ 让模型主导清洁信号
- 仅作用于位置维（pos slices），**接触维不动**，避免破坏训练分布
- 通过 CLI `--progress_lambda`（默认 0.0）开关，向后兼容

### Result（K=6, B=30，相同 case 对照）

| Waypoint | $\lambda=0$ d→goal | $\lambda=0.1$ d→goal |
|---|---|---|
| start | 0.583 | 0.583 |
| wp1 | **0.604 ↑（反向）** | 0.584 |
| wp2 | 0.570 | 0.583 |
| wp3 | 0.565 | **0.430 ↓** |
| wp4 | 0.519 | **0.292 ↓** |
| wp5 | 0.489 | **0.203 ↓** |
| goal | 0.000 | 0.000 |

- $\lambda=0$：5 步只走 16% 距离，**末段单步要走 84%**（违反物理预算）
- $\lambda=0.1$：近似单调递减，末段仅剩 20%，过渡平滑
- 同时单 flip 合规率从混杂提升到全部 = 1

---

## 8. 链规划 Best-Chain 多目标评分（D）

### Situation
原 best-chain 选择仅依赖段间 overlap：
$$\text{score} = \sum_{k=0}^{K-2} \|\text{finals}_k - \text{initials}_{k+1}\|$$

这只衡量"段间是否平滑衔接"，完全不评估"是否到达 goal"或"物理合规度"，于是会选出 raw 末段离 goal 极远的 candidate。

### Task
将 best-chain 选择升级为多目标评分，同时考虑过渡平滑性、目标对齐、单调进度、单 flip 合规。

### Action（[`models/diffusion.py`](models/diffusion.py) `plan_chain` 末段）

$$
\text{score}_b = \underbrace{\sum_{k} \|f_k^{(b)} - i_{k+1}^{(b)}\|}_{\text{overlap}} + 5\cdot\underbrace{\|wp_{K-1}^{(b)} - \text{goal}\|}_{\text{anchor}} + 2\cdot\underbrace{\sum_k \max(0, d_{k+1}^{(b)} - d_k^{(b)})}_{\text{regress penalty}} + 0.5\cdot\underbrace{\sum_k |n^{(b)}_{\text{flip},k} - 1|}_{\text{single-flip violation}}
$$

其中 $d_k^{(b)} = \|wp_k^{(b),\text{pos}} - \text{goal}^{\text{pos}}\|$。

权重经经验调试：anchor 项最重以保证选到"真正到达"的链；进度项次之；单 flip 合规项较轻（已被后处理大致保证）。

### Result
- 选出的 best chain 末段距离明显小于原方案
- 与 §7 的 progress anchor 协同：先在去噪阶段引导，再在选择阶段挑最佳

---

## 9. 设计取舍记录

### 为什么不用 Contact Schedule Inpainting (方案 A)?
> 模型从未训练过"contact 维 clean + pos 维带噪"的输入分布。即使用 RePaint 风格匹配 noise level，**模型也没学过 $p(\text{pos} \mid \text{contact schedule})$ 的条件分布**，强行钉死 contact 会导致 pos 与 contact 脱节（如要求 LF 落地但生成 LF.z=0.3m）。该方案保留为未来重训方向。

### 为什么 progress anchor 仅作用于 pos 维?
> Contact 维上 anchor 会等价于"硬钉接触"，回到方案 A 的 OOD 问题。位置维上的线性插值是几何上的弱先验，不引入分布偏移。

### 为什么 $\lambda$ 随 $t$ 衰减?
> 早期 (t 大) 模型预测含大量噪声，弱拉力可缓解漂移；后期 (t 小) 模型已接近清洁信号，强拉力会与模型预测冲突。$\lambda_t = \lambda_{\max} \cdot t/T$ 是 simple yet 经典的 step-decay schedule，与 classifier guidance 调度族同源。

---

## 10. 文件改动清单

| 文件 | 类型 | 说明 |
|---|---|---|
| [`models/denoiser/mlp.py`](models/denoiser/mlp.py) | 修改 | 添加 change-mask 分类头 |
| [`models/denoiser/transformer.py`](models/denoiser/transformer.py) | 修改 | 占位 change_logits 接口对齐 |
| [`models/diffusion.py`](models/diffusion.py) | 重大修改 | 加权 MSE / CE / 后处理 / chain 修复 / 进度引导 / 多目标 best 选择 |
| [`data/dataset.py`](data/dataset.py) | 修改 | `clean_non_flip` 数据清洗 |
| [`configs/unified_2080ti.yaml`](configs/unified_2080ti.yaml) | 修改 | `w_change_cls`, `w_changed_dim` 参数 |
| [`scripts/train.py`](scripts/train.py) | 修改 | 评估全景指标 + per-bucket logging |
| [`scripts/sample.py`](scripts/sample.py) | 修改 | 新增 `--progress_lambda` |
| [`scripts/merge_datasets.py`](scripts/merge_datasets.py) | 新建 | 多 JSON 合并工具 |
| [`scripts/analyze_dataset.py`](scripts/analyze_dataset.py) | 新建 | 数据结构/覆盖可视化分析 |
| [`dataset/seiko_talos_0428.json`](dataset/seiko_talos_0428.json) | 新建 | 合并后 15000 对训练集 |
| [`visualization/figures/dataset_0428/`](visualization/figures/dataset_0428/) | 新建 | 数据分析输出 |

---

## 11. 后续建议

1. **架构层**：把 `change_logits` 通过 softmax 反向门控 ε 预测 —— 让模型在结构上"非目标 effector ε ≈ 0"，从根上消除对后处理的依赖。
2. **训练层**：用合并后 15000 对 + `clean_non_flip` 重训，期望 `pos_consistency_raw` 显著提升。
3. **数据层**：考虑 contact-conditioned 训练目标（mask 部分 contact 让模型学 $p(\text{pos} \mid \text{contact schedule})$），为方案 A（Contact Schedule Inpainting）铺路。
4. **推理层**：扫 $\lambda_{\max} \in \{0.05, 0.1, 0.2\}$ 与 K $\in \{6,8,10\}$ 的网格，找到不同 (start, goal) 距离下的最佳组合。
