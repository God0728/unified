# Unified Transition Diffusion Model — 架构重构指令

> **目标读者**：Claude Opus 4（或同等能力的 LLM），用于一次性完成跨文件的架构级代码重构。
> **项目仓库**：`God0728/unified`（GitHub）
> **核心诉求**：DDL 将近，需要模型快速出效果。

---

## 0. 项目背景与核心约束

### 0.1 项目目标

通过 Diffusion 模型学习大量符合物理约束（SEIKO 生成）的 `(initial, final)` 末端执行器位置对，训练一个统一模型，具备：

1. **多接触长过程规划（Chain Planning）**：给定起始和目标状态，生成中间 transition 序列
2. **条件生成**：给定 initial 生成 final（前向），或给定 final 生成 initial（后向）
3. **无条件生成**：联合采样 `(initial, final)` 对

### 0.2 数据集的核心先验（必须严格遵守）

数据集由 SEIKO（Sequential Equilibrium Inverse Kinematic Optimization）生成，针对 Talos 人形机器人（4 个末端执行器：LF, RF, LH, RH）。**每一对 (initial, final) transition 严格满足以下约束**：

1. **恰好且仅有 1 个末端执行器的 contact 状态发生翻转**（0→1 或 1→0）
2. **发生 contact 翻转的那个末端执行器的位置（pos）同时发生变化**（均值约 0.075m）
3. **其余 3 个末端执行器的位置和 contact 保持不变**（漂移 < 0.001m，为数值噪声）

这是一个**极强的结构化先验**，当前模型架构完全没有利用它。

### 0.3 当前架构概述

| 组件 | 文件 | 说明 |
|------|------|------|
| Diffusion 主模型 | `models/diffusion.py` | 训练 loss、采样、chain planning |
| MLP 去噪器 | `models/denoiser/mlp.py` | 4 层 MLP，per-layer condition injection |
| Conditioning | `models/denoiser/conditioning.py` | Overlap state encoder |
| Embeddings | `models/denoiser/embeddings.py` | Sinusoidal + MLP timestep embedding |
| 工厂函数 | `models/denoiser/__init__.py` | `build_denoiser()` |
| 数据集 | `data/dataset.py` | 加载 JSON、归一化、反转增强 |
| 训练脚本 | `scripts/train.py` | 训练循环、EMA、TensorBoard |
| 采样脚本 | `scripts/sample.py` | 推理 CLI |
| 配置文件 | `configs/unified_2080ti.yaml` | 主要使用的配置 |

### 0.4 当前数据表示（16D，`use_orientation=False`）

```
config = [LF_pos(3), RF_pos(3), LH_pos(3), RH_pos(3), contacts(4)]
         [0:3]       [3:6]       [6:9]       [9:12]      [12:16]
```

Contact 维度为 0/1 布尔值，但当前被当作连续变量参与扩散。

---

## 1. 需要执行的改动（按优先级排序）

### 改动 A：MLP 增加 Change Mask 分类头（最关键）

**目标**：让模型显式预测"哪个末端执行器发生了变化"，并在推理时用硬掩码强制约束。

**修改文件**：`models/denoiser/mlp.py`

**具体改动**：

1. 在 `UnifiedTransitionDenoiserMLP.__init__` 中，在现有的 `self.output_init` 和 `self.output_final` 之后，新增一个分类头：

```python
# ---- Change Mask Classification Head ----
# Predicts which of the 4 effectors changed (LF=0, RF=1, LH=2, RH=3)
self.change_cls_head = nn.Sequential(
    nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
    nn.SiLU(),
    nn.Linear(hidden_dims[-1] // 2, 4),  # 4-way classification
)
```

2. 在 `forward()` 方法中，在计算 `eps_init_pred` 和 `eps_final_pred` 之后，额外计算分类 logits：

```python
# Predict noise for both parts
eps_init_pred = self.output_init(h)    # (B, D)
eps_final_pred = self.output_final(h)  # (B, D)

# Predict which effector changed
change_logits = self.change_cls_head(h)  # (B, 4)

return eps_init_pred, eps_final_pred, change_logits
```

3. **返回值变更**：`forward()` 现在返回 3 个值而非 2 个。所有调用 `self.denoiser(...)` 的地方都需要适配。

### 改动 B：Diffusion 训练 Loss 增加分类损失 + 加权 MSE

**修改文件**：`models/diffusion.py`

**具体改动**：

1. 在 `__init__` 中新增配置参数：

```python
# Change mask loss weight
self.w_change_cls = train_cfg.get('w_change_cls', 1.0)
# Per-dimension loss weighting: boost changed effector dims
self.w_changed_dim = train_cfg.get('w_changed_dim', 5.0)
```

2. 在 `__init__` 中存储 effector 的维度切片（用于构建掩码）：

```python
# Effector dimension slices for change mask
# 这里使用 config 中的 slices 信息
self._effector_slices = []
for key in ['left_foot', 'right_foot', 'left_hand', 'right_hand']:
    s = config['data']['slices'][key]
    self._effector_slices.append((s[0], s[1]))
# Contact slices (each effector has 1 contact dim)
cs = config['data']['slices']['contacts']
self._contact_start = cs[0]  # 12 for 16D
```

3. 在 `compute_loss` 中，**计算 ground-truth change label 和加权 loss**：

```python
# ---- Compute ground-truth change labels ----
# Compare initial and final contact dims to find which effector flipped
# contacts are at indices [12:16] for 16D config
contact_init = initial[:, self._contact_start:]  # (B, 4)
contact_final = final[:, self._contact_start:]    # (B, 4)
contact_diff = (contact_final - contact_init).abs()  # (B, 4)
change_labels = contact_diff.argmax(dim=-1)  # (B,) values in {0,1,2,3}

# ---- Build per-dimension weight mask ----
# Base weight = 1.0 for all dims, boosted for the changed effector's dims
dim_weights = torch.ones(B, self.config_dim, device=device)
for b_idx in range(B):
    eff_idx = change_labels[b_idx].item()
    s, e = self._effector_slices[eff_idx]
    dim_weights[b_idx, s:e] = self.w_changed_dim
    # Also boost the corresponding contact dim
    dim_weights[b_idx, self._contact_start + eff_idx] = self.w_changed_dim
```

**注意**：上面的逐样本循环效率较低，可以用向量化实现：

```python
# Vectorized version
dim_weights = torch.ones(B, self.config_dim, device=device)
for eff_idx in range(4):
    mask = (change_labels == eff_idx)  # (B,)
    if mask.any():
        s, e = self._effector_slices[eff_idx]
        dim_weights[mask, s:e] = self.w_changed_dim
        dim_weights[mask, self._contact_start + eff_idx] = self.w_changed_dim
```

4. 修改 MSE loss 计算，引入维度权重：

```python
# ---- Compute weighted losses ----
# 原来：loss_init_per_sample = ((eps_init_pred - noise_init) ** 2).mean(dim=-1)
# 改为：
loss_init_per_dim = (eps_init_pred - noise_init) ** 2  # (B, D)
loss_final_per_dim = (eps_final_pred - noise_final) ** 2  # (B, D)

loss_init_per_sample = (loss_init_per_dim * dim_weights).mean(dim=-1)  # (B,)
loss_final_per_sample = (loss_final_per_dim * dim_weights).mean(dim=-1)  # (B,)
```

5. 增加分类损失：

```python
# ---- Change classification loss ----
import torch.nn.functional as F
change_cls_loss = F.cross_entropy(change_logits, change_labels)

total_loss = (self.w_initial * loss_init + self.w_final * loss_final
              + self.w_change_cls * change_cls_loss)

return {
    'loss': total_loss,
    'loss_init': loss_init.item(),
    'loss_final': loss_final.item(),
    'loss_change_cls': change_cls_loss.item(),
}
```

6. **适配 denoiser 返回值**：所有调用 `self.denoiser(...)` 的地方（`compute_loss`、`_cfg_predict`）需要接收 3 个返回值：

```python
# 在 compute_loss 中：
eps_init_pred, eps_final_pred, change_logits = self.denoiser(
    initial_noisy, final_noisy, t_init, t_final, tj_cond=tj_cond
)

# 在 _cfg_predict 中：
eps_init_cond, eps_final_cond, change_logits_cond = self.denoiser(
    initial, final, t_init, t_final, tj_cond=tj_cond
)
# ... unconditional pass ...
eps_init_uncond, eps_final_uncond, _ = self.denoiser(
    initial, final, t_init, t_final, tj_cond=None
)
# CFG 只应用于 eps，change_logits 使用 conditional 的结果
eps_init = eps_init_uncond + guidance_scale * (eps_init_cond - eps_init_uncond)
eps_final = eps_final_uncond + guidance_scale * (eps_final_cond - eps_final_uncond)
return eps_init, eps_final, change_logits_cond
```

### 改动 C：推理时硬约束后处理

**修改文件**：`models/diffusion.py`（`sample_transition` 和 `plan_chain` 相关方法）

**核心思路**：在去噪完成后，利用 `change_logits` 的预测结果，对输出施加硬约束。

在 `sample_transition` 的**最后**（`return` 之前），增加后处理：

```python
# ---- Post-processing: enforce single-effector change constraint ----
# Get change prediction from the last denoising step's logits
# (change_logits is from the last _cfg_predict call)
if hasattr(self, '_last_change_logits'):
    change_pred = self._last_change_logits.argmax(dim=-1)  # (B,)
    
    # For each sample, zero out the delta on non-target effectors
    delta = final_t - initial_t  # (B, D)
    masked_delta = torch.zeros_like(delta)
    for eff_idx in range(4):
        mask = (change_pred == eff_idx)  # (B,)
        if mask.any():
            s, e = self._effector_slices[eff_idx]
            masked_delta[mask, s:e] = delta[mask, s:e]
            # Also allow the corresponding contact dim to change
            masked_delta[mask, self._contact_start + eff_idx] = delta[mask, self._contact_start + eff_idx]
    
    final_t = initial_t + masked_delta
    
    # Binarize contacts
    final_t[:, self._contact_start:] = (final_t[:, self._contact_start:] > 0).float()
    initial_t[:, self._contact_start:] = (initial_t[:, self._contact_start:] > 0).float()
```

**更优雅的实现方式**：不要用 `_last_change_logits` 这种 hack，而是让 `_cfg_predict` 也返回 `change_logits`，然后在采样循环中传递。具体来说：

1. `_cfg_predict` 返回 `(eps_init, eps_final, change_logits)`
2. 在 `sample_transition` 的循环中，保存最后一步的 `change_logits`
3. 循环结束后执行后处理

### 改动 D：Contact 维度的特殊处理

**修改文件**：`models/diffusion.py`

**问题**：Contact 是离散的 0/1 值，但当前被当作连续变量参与高斯扩散。这会导致去噪后的 contact 值在 0~1 之间浮动。

**方案**：在每一步去噪后，对 contact 维度进行 clamp：

在 `sample_transition` 的去噪循环中，每一步之后加入：

```python
# Clamp contact dimensions to [0, 1] range during denoising
# (注意：这里是在归一化空间中操作，contact 维度的归一化是 0->-1, 1->1)
# 所以 clamp 到 [-1, 1]
if denoise_init:
    initial_t[:, self._contact_start:] = initial_t[:, self._contact_start:].clamp(-1, 1)
if denoise_final:
    final_t[:, self._contact_start:] = final_t[:, self._contact_start:].clamp(-1, 1)
```

**最终输出时**，硬二值化：

```python
# After denoising loop, binarize contacts
# In normalized space: < 0 -> -1 (contact=0), >= 0 -> 1 (contact=1)
initial_t[:, self._contact_start:] = torch.where(
    initial_t[:, self._contact_start:] >= 0,
    torch.ones_like(initial_t[:, self._contact_start:]),
    -torch.ones_like(initial_t[:, self._contact_start:]),
)
final_t[:, self._contact_start:] = torch.where(
    final_t[:, self._contact_start:] >= 0,
    torch.ones_like(final_t[:, self._contact_start:]),
    -torch.ones_like(final_t[:, self._contact_start:]),
)
```

### 改动 E：配置文件更新

**修改文件**：`configs/unified_2080ti.yaml`

在 `training` 部分新增：

```yaml
training:
  # ... existing ...
  
  # Change mask classification loss weight
  w_change_cls: 1.0
  # Boost factor for changed effector dimensions in MSE loss
  w_changed_dim: 5.0
```

### 改动 F：训练脚本适配

**修改文件**：`scripts/train.py`

1. 在 TensorBoard logging 中增加 `loss_change_cls`：

```python
writer.add_scalar('train/loss_change_cls', losses.get('loss_change_cls', 0), global_step)
```

2. 在 epoch summary 打印中增加分类损失：

```python
print(f"  Epoch {epoch+1:4d}/{train_cfg['num_epochs']} | "
      f"Loss: {avg_loss:.4f} (init: {avg_loss_init:.4f}, final: {avg_loss_final:.4f}, "
      f"cls: {avg_loss_cls:.4f}) | ...")
```

---

## 2. 需要注意的关键细节

### 2.1 归一化空间 vs 原始空间

所有模型内部的操作都在**归一化空间**中进行（`LimitsNormalizer`：`[min, max] -> [-1, 1]`）。Contact 维度的归一化映射是 `0 -> -1, 1 -> 1`。因此：

- 在归一化空间中，contact 的"阈值"是 `0`（而非 `0.5`）
- 二值化应该是 `>= 0 -> 1, < 0 -> -1`

### 2.2 Denoiser 返回值的一致性

修改 MLP 的 `forward()` 返回 3 个值后，**Transformer denoiser 也需要同步修改**（即使当前不使用），否则 `build_denoiser` 的接口不一致会导致未来切换架构时崩溃。在 `models/denoiser/transformer.py` 的 `UnifiedTransitionDenoiser.forward()` 末尾也返回一个 dummy 的 `change_logits`：

```python
# In transformer.py forward():
change_logits = torch.zeros(B, 4, device=device)  # placeholder
return eps_init_pred, eps_final_pred, change_logits
```

### 2.3 Chain Planning 中的后处理

`plan_chain` 中的 `_chain_parallel` 和 `_chain_autoregressive` 也调用了 `_cfg_predict`。在 chain planning 中，后处理逻辑应该在**每个 transition 独立应用**，而不是在整个 chain 上全局应用。具体来说，在 chain 的最终输出构建阶段（`all_chains` 赋值之后），对每个相邻的 `(waypoint[k], waypoint[k+1])` 对独立执行 contact 二值化和掩码约束。

### 2.4 Translation Augmentation 的兼容性

`compute_loss` 中的 translation augmentation 对所有 4 个 effector 的位置施加相同的随机平移。这与 change mask 是兼容的，因为平移不改变"哪个关节变了"的标签。**无需修改**。

### 2.5 数据增强（反转）的兼容性

`dataset.py` 中的 `_augment_data` 会将 `(initial, final)` 反转为 `(final, initial)` 作为增强。反转后，change label 仍然是同一个 effector（只是 contact 翻转方向相反）。**无需修改**。

---

## 3. 完整的文件修改清单

| 文件 | 改动类型 | 改动内容 |
|------|---------|---------|
| `models/denoiser/mlp.py` | **结构性修改** | 增加 `change_cls_head`，`forward()` 返回 3 个值 |
| `models/denoiser/transformer.py` | **兼容性修改** | `forward()` 返回 3 个值（第 3 个为 placeholder） |
| `models/diffusion.py` | **核心修改** | `__init__` 增加参数；`compute_loss` 增加分类损失 + 加权 MSE；`_cfg_predict` 适配 3 返回值；`sample_transition` 增加后处理；chain 方法适配 |
| `configs/unified_2080ti.yaml` | **配置新增** | `w_change_cls`, `w_changed_dim` |
| `scripts/train.py` | **日志适配** | 增加 `loss_change_cls` 的 logging |
| `scripts/sample.py` | **无需修改** | 后处理在 `diffusion.py` 内部完成 |

---

## 4. 验证标准

改动完成后，应满足以下验证标准：

1. **训练正常启动**：`python scripts/train.py --config configs/unified_2080ti.yaml --data dataset/seiko_talos_merged.json` 无报错
2. **Loss 下降**：`loss_change_cls` 应在前 100 epoch 内快速下降（分类任务简单）
3. **采样合法性**：`sample_transition` 的输出中，每个 transition 的 contact 变化恰好为 1 位翻转
4. **非目标关节不变**：`initial` 和 `final` 之间，非目标 effector 的位置差异 < 0.01m（归一化空间中）

---

## 5. 当前完整代码参考

为了确保你有完整的上下文，以下是当前各文件的关键代码。**请基于这些代码进行修改，不要从头重写**。

### 5.1 `models/denoiser/mlp.py`（完整）

```python
"""
MLP-based Unified Transition Denoiser
=======================================
Per-layer condition injection following stgl's ResidualTemporalBlock_dd pattern.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict

from .embeddings import TimestepEmbedder
from .conditioning import OverlapStateEncoder


class UnifiedTransitionDenoiserMLP(nn.Module):
    def __init__(
        self,
        config_dim: int,
        hidden_dims: list = [512, 512, 512, 512],
        time_embed_dim: int = 128,
        dropout: float = 0.1,
        act_fn: str = 'silu',
        use_dropout: bool = True,
        ovlp_feat_dim: int = 64,
    ):
        super().__init__()
        self.config_dim = config_dim
        self.ovlp_feat_dim = ovlp_feat_dim

        time_hidden = hidden_dims[0]
        self.time_embed_init = TimestepEmbedder(time_embed_dim, time_hidden)
        self.time_embed_final = TimestepEmbedder(time_embed_dim, time_hidden)

        self.init_ovlp_encoder = OverlapStateEncoder(config_dim, time_embed_dim, ovlp_feat_dim)
        self.final_ovlp_encoder = OverlapStateEncoder(config_dim, time_embed_dim, ovlp_feat_dim)

        self.init_inpat_token = nn.Parameter(torch.randn(1, ovlp_feat_dim))
        self.final_inpat_token = nn.Parameter(torch.randn(1, ovlp_feat_dim))

        cond_dim = time_hidden * 2 + ovlp_feat_dim * 2
        input_dim = config_dim * 2

        act_cls = {'relu': nn.ReLU, 'prelu': nn.PReLU, 'silu': nn.SiLU}[act_fn]
        layer_dims = [input_dim] + hidden_dims
        num_layers = len(layer_dims) - 1

        self.linears = nn.ModuleList()
        self.cond_projs = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(num_layers):
            self.linears.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            self.cond_projs.append(nn.Sequential(
                nn.SiLU(),
                nn.Linear(cond_dim, layer_dims[i + 1]),
            ))
            if i < num_layers - 1:
                self.acts.append(act_cls())
            else:
                self.acts.append(nn.Identity())
            if use_dropout and i < num_layers - 2:
                self.dropouts.append(nn.Dropout(p=dropout))
            else:
                self.dropouts.append(nn.Identity())

        self.output_init = nn.Linear(hidden_dims[-1], config_dim)
        self.output_final = nn.Linear(hidden_dims[-1], config_dim)

    def forward(
        self,
        initial_noisy: torch.Tensor,
        final_noisy: torch.Tensor,
        t_init: torch.Tensor,
        t_final: torch.Tensor,
        tj_cond: Optional[Dict] = None,
    ) -> tuple:
        B = initial_noisy.shape[0]
        device = initial_noisy.device

        t_init_emb = self.time_embed_init(t_init)
        t_final_emb = self.time_embed_final(t_final)

        if tj_cond is not None:
            init_ovlp_raw = self.init_ovlp_encoder(
                tj_cond['init_ovlp_state'], tj_cond['init_ovlp_t']
            )
            final_ovlp_raw = self.final_ovlp_encoder(
                tj_cond['final_ovlp_state'], tj_cond['final_ovlp_t']
            )
            init_side_feat = torch.zeros(B, self.ovlp_feat_dim, device=device)
            final_side_feat = torch.zeros(B, self.ovlp_feat_dim, device=device)

            m = tj_cond['init_cd_use_ovlp']
            if m.any():
                init_side_feat[m] = init_ovlp_raw[m]
            m = tj_cond['init_cd_use_inpat']
            if m.any():
                init_side_feat[m] = self.init_inpat_token.expand(B, -1)[m]

            m = tj_cond['final_cd_use_ovlp']
            if m.any():
                final_side_feat[m] = final_ovlp_raw[m]
            m = tj_cond['final_cd_use_inpat']
            if m.any():
                final_side_feat[m] = self.final_inpat_token.expand(B, -1)[m]
        else:
            init_side_feat = torch.zeros(B, self.ovlp_feat_dim, device=device)
            final_side_feat = torch.zeros(B, self.ovlp_feat_dim, device=device)

        cond = torch.cat([t_init_emb, t_final_emb, init_side_feat, final_side_feat], dim=-1)
        h = torch.cat([initial_noisy, final_noisy], dim=-1)

        for linear, cond_proj, act, dp in zip(
            self.linears, self.cond_projs, self.acts, self.dropouts
        ):
            h = dp(act(linear(h) + cond_proj(cond)))

        eps_init_pred = self.output_init(h)
        eps_final_pred = self.output_final(h)

        return eps_init_pred, eps_final_pred
```

### 5.2 `configs/unified_2080ti.yaml`（完整）

```yaml
data:
  left_foot_dim: 3
  right_foot_dim: 3
  left_hand_dim: 3
  right_hand_dim: 3
  contact_dim: 4
  config_dim: 16
  transition_dim: 32
  use_orientation: false
  slices:
    left_foot:  [0, 3]
    right_foot: [3, 6]
    left_hand:  [6, 9]
    right_hand: [9, 12]
    contacts:   [12, 16]
  normalize: true

model:
  architecture: "mlp"
  hidden_dim: 384
  num_heads: 8
  num_layers: 8
  dropout: 0.1
  time_embed_dim: 192
  mlp_hidden_dims: [768, 768, 768, 768]

diffusion:
  num_timesteps: 1000
  beta_start: 0.0001
  beta_end: 0.02
  beta_schedule: "cosine"
  prediction_type: "epsilon"
  use_ddim: true
  ddim_steps: 50

training:
  batch_size: 512
  learning_rate: 2.0e-4
  weight_decay: 1.0e-5
  num_epochs: 50000
  warmup_steps: 1000
  w_initial: 1.0
  w_final: 1.0
  p_uncond: 0.1
  use_ema: true
  ema_decay: 0.9999
  aug_trans_scale: 0.1
  log_interval: 50
  save_interval: 200
  eval_interval: 200

sampling:
  guidance_scale: 2.0
  chain:
    num_transitions: 3
    constraint_weight: 0.5
    mode: "parallel"
```

---

## 6. 执行指令

请按照以下顺序修改文件，并输出每个文件的**完整修改后代码**（不要只输出 diff）：

1. `models/denoiser/mlp.py` — 增加 change_cls_head，修改 forward 返回值
2. `models/denoiser/transformer.py` — 兼容性修改，forward 返回 3 值
3. `models/denoiser/__init__.py` — 如需修改
4. `models/diffusion.py` — 核心修改：loss、采样后处理、_cfg_predict 适配
5. `configs/unified_2080ti.yaml` — 新增配置项
6. `scripts/train.py` — 日志适配

**请确保**：
- 所有修改后的代码可以直接复制粘贴使用，无需额外调整
- 保持现有的代码风格和注释习惯
- 不要删除任何现有功能（CFG、chain planning、overlap conditioning 等）
- 每个文件输出完整代码，不要省略未修改的部分
