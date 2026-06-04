# Mini VLA 快速上手指南

作者：**Manus AI**

## 1. 目标与定位

本仓库新增的 `mini_vla/` 模块提供一个从零可运行的 **Vision-Language-Action（VLA）最小闭环**。它不是为了立即达到 OpenVLA 或 SmolVLA 的规模，而是为了帮助你快速掌握 VLA 系统的关键工程边界：图像 token、语言指令 token、机器人状态 token、跨模态融合、连续 action chunk 预测、synthetic pretrain 以及基于开源机器人数据的后训练。OpenVLA 将视觉语言模型与机器人动作输出结合，用于跨任务机器人控制；LeRobot 则提供了面向真实世界机器人学习的数据集、模型和训练工具生态；SmolVLA 进一步说明了轻量 VLA 在消费级硬件与开源数据上的可行性。[1] [2] [3]

> 本实现的核心原则是先完成 **可跑通、可调试、可替换** 的工程骨架，再逐步把视觉编码器、语言骨干、动作头和数据源替换为更强组件。

## 2. 目录结构

| 路径 | 作用 |
| --- | --- |
| `mini_vla/models/` | Mini VLA 模型实现，包括 patch vision encoder、simple text encoder、fusion transformer 与 continuous action head。 |
| `mini_vla/data/` | synthetic pretrain 数据集、hash tokenizer、batch collator、local JSONL 与 Hugging Face/LeRobot 风格机器人数据适配器。 |
| `mini_vla/training/` | 配置加载、checkpoint、单机训练循环与验证逻辑。 |
| `configs/mini_vla_pretrain.yaml` | 可直接运行的 synthetic pretrain 配置。 |
| `configs/mini_vla_finetune_lerobot.yaml` | 面向 Hugging Face/LeRobot 风格数据集的后训练配置。 |
| `scripts/pretrain_mini_vla.py` | 预训练入口。 |
| `scripts/finetune_mini_vla.py` | 后训练入口，可从预训练 checkpoint 初始化。 |
| `docs/mini_vla_design.md` | 架构设计文档。 |

## 3. 模型结构

Mini VLA 的输入是 RGB 图像、语言指令和可选 proprioceptive state，输出是连续动作 chunk。与 OpenVLA 一类系统相比，本实现没有直接依赖大型 VLM checkpoint，而是采用轻量可训练模块模拟 VLA 的端到端数据流，这样可以先验证预训练和后训练管线。[1]

| 模块 | 当前实现 | 后续可替换方向 |
| --- | --- | --- |
| Vision Encoder | Conv2d patch embedding，支持单视角与多视角输入。 | SigLIP、DINOv2、CLIP、ViT adapter。 |
| Language Encoder | 固定 vocab 的 deterministic hash tokenizer 加轻量 Transformer。 | Llama/Qwen/TinyLlama tokenizer 与 causal backbone。 |
| Fusion Backbone | 带类型 embedding 的 Transformer Encoder。 | Flamingo-style cross-attention、Perceiver Resampler、LLM prefix tuning。 |
| Action Head | MLP 输出 `[chunk_size, action_dim]` 连续动作。 | Diffusion action head、flow matching head、离散 action token。 |
| Dataset | Synthetic 几何图像预训练，本地 JSONL 与 HF/LeRobot adapter 后训练。 | Open X-Embodiment、BridgeData、LIBERO、ALOHA 等真实机器人数据。 |

## 4. 环境准备

在租用 GPU 机器上，建议先进入仓库根目录安装基础依赖。当前代码依赖 PyTorch、Pillow、PyYAML 和 NumPy；如果要直接读取 Hugging Face 数据集，还需要安装 `datasets`。

```bash
cd /path/to/unified
pip install torch pillow pyyaml numpy
pip install datasets  # 仅当你要使用 Hugging Face / LeRobot 数据集时需要
```

如果 GPU 支持 BF16，默认配置会启用 `amp_dtype: bf16`。如果你的 GPU 只适合 FP16，可以把配置改成 `amp_dtype: fp16`；如果显存紧张，优先降低 `model.embed_dim`、`train.batch_size`、`model.image_size` 和 `model.fusion_layers`。

## 5. 预训练：先跑通完整 pipeline

预训练阶段默认使用 synthetic 数据。这个数据集会生成带颜色几何物体的图像、自然语言指令、机器人状态向量以及连续动作 chunk。它的目的不是模拟真实机器人，而是提供一个确定性且可学习的 VLA 信号，用于确认模型、数据、loss、优化器、验证和 checkpoint 全链路可运行。

```bash
cd /path/to/unified
python scripts/pretrain_mini_vla.py --config configs/mini_vla_pretrain.yaml
```

常用调试命令如下。第一条适合 CPU 或小 GPU smoke test，第二条适合你租到 GPU 后做稍长的预训练。

```bash
python scripts/pretrain_mini_vla.py \
  --config configs/mini_vla_pretrain.yaml \
  --override train.device=cpu \
  --override train.steps=5 \
  --override train.batch_size=4 \
  --override data.train.num_samples=64

python scripts/pretrain_mini_vla.py \
  --config configs/mini_vla_pretrain.yaml \
  --override train.steps=5000 \
  --override train.batch_size=64 \
  --override model.embed_dim=256 \
  --override train.output_dir=checkpoints/mini_vla_pretrain_5k
```

训练结束后，checkpoint 会保存到 `train.output_dir`，包括 `checkpoint_last.pt`、周期性 checkpoint，以及存在验证集时的 `checkpoint_best.pt`。

## 6. 后训练：接入开源机器人数据

后训练脚本支持 Hugging Face/LeRobot 风格数据集，也支持本地 JSONL manifest。LeRobot 项目将机器人数据集、策略训练和评估工具开源化，适合作为 Mini VLA 的真实数据入口；Open X-Embodiment 说明了跨机器人、多任务数据规模对通用机器人策略的重要性。[2] [4]

```bash
python scripts/finetune_mini_vla.py \
  --config configs/mini_vla_finetune_lerobot.yaml \
  --override train.pretrained_checkpoint=checkpoints/mini_vla_pretrain_debug/checkpoint_last.pt
```

由于不同开源数据集的字段命名不完全一致，配置里提供了 `image_column`、`instruction_column`、`action_column` 和 `state_column`。如果默认列名不匹配，可以用命令行覆盖：

```bash
python scripts/finetune_mini_vla.py \
  --config configs/mini_vla_finetune_lerobot.yaml \
  --override data.name=lerobot/pusht \
  --override data.image_column=observation.image \
  --override data.action_column=action \
  --override data.state_column=observation.state \
  --override data.instruction_column=task
```

如果你下载或整理了自己的机器人数据，可以使用 `local_jsonl` 格式。每一行 JSON 至少需要包含图像路径和动作，建议包含语言指令与状态：

```json
{"image_path":"images/000001.png","instruction":"pick the red cube","state":[0.1,0.2,0.0,0.0,0.0,0.0,1.0,0.0],"actions":[[0.01,0.02,0.0,0.0,0.0,0.0,1.0]]}
```

对应配置可以这样覆盖：

```bash
python scripts/finetune_mini_vla.py \
  --config configs/mini_vla_finetune_lerobot.yaml \
  --override data.type=local_jsonl \
  --override data.train.path=/path/to/train.jsonl \
  --override data.val.path=/path/to/val.jsonl \
  --override train.pretrained_checkpoint=checkpoints/mini_vla_pretrain_debug/checkpoint_last.pt
```

## 7. 推荐实验路线

| 阶段 | 目标 | 建议设置 | 判断标准 |
| --- | --- | --- | --- |
| Smoke Test | 验证代码无语法/shape 错误。 | `steps=5, batch_size=4, device=cpu`。 | 可以生成 `checkpoint_last.pt`。 |
| Synthetic Pretrain | 让模型学会简单视觉-语言-动作映射。 | `steps=2k-10k, embed_dim=128/256`。 | 训练 loss 明显下降，验证 loss 稳定。 |
| Real-data Finetune | 用开源机器人数据适配真实观测和动作。 | 从 pretrain checkpoint 初始化，较小 LR。 | loss 能下降，action 统计与数据集范围一致。 |
| Architecture Upgrade | 替换视觉/语言 backbone 或 action head。 | 保持 batch schema 不变，逐个替换模块。 | 新模块通过同一训练脚本跑通。 |

## 8. 当前限制与下一步

当前 Mini VLA 是一个工程骨架，重点在于训练闭环完整，而不是 SOTA 性能。它还没有实现真实机器人评估、分布式训练、LLM tokenizer、diffusion action head、action normalization 文件和多数据集混合采样。下一步建议优先补充真实数据的 action 归一化与反归一化、加入多视角图像字段、把 language encoder 替换为小型 pretrained LM，并在后训练阶段引入 LeRobot/Open X-Embodiment 子集。

## References

[1]: https://arxiv.org/abs/2406.09246 "OpenVLA: An Open-Source Vision-Language-Action Model"
[2]: https://huggingface.co/docs/lerobot/en/index "LeRobot Documentation"
[3]: https://huggingface.co/docs/lerobot/en/smolvla "SmolVLA Documentation"
[4]: https://robotics-transformer-x.github.io/ "Open X-Embodiment / RT-X Project"
