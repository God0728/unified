"""Main Mini VLA model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .action_head import ActionHeadConfig, ContinuousActionHead
from .fusion import FusionConfig, FusionTransformer
from .text import SimpleTextEncoder, TextConfig
from .vision import PatchVisionEncoder, VisionConfig


@dataclass
class StateConfig:
    """Configuration for proprioceptive-state encoding."""

    state_dim: int = 0
    embed_dim: int = 256
    hidden_dim: int = 256
    dropout: float = 0.1


@dataclass
class MiniVLAConfig:
    """Full model configuration."""

    vision: VisionConfig = field(default_factory=VisionConfig)
    text: TextConfig = field(default_factory=TextConfig)
    state: StateConfig = field(default_factory=StateConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    action_head: ActionHeadConfig = field(default_factory=ActionHeadConfig)
    loss_type: str = "l1"

    def validate(self) -> None:
        dims = {
            self.vision.embed_dim,
            self.text.embed_dim,
            self.fusion.embed_dim,
            self.action_head.embed_dim,
        }
        if self.state.state_dim > 0:
            dims.add(self.state.embed_dim)
        if len(dims) != 1:
            raise ValueError(f"all embed dims must match, got {sorted(dims)}")
        if self.loss_type not in {"l1", "mse", "smooth_l1"}:
            raise ValueError("loss_type must be one of: l1, mse, smooth_l1")


class StateEncoder(nn.Module):
    """Encode robot proprioceptive state into a single token."""

    def __init__(self, config: StateConfig):
        super().__init__()
        self.config = config
        if config.state_dim <= 0:
            self.net = None
        else:
            self.net = nn.Sequential(
                nn.Linear(config.state_dim, config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.embed_dim),
                nn.LayerNorm(config.embed_dim),
            )

    def forward(self, state: torch.Tensor | None, batch_size: int, device: torch.device) -> torch.Tensor | None:
        if self.net is None:
            return None
        if state is None:
            state = torch.zeros(batch_size, self.config.state_dim, device=device)
        if state.ndim != 2 or state.shape[-1] != self.config.state_dim:
            raise ValueError(
                f"state must have shape [B,{self.config.state_dim}], got {tuple(state.shape)}"
            )
        return self.net(state).unsqueeze(1)


class MiniVLA(nn.Module):
    """A compact Vision-Language-Action policy.

    The model consumes RGB observations, tokenized instructions and optional
    robot state, then predicts a continuous action chunk.
    """

    def __init__(self, config: MiniVLAConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.vision_encoder = PatchVisionEncoder(config.vision)
        self.text_encoder = SimpleTextEncoder(config.text)
        self.state_encoder = StateEncoder(config.state)
        self.fusion = FusionTransformer(config.fusion)
        self.action_head = ContinuousActionHead(config.action_head)

    @property
    def action_dim(self) -> int:
        return self.config.action_head.action_dim

    @property
    def chunk_size(self) -> int:
        return self.config.action_head.chunk_size

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        vision_tokens = self.vision_encoder(images)
        text_tokens = self.text_encoder(input_ids, attention_mask)
        state_tokens = self.state_encoder(state, images.shape[0], images.device)
        fused_tokens, pooled = self.fusion(
            text_tokens=text_tokens,
            vision_tokens=vision_tokens,
            state_tokens=state_tokens,
            text_attention_mask=attention_mask,
        )
        pred_actions = self.action_head(pooled)
        output: dict[str, torch.Tensor] = {
            "actions_pred": pred_actions,
            "pooled": pooled,
            "fused_tokens": fused_tokens,
        }
        if actions is not None:
            output["loss"] = self.compute_loss(pred_actions, actions)
        return output

    def compute_loss(self, pred_actions: torch.Tensor, target_actions: torch.Tensor) -> torch.Tensor:
        if target_actions.shape != pred_actions.shape:
            raise ValueError(
                f"target actions shape {tuple(target_actions.shape)} does not match "
                f"predicted shape {tuple(pred_actions.shape)}"
            )
        if self.config.loss_type == "l1":
            return F.l1_loss(pred_actions, target_actions)
        if self.config.loss_type == "mse":
            return F.mse_loss(pred_actions, target_actions)
        return F.smooth_l1_loss(pred_actions, target_actions)

    def predict_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Convenience wrapper for inference."""
        self.eval()
        with torch.no_grad():
            out = self(
                images=batch["images"],
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                state=batch.get("state"),
            )
        return out["actions_pred"]


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    params = model.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def build_mini_vla_from_dict(config_dict: dict[str, Any]) -> MiniVLA:
    """Build ``MiniVLA`` from a nested dictionary.

    This helper keeps training scripts independent from external configuration
    libraries such as Hydra or OmegaConf.
    """

    model_cfg = config_dict.get("model", config_dict)
    embed_dim = int(model_cfg.get("embed_dim", 256))
    action_dim = int(model_cfg.get("action_dim", 7))
    chunk_size = int(model_cfg.get("chunk_size", 8))
    image_size = int(model_cfg.get("image_size", 128))
    patch_size = int(model_cfg.get("patch_size", 16))
    max_text_len = int(model_cfg.get("max_text_len", 64))
    vocab_size = int(model_cfg.get("vocab_size", 4096))
    state_dim = int(model_cfg.get("state_dim", 0))
    dropout = float(model_cfg.get("dropout", 0.1))

    cfg = MiniVLAConfig(
        vision=VisionConfig(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            max_views=int(model_cfg.get("max_views", 4)),
            dropout=dropout,
        ),
        text=TextConfig(
            vocab_size=vocab_size,
            max_length=max_text_len,
            embed_dim=embed_dim,
            num_layers=int(model_cfg.get("text_layers", 2)),
            num_heads=int(model_cfg.get("num_heads", 4)),
            ff_dim=int(model_cfg.get("ff_dim", 4 * embed_dim)),
            dropout=dropout,
        ),
        state=StateConfig(
            state_dim=state_dim,
            embed_dim=embed_dim,
            hidden_dim=int(model_cfg.get("state_hidden_dim", embed_dim)),
            dropout=dropout,
        ),
        fusion=FusionConfig(
            embed_dim=embed_dim,
            num_layers=int(model_cfg.get("fusion_layers", 4)),
            num_heads=int(model_cfg.get("num_heads", 4)),
            ff_dim=int(model_cfg.get("ff_dim", 4 * embed_dim)),
            dropout=dropout,
            use_act_token=bool(model_cfg.get("use_act_token", True)),
        ),
        action_head=ActionHeadConfig(
            embed_dim=embed_dim,
            hidden_dim=int(model_cfg.get("action_hidden_dim", 2 * embed_dim)),
            action_dim=action_dim,
            chunk_size=chunk_size,
            dropout=dropout,
        ),
        loss_type=str(model_cfg.get("loss_type", "l1")),
    )
    return MiniVLA(cfg)
