"""
Unified Transition Denoiser Package
=====================================
Modular denoiser components for the unified diffusion model.

Modules:
  - embeddings:    SinusoidalPosEmb, TimestepEmbedder
  - conditioning:  OverlapStateEncoder, AdaLayerNorm
  - transformer:   TransformerBlock, UnifiedTransitionDenoiser
  - mlp:           UnifiedTransitionDenoiserMLP
"""

import torch.nn as nn

from .embeddings import SinusoidalPosEmb, TimestepEmbedder
from .conditioning import OverlapStateEncoder, AdaLayerNorm
from .transformer import TransformerBlock, UnifiedTransitionDenoiser
from .mlp import UnifiedTransitionDenoiserMLP


def build_denoiser(config: dict) -> nn.Module:
    """Build denoiser from config dict."""
    model_cfg = config['model']
    data_cfg = config['data']

    ovlp_feat_dim = model_cfg.get('ovlp_feat_dim', 64)

    if model_cfg['architecture'] == 'transformer':
        return UnifiedTransitionDenoiser(
            config_dim=data_cfg['config_dim'],
            hidden_dim=model_cfg['hidden_dim'],
            num_heads=model_cfg['num_heads'],
            num_layers=model_cfg['num_layers'],
            time_embed_dim=model_cfg['time_embed_dim'],
            dropout=model_cfg['dropout'],
            ovlp_feat_dim=ovlp_feat_dim,
        )
    elif model_cfg['architecture'] == 'mlp':
        return UnifiedTransitionDenoiserMLP(
            config_dim=data_cfg['config_dim'],
            hidden_dims=model_cfg['mlp_hidden_dims'],
            time_embed_dim=model_cfg['time_embed_dim'],
            dropout=model_cfg.get('dropout', 0.1),
            act_fn=model_cfg.get('act_fn', 'silu'),
            use_dropout=model_cfg.get('use_dropout', True),
            ovlp_feat_dim=ovlp_feat_dim,
        )
    else:
        raise ValueError(f"Unknown architecture: {model_cfg['architecture']}")


__all__ = [
    'SinusoidalPosEmb',
    'TimestepEmbedder',
    'OverlapStateEncoder',
    'AdaLayerNorm',
    'TransformerBlock',
    'UnifiedTransitionDenoiser',
    'UnifiedTransitionDenoiserMLP',
    'build_denoiser',
]
