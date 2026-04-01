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
    """
    MLP-based unified denoiser with per-layer condition injection.

    Following stgl's architecture: at each hidden layer, the global condition
    vector is projected and ADDED to the hidden features. This is analogous to
    ResidualTemporalBlock_dd's `out = block(x) + time_mlp(cond)` pattern, where
    the condition influences every layer of the network — not just the input.

    Per-side conditioning is mutually exclusive: for each side (init/final),
    exactly one of {overlap, inpaint, drop} is active, occupying a single
    feature slot of size ovlp_feat_dim.

    Structure:
      1. Timestep embedding: t_init, t_final → emb vectors
      2. Per-side conditioning: mutually exclusive ovlp/inpat/drop → side_feat
      3. Global condition: [t_init_emb, t_final_emb, init_side_feat, final_side_feat]
      4. MLP input: [initial_noisy, final_noisy] (data only)
      5. Per-layer: h = act(linear(h) + cond_proj(cond))   ← condition injected every layer
      6. Two output heads: separate Linear for eps_init and eps_final
    """

    def __init__(
        self,
        config_dim: int,
        hidden_dims: list = [512, 512, 512, 512],
        time_embed_dim: int = 128,
        dropout: float = 0.1,
        act_fn: str = 'silu',       # 'relu' / 'prelu' / 'silu'
        use_dropout: bool = True,
        ovlp_feat_dim: int = 64,
    ):
        super().__init__()
        self.config_dim = config_dim
        self.ovlp_feat_dim = ovlp_feat_dim

        # ---- Timestep embedders (independent for init and final) ----
        time_hidden = hidden_dims[0]
        self.time_embed_init = TimestepEmbedder(time_embed_dim, time_hidden)
        self.time_embed_final = TimestepEmbedder(time_embed_dim, time_hidden)

        # ---- Overlap state encoders (stgl-style) ----
        self.init_ovlp_encoder = OverlapStateEncoder(config_dim, time_embed_dim, ovlp_feat_dim)
        self.final_ovlp_encoder = OverlapStateEncoder(config_dim, time_embed_dim, ovlp_feat_dim)

        # ---- Inpaint tokens (same dim as ovlp_feat for mutually exclusive slot) ----
        self.init_inpat_token = nn.Parameter(torch.randn(1, ovlp_feat_dim))
        self.final_inpat_token = nn.Parameter(torch.randn(1, ovlp_feat_dim))

        # ---- Condition dim: [t_init_emb, t_final_emb, init_side_feat, final_side_feat] ----
        cond_dim = time_hidden * 2 + ovlp_feat_dim * 2

        # ---- MLP input: just [initial_noisy, final_noisy] ----
        input_dim = config_dim * 2

        # ---- Build per-layer MLP with condition injection ----
        act_cls = {'relu': nn.ReLU, 'prelu': nn.PReLU, 'silu': nn.SiLU}[act_fn]
        layer_dims = [input_dim] + hidden_dims
        num_layers = len(layer_dims) - 1

        self.linears = nn.ModuleList()
        self.cond_projs = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(num_layers):
            self.linears.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            # Per-layer condition projection (analogous to ResBlock's time_mlp)
            self.cond_projs.append(nn.Sequential(
                nn.SiLU(),
                nn.Linear(cond_dim, layer_dims[i + 1]),
            ))
            if i < num_layers - 1:
                self.acts.append(act_cls())
            else:
                self.acts.append(nn.Identity())  # no activation on last hidden layer
            if use_dropout and i < num_layers - 2:
                self.dropouts.append(nn.Dropout(p=dropout))
            else:
                self.dropouts.append(nn.Identity())

        # ---- Two output heads ----
        self.output_init = nn.Linear(hidden_dims[-1], config_dim)
        self.output_final = nn.Linear(hidden_dims[-1], config_dim)

    def forward(
        self,
        initial_noisy: torch.Tensor,   # (B, D)
        final_noisy: torch.Tensor,      # (B, D)
        t_init: torch.Tensor,           # (B,)
        t_final: torch.Tensor,          # (B,)
        tj_cond: Optional[Dict] = None, # stgl-style overlap/inpaint conditioning
    ) -> tuple:
        """
        Per-layer conditioned forward pass.

        Returns:
            eps_init_pred (B, D): predicted noise for initial
            eps_final_pred (B, D): predicted noise for final
        """
        B = initial_noisy.shape[0]
        device = initial_noisy.device

        # Timestep embeddings
        t_init_emb = self.time_embed_init(t_init)
        t_final_emb = self.time_embed_final(t_final)

        # Per-side conditioning (mutually exclusive: ovlp / inpat / drop)
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


        # Global condition vector
        cond = torch.cat([t_init_emb, t_final_emb, init_side_feat, final_side_feat], dim=-1)

        # MLP input: just the noisy data
        h = torch.cat([initial_noisy, final_noisy], dim=-1)

        # Per-layer forward with condition injection at every layer
        """
         for each layer:
            1.process x only
            2.process cond only
            3.add them together
            4.apply activation and dropout
            """
        for linear, cond_proj, act, dp in zip(
            self.linears, self.cond_projs, self.acts, self.dropouts
        ):
            h = dp(act(linear(h) + cond_proj(cond)))

        # Predict noise for both parts
        eps_init_pred = self.output_init(h)    # (B, D)
        eps_final_pred = self.output_final(h)  # (B, D)

        return eps_init_pred, eps_final_pred
