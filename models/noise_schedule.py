"""
Noise Schedule for DDPM / DDIM
==============================
Supports linear and cosine beta schedules.
"""

import torch
import numpy as np
import math


class NoiseSchedule:
    """Manages the noise schedule for the diffusion process."""

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule_type: str = "cosine",
        device: str = "cpu",
    ):
        self.num_timesteps = num_timesteps
        self.device = device

        if schedule_type == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule_type == "cosine":
            betas = self._cosine_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        self.betas = betas.to(device)
        self.alphas = (1.0 - self.betas).to(device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]]
        )

        # For q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # For posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

    def _cosine_schedule(self, T: int, s: float = 0.008) -> torch.Tensor:
        """Cosine schedule as proposed in 'Improved DDPM'."""
        steps = torch.arange(T + 1, dtype=torch.float64)
        f = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = f / f[0]
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        betas = torch.clamp(betas, min=0.0001, max=0.9999)
        return betas.float()

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """
        Forward diffusion: q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps

        Args:
            x_0: (B, D) clean data
            t:   (B,) timestep indices
            noise: (B, D) optional pre-sampled noise
        Returns:
            x_t: (B, D) noisy data
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t].unsqueeze(-1)        # (B, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)  # (B, 1)

        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """Predict x_0 from x_t and predicted noise."""
        sqrt_alpha = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        return (x_t - sqrt_one_minus_alpha * eps) / sqrt_alpha

    def q_posterior_mean(self, x_0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute posterior mean: mu(x_t, x_0)."""
        coef1 = self.posterior_mean_coef1[t].unsqueeze(-1)
        coef2 = self.posterior_mean_coef2[t].unsqueeze(-1)
        return coef1 * x_0 + coef2 * x_t

    def p_sample_ddpm(
        self, x_t: torch.Tensor, t: torch.Tensor, predicted_noise: torch.Tensor
    ) -> torch.Tensor:
        """One step of DDPM reverse process."""
        x_0_pred = self.predict_x0_from_eps(x_t, t, predicted_noise)
        x_0_pred = torch.clamp(x_0_pred, -10.0, 10.0)

        mean = self.q_posterior_mean(x_0_pred, x_t, t)

        if t[0] > 0:
            noise = torch.randn_like(x_t)
            variance = self.posterior_variance[t].unsqueeze(-1)
            return mean + torch.sqrt(variance) * noise
        else:
            return mean

    def ddim_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        predicted_noise: torch.Tensor,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """One step of DDIM reverse process."""
        alpha_t = self.alphas_cumprod[t].unsqueeze(-1)
        alpha_prev = self.alphas_cumprod[t_prev].unsqueeze(-1) if t_prev[0] >= 0 else torch.ones_like(alpha_t)

        # Predict x_0
        x_0_pred = (x_t - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
        x_0_pred = torch.clamp(x_0_pred, -10.0, 10.0)

        # Compute sigma
        sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev))

        # Direction pointing to x_t
        dir_xt = torch.sqrt(1 - alpha_prev - sigma ** 2) * predicted_noise

        # Noise
        noise = torch.randn_like(x_t) if eta > 0 and t[0] > 0 else torch.zeros_like(x_t)

        x_prev = torch.sqrt(alpha_prev) * x_0_pred + dir_xt + sigma * noise
        return x_prev

    def to(self, device):
        """Move all tensors to device."""
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        self.posterior_log_variance = self.posterior_log_variance.to(device)
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(device)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(device)
        return self
