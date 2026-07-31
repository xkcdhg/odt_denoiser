"""
EDM (Karras et al. 2022) preconditioning, denoiser wrapper, and loss.

This is the piece that converts your Table-4 result (RK4 inert on a DDPM score)
into Table-6 behaviour (RK4 wins at low NFE). The preconditioning keeps the
network's input and target unit-variance across ALL sigma, giving a smooth,
sigma-parameterized score field whose probability-flow ODE is well conditioned
for a higher-order solver like RK4.

Denoiser:
    D(x, sigma) = c_skip(sigma) * x + c_out(sigma) * F(c_in(sigma) * x, c_noise(sigma))

Training target is the clean signal x0; loss is weighted MSE  lambda(sigma) * ||D - x0||^2
with the EDM weighting that makes the effective target unit-variance at every sigma.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from .config import Config
from .backbone import MotionTransformer


class EDMDenoiser(nn.Module):
    """Wraps the raw transformer F into the EDM-preconditioned denoiser D."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.sigma_data = cfg.edm.sigma_data
        self.net = MotionTransformer(cfg)

    def _coeffs(self, sigma: torch.Tensor):
        sd2 = self.sigma_data ** 2
        s2 = sigma ** 2
        c_skip = sd2 / (s2 + sd2)
        c_out = sigma * self.sigma_data / (s2 + sd2).sqrt()
        c_in = 1.0 / (s2 + sd2).sqrt()
        # c_noise: EDM feeds log(sigma)/4 as the conditioning scalar; our
        # SigmaEmbedding already takes log(sigma), so pass sigma straight through.
        return c_skip, c_out, c_in

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """D(x, sigma).

        x:     (B, T, C) noisy INPUT features [pos(,vel)(,acc)]
        sigma: (B, 1, 1)
        returns denoised POSITION (B, T, n_pos_dims).

        EDM preconditioning: the skip term operates on the position channels of
        the input (the quantity being denoised); the network sees all input
        channels (scaled by c_in) so attention can use velocity/acceleration.
        """
        npos = self.cfg.data.n_pos_dims
        c_skip, c_out, c_in = self._coeffs(sigma)
        f = self.net(c_in * x, sigma)                 # (B, T, n_pos_dims)
        x_pos = x[..., :npos]                          # position channels only
        return c_skip * x_pos + c_out * f


def edm_loss(model: EDMDenoiser, clean_pos_feat: torch.Tensor,
             x_noisy: torch.Tensor, sigma: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Weighted denoising loss on POSITION only.

    clean_pos_feat: (B, T, n_pos_dims) clean position offsets (the target).
    x_noisy:        (B, T, C) noisy input features.
    sigma:          (B, 1, 1).
    lambda(sigma) = (sigma^2 + sigma_data^2) / (sigma*sigma_data)^2 (EDM Eq. 8).
    """
    sd = cfg.edm.sigma_data
    weight = (sigma ** 2 + sd ** 2) / ((sigma * sd) ** 2 + 1e-12)  # (B,1,1)
    pred = model(x_noisy, sigma)                  # (B, T, n_pos_dims)
    se = (pred - clean_pos_feat) ** 2
    return (weight * se).mean()


def edm_sigma_schedule(n_steps: int, cfg: Config, device) -> torch.Tensor:
    """EDM sigma grid for sampling (rho-curved), from sigma_max down to sigma_min,
    with a trailing 0. Used by the RK4/ADSS sampler later. Returned length n_steps+1.
    """
    rho = cfg.edm.rho
    smin, smax = cfg.edm.sigma_min, cfg.edm.sigma_max
    i = torch.arange(n_steps, device=device)
    a = smax ** (1 / rho)
    b = smin ** (1 / rho)
    sigmas = (a + i / (n_steps - 1) * (b - a)) ** rho
    return torch.cat([sigmas, torch.zeros(1, device=device)])
