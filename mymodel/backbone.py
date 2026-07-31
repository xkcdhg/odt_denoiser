"""
Transformer backbone F(x, sigma) for the motion prior.

- Input  : noisy feature window (B, T, C) + noise level sigma (B, 1, 1)
- Output : raw network prediction (B, T, C), wrapped by EDM preconditioning
           in edm.py to form the denoiser D(x, sigma).

sigma enters two ways:
  1. A Fourier embedding of log(sigma) -> a conditioning vector.
  2. FiLM (feature-wise linear modulation): the conditioning vector produces
     per-channel scale/shift applied inside each transformer block.

Attention is bidirectional within the window (denoising benefits from seeing
both sides of a noisy point). Multi-agent is deliberately NOT built in for v1.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
from .config import Config


class SigmaEmbedding(nn.Module):
    """Fourier features of log(sigma) -> MLP -> conditioning vector."""
    def __init__(self, d_model: int, n_freqs: int = 32):
        super().__init__()
        self.register_buffer(
            "freqs", torch.randn(n_freqs) * 2.0 * math.pi, persistent=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(2 * n_freqs, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        # sigma: (B, 1, 1) -> (B,)
        log_sigma = sigma.reshape(-1).clamp_min(1e-8).log()
        ang = log_sigma[:, None] * self.freqs[None, :]
        emb = torch.cat([ang.sin(), ang.cos()], dim=-1)
        return self.mlp(emb)  # (B, d_model)


class FiLM(nn.Module):
    """Per-channel scale/shift from the sigma conditioning vector."""
    def __init__(self, d_model: int):
        super().__init__()
        self.to_scale_shift = nn.Linear(d_model, 2 * d_model)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.to_scale_shift(cond).chunk(2, dim=-1)
        return x * (1 + scale[:, None, :]) + shift[:, None, :]


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        d = cfg.model.d_model
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(
            d, cfg.model.n_heads, dropout=cfg.model.dropout, batch_first=True
        )
        self.film1 = FiLM(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, cfg.model.d_ff), nn.SiLU(),
            nn.Dropout(cfg.model.dropout),
            nn.Linear(cfg.model.d_ff, d),
        )
        self.film2 = FiLM(d)
        self.causal = cfg.model.causal_attention

    def forward(self, x, cond, attn_mask=None):
        h = self.film1(self.norm1(x), cond)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        h = self.film2(self.norm2(x), cond)
        x = x + self.ff(h)
        return x


class MotionTransformer(nn.Module):
    """Raw network F. Denoiser D wraps this in edm.py."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        C, d, T = cfg.data.n_channels, cfg.model.d_model, cfg.data.window_len
        self.in_proj = nn.Linear(C, d)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, d))
        nn.init.normal_(self.pos_emb, std=0.02)
        self.sigma_emb = SigmaEmbedding(d)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.model.n_layers)])
        self.norm_out = nn.LayerNorm(d)
        self.out_proj = nn.Linear(d, cfg.data.n_out_dims)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _attn_mask(self, T, device):
        if not self.cfg.model.causal_attention:
            return None
        m = torch.full((T, T), float("-inf"), device=device)
        return torch.triu(m, diagonal=1)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C), sigma: (B, 1, 1)
        B, T, _ = x.shape
        cond = self.sigma_emb(sigma)            # (B, d)
        h = self.in_proj(x) + self.pos_emb[:, :T, :]
        mask = self._attn_mask(T, x.device)
        for blk in self.blocks:
            h = blk(h, cond, attn_mask=mask)
        return self.out_proj(self.norm_out(h))  # (B, T, C)
