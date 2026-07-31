"""
ADE / FDE for trajectory denoising, in metres.

Standard definitions (same as used across ETH/UCY trajectory literature, so the
numbers are directly comparable to published tables):

    ADE = mean over all frames of  || pred_t - gt_t ||_2
    FDE = || pred_T - gt_T ||_2        (last frame of the window)

Note on frames of reference: because both prediction and ground truth are
expressed relative to the SAME window origin (pos - pos[0]), the subtraction
cancels in the difference. ADE/FDE computed on origin-relative offsets are
numerically identical to ADE/FDE on absolute world positions. No conversion
needed.

One caveat when comparing to prediction papers: their ADE/FDE is over a FUTURE
horizon. Ours is over the denoised window, since this is a denoiser, not a
predictor. Same formula, different set of frames — state that when reporting.
"""
from __future__ import annotations
import torch


def ade(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """pred, gt: (B, T, 2) -> scalar mean L2 error over all frames, in metres."""
    return torch.linalg.norm(pred - gt, dim=-1).mean()


def fde(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """pred, gt: (B, T, 2) -> scalar L2 error at the final frame, in metres."""
    return torch.linalg.norm(pred[:, -1, :] - gt[:, -1, :], dim=-1).mean()


def ade_fde(pred: torch.Tensor, gt: torch.Tensor) -> tuple[float, float]:
    """Convenience: returns (ADE, FDE) as floats in metres."""
    return float(ade(pred, gt).item()), float(fde(pred, gt).item())


def per_frame_ade(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """(B, T, 2) -> (T,) mean L2 error at each frame index.

    Useful for the delayed-freeze design: it shows how error varies across the
    window, so you can verify that interior frames (which have two-sided
    context) are denoised better than the leading edge. That gap is the
    empirical justification for freezing with lag k rather than at the edge.
    """
    return torch.linalg.norm(pred - gt, dim=-1).mean(dim=0)
