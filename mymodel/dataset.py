"""
ETH/UCY trajectory windowing.

ETH/UCY ground-truth format (the common preprocessed form): whitespace-separated
rows of  [frame_id, ped_id, x, y]  in world metres, one row per pedestrian per
observed frame, ~2.5 Hz. Files: eth, hotel, univ, zara1, zara2.

This loader:
  1. Groups rows by ped_id into full trajectories.
  2. Slides a window of length cfg.data.window_len (stride 1) over each
     trajectory, producing clean (T, 2) position windows.
  3. Returns them as a tensor dataset of shape (N, T, 2).

The clean POSITIONS are the training target basis; corruption.make_training_pair
injects noise and builds features at train time (so each epoch sees fresh noise).

A synthetic generator is provided so the whole pipeline can be validated before
the real files are present. Replace load_ethucy(...) path with your data dir.
"""
from __future__ import annotations
import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from .config import Config


def _windows_from_traj(xy: np.ndarray, T: int):
    """xy: (L, 2) one pedestrian's full path -> list of (T,2) windows, stride 1."""
    out = []
    for s in range(0, len(xy) - T + 1):
        out.append(xy[s:s + T])
    return out


def load_ethucy(data_dir: str, cfg: Config) -> np.ndarray:
    """Read all ETH/UCY txt files in data_dir, return (N, T, 2) clean windows."""
    T = cfg.data.window_len
    files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    if not files:
        raise FileNotFoundError(f"No .txt trajectory files in {data_dir}")
    windows = []
    for fp in files:
        rows = np.loadtxt(fp)
        if rows.ndim == 1:
            rows = rows[None, :]
        # columns: frame, ped, x, y
        for ped in np.unique(rows[:, 1]):
            traj = rows[rows[:, 1] == ped]
            traj = traj[np.argsort(traj[:, 0])]        # sort by frame
            xy = traj[:, 2:4].astype(np.float32)
            windows.extend(_windows_from_traj(xy, T))
    if not windows:
        raise ValueError("No windows produced; check window_len vs trajectory lengths.")
    return np.stack(windows, axis=0)


def synthetic_windows(n: int, cfg: Config, seed: int = 0) -> np.ndarray:
    """Smooth, human-like synthetic walks for smoke-testing the pipeline.

    Each walk: near-constant velocity with slow heading drift + occasional stop.
    Produces (n, T, 2) clean position windows in metres.
    """
    rng = np.random.default_rng(seed)
    T = cfg.data.window_len
    dt = 1.0 / cfg.data.fps
    out = np.zeros((n, T, 2), dtype=np.float32)
    for k in range(n):
        speed = rng.uniform(0.5, 1.6)                  # m/s, human walking
        heading = rng.uniform(0, 2 * np.pi)
        turn_rate = rng.normal(0, 0.15)                # rad/step
        pos = np.zeros(2, dtype=np.float32)
        stop_at = rng.integers(0, T) if rng.random() < 0.3 else -1
        for t in range(T):
            out[k, t] = pos
            v = 0.0 if t == stop_at else speed
            heading += turn_rate + rng.normal(0, 0.05)
            pos = pos + dt * v * np.array([np.cos(heading), np.sin(heading)],
                                          dtype=np.float32)
    return out


class WindowDataset(Dataset):
    """Serves clean (T, 2) position windows. Corruption happens in the train loop."""
    def __init__(self, windows: np.ndarray):
        self.x = torch.from_numpy(windows).float()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i]
