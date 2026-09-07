"""
Evaluate a trained motion-prior checkpoint on a held-out split.
Loads EMA weights, builds windows via load_ethucy, reports ADE/FDE
(noisy -> denoised) at fixed probe sigmas. Test-split run = the real
generalization measure (training eval was on train windows).

Run:
    python -m mymodel.eval_test --ckpt checkpoints/final.pt --data_dir /path/to/split
"""
from __future__ import annotations
import argparse
import torch

from .config import Config
from .edm import EDMDenoiser
from .dataset import load_ethucy
from .train import eval_ade_fde


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data_dir", type=str, required=True,
                    help="Dir with *.txt (frame ped x y), one level deep.")
    ap.add_argument("--n", type=int, default=100000,
                    help="Max windows to eval (all of them if fewer).")
    ap.add_argument("--sigmas", type=float, nargs="+",
                    default=[0.1, 0.2, 0.4, 0.8])
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg: Config = ckpt["cfg"]           # exact config the model trained with
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = EDMDenoiser(cfg).to(device)
    model.load_state_dict(ckpt["ema"])  # EMA = the frozen prior
    model.eval()

    windows = load_ethucy(args.data_dir, cfg)
    print(f"loaded {len(windows)} test windows from {args.data_dir}")
    print(f"ckpt step {ckpt['step']} | sigma_data {cfg.edm.sigma_data:.4f} "
          f"| device {device}")

    n = min(args.n, len(windows))
    out = eval_ade_fde(model, windows, cfg, device, n=n,
                       sigmas=tuple(args.sigmas))

    print(f"\nADE/FDE metres on {n} TEST windows (noisy -> denoised)")
    print(f"{'sigma':>6} | {'ADE noisy':>9} {'ADE den':>8} {'red%':>5} | "
          f"{'FDE noisy':>9} {'FDE den':>8} {'red%':>5}")
    for s, (a_n, f_n, a_d, f_d) in out.items():
        ar = 100 * (1 - a_d / a_n)
        fr = 100 * (1 - f_d / f_n)
        print(f"{s:>6} | {a_n:>9.4f} {a_d:>8.4f} {ar:>4.0f}% | "
              f"{f_n:>9.4f} {f_d:>8.4f} {fr:>4.0f}%")


if __name__ == "__main__":
    main()