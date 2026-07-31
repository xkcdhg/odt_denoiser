"""
Training loop for the EDM motion-prior denoiser.

Produces the FROZEN prior. After training, the EMA weights are the artifact you
attach RK4 -> ADSS -> LEAP to at inference. Nothing here is sensor-specific.

Run:
    python -m motion_prior.train --data_dir /path/to/ethucy   # real data
    python -m motion_prior.train --synthetic                  # smoke test
"""
from __future__ import annotations
import argparse
import copy
import math
import os
import torch
from torch.utils.data import DataLoader

from .config import Config
from .corruption import make_training_pair, positions_to_features, corrupt_positions
from .edm import EDMDenoiser, edm_loss
from .dataset import WindowDataset, load_ethucy, synthetic_windows
from .metrics import ade_fde


def calibrate_sigma_data(windows, cfg: Config) -> float:
    """EDM sigma_data MUST match the std of the signal in the model's own
    representation (offsets). Mis-setting it silently breaks preconditioning
    (we hit exactly this: guessed 0.3 vs true 1.85 made the denoiser worse).
    Measure it from the data instead of guessing.
    """
    x = torch.from_numpy(windows).float()
    off = positions_to_features(x, cfg)[..., :cfg.data.n_pos_dims]
    return float(off.std().item())


@torch.no_grad()
def eval_ade_fde(model, windows, cfg, device, n=1024,
                 sigmas=(0.1, 0.2, 0.4, 0.8)):
    """ADE/FDE in metres, for the noisy input vs the denoised output.

    Returns {sigma: (ade_noisy, fde_noisy, ade_denoised, fde_denoised)}.
    Denoised should be LOWER than noisy at every sigma once training works;
    the noisy column is the no-op baseline the prior has to beat.
    """
    model.eval()
    x = torch.from_numpy(windows[:n]).float().to(device)
    npos = cfg.data.n_pos_dims
    gt = positions_to_features(x, cfg)[..., :npos]
    out = {}
    for s in sigmas:
        sig = torch.full((x.shape[0], 1, 1), s, device=device)
        xn = positions_to_features(corrupt_positions(x, sig), cfg)
        pred = model(xn, sig)
        a_n, f_n = ade_fde(xn[..., :npos], gt)
        a_d, f_d = ade_fde(pred, gt)
        out[s] = (a_n, f_n, a_d, f_d)
    model.train()
    return out


def _lr_lambda(step, cfg: Config):
    w = cfg.train.warmup_steps
    if step < w:
        return step / max(1, w)
    prog = (step - w) / max(1, cfg.train.max_steps - w)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))


@torch.no_grad()
def _update_ema(ema, model, decay):
    for pe, pm in zip(ema.parameters(), model.parameters()):
        pe.mul_(decay).add_(pm.detach(), alpha=1 - decay)
    for be, bm in zip(ema.buffers(), model.buffers()):
        be.copy_(bm)


def build(cfg: Config):
    device = cfg.train.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.train.seed)
    model = EDMDenoiser(cfg).to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, betas=cfg.train.betas,
        weight_decay=cfg.train.weight_decay,
    )
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: _lr_lambda(s, cfg))
    return device, model, ema, opt, sched


def train(cfg: Config, windows, out_dir: str = "checkpoints",
          auto_sigma_data: bool = True):
    os.makedirs(out_dir, exist_ok=True)
    if auto_sigma_data:
        cfg.edm.sigma_data = calibrate_sigma_data(windows, cfg)
        print(f"calibrated sigma_data = {cfg.edm.sigma_data:.4f} (from data offset std)")
    device, model, ema, opt, sched = build(cfg)
    ds = WindowDataset(windows)
    dl = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True,
                    drop_last=True, num_workers=0)

    step = 0
    model.train()
    data_iter = iter(dl)
    while step < cfg.train.max_steps:
        try:
            clean_pos = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            clean_pos = next(data_iter)
        clean_pos = clean_pos.to(device)

        x_noisy, x_clean, sigma = make_training_pair(clean_pos, cfg)
        loss = edm_loss(model, x_clean, x_noisy, sigma, cfg)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        opt.step()
        sched.step()
        _update_ema(ema, model, cfg.train.ema_decay)

        if step % cfg.train.log_every == 0:
            lr = sched.get_last_lr()[0]
            print(f"step {step:>7d} | loss {loss.item():.4f} | lr {lr:.2e}")
        if step > 0 and step % cfg.train.ckpt_every == 0:
            m = eval_ade_fde(ema, windows, cfg, device)
            print(f"  [eval@{step}] ADE/FDE metres (noisy -> denoised)")
            for s, (a_n, f_n, a_d, f_d) in m.items():
                print(f"     sigma={s:<4} ADE {a_n:.4f} -> {a_d:.4f} | "
                      f"FDE {f_n:.4f} -> {f_d:.4f}")
            torch.save({"ema": ema.state_dict(), "model": model.state_dict(),
                        "cfg": cfg, "step": step},
                       os.path.join(out_dir, f"ckpt_{step}.pt"))
        step += 1

    torch.save({"ema": ema.state_dict(), "model": model.state_dict(),
                "cfg": cfg, "step": step}, os.path.join(out_dir, "final.pt"))
    print("done. frozen prior (EMA) saved to", os.path.join(out_dir, "final.pt"))
    return ema


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--n_synth", type=int, default=20000)
    ap.add_argument("--out_dir", type=str, default="checkpoints")
    ap.add_argument("--max_steps", type=int, default=None)
    args = ap.parse_args()

    cfg = Config()
    if args.max_steps:
        cfg.train.max_steps = args.max_steps

    if args.synthetic or args.data_dir is None:
        print("using synthetic windows (smoke test)")
        windows = synthetic_windows(args.n_synth, cfg)
    else:
        windows = load_ethucy(args.data_dir, cfg)
        print(f"loaded {len(windows)} windows from {args.data_dir}")

    train(cfg, windows, args.out_dir)


if __name__ == "__main__":
    main()
