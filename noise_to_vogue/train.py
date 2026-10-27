"""Reproducible paired schedule training, EMA, atomic checkpoints, exact resume."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from .data import load_data
from .diffusion import Schedule
from .model import UNet


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def state_hash(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--schedule", choices=["linear", "cosine"], required=True)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--timesteps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--data", default="data")
    p.add_argument("--output", default="checkpoints")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        p.error("steps and batch-size must be positive")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4)
    seed_all(args.seed)
    device = torch.device(args.device)
    model = UNet(args.width).to(device)
    initial_hash = state_hash(model)
    ema = deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    schedule = Schedule(args.schedule, args.timesteps, device)
    images, _ = load_data(args.data)
    if args.batch_size > len(images):
        raise ValueError("batch-size exceeds dataset size")
    images = images.to(device)
    order_rng = torch.Generator().manual_seed(args.seed + 1)
    noise_rng = torch.Generator(device=device).manual_seed(args.seed + 2)
    order = torch.randperm(len(images), generator=order_rng)
    cursor = 0
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{args.schedule}.pt"
    history, start, elapsed = [], 0, 0.0
    if args.resume:
        ckpt = torch.load(path, map_location=device, weights_only=True)
        for key in ("schedule", "timesteps", "batch_size", "width", "seed", "lr"):
            if ckpt["config"][key] != vars(args)[key]:
                raise ValueError(f"Resume configuration mismatch: {key}")
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        order_rng.set_state(ckpt["order_rng"].cpu())
        noise_rng.set_state(ckpt["noise_rng"].cpu())
        order, cursor = ckpt["order"].cpu(), ckpt["cursor"]
        history, start, elapsed = ckpt["history"], ckpt["step"], ckpt["elapsed_seconds"]
        initial_hash = ckpt["initial_hash"]
    started, losses = time.perf_counter(), []
    print(json.dumps({"config": vars(args), "parameters": sum(v.numel() for v in model.parameters()), "initial_hash": initial_hash, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"}), flush=True)
    for step in range(start + 1, args.steps + 1):
        if cursor + args.batch_size > len(order):
            order = torch.randperm(len(images), generator=order_rng)
            cursor = 0
        x = images[order[cursor:cursor + args.batch_size].to(device)]
        cursor += args.batch_size
        t = torch.randint(args.timesteps, (len(x),), device=device, generator=noise_rng)
        noise = torch.randn(x.shape, device=device, generator=noise_rng)
        xt = schedule.corrupt(x, t, noise)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.float16):
            loss = F.mse_loss(model(xt, t), noise)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        with torch.no_grad():
            for avg, current in zip(ema.parameters(), model.parameters()):
                avg.lerp_(current, 0.005)
        losses.append(loss.item())
        if step % 100 == 0 or step == args.steps:
            row = {"step": step, "mse": sum(losses) / len(losses), "elapsed_seconds": elapsed + time.perf_counter() - started}
            history.append(row)
            losses.clear()
            print(json.dumps(row), flush=True)
        if step % 1000 == 0 or step == args.steps:
            checkpoint = {"config": vars(args), "step": step, "model": model.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "history": history, "initial_hash": initial_hash, "order_rng": order_rng.get_state(), "noise_rng": noise_rng.get_state(), "order": order, "cursor": cursor, "elapsed_seconds": elapsed + time.perf_counter() - started, "torch_version": str(torch.__version__)}
            temp = path.with_suffix(".tmp")
            torch.save(checkpoint, temp)
            temp.replace(path)
            (output / f"{args.schedule}-history.json").write_text(json.dumps(history, indent=2))
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
