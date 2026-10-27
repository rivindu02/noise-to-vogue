"""Evaluate paired checkpoints and write measured, versioned viewer artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import torch
from .data import load_data, LABELS, FILES
from .diffusion import sample
from .inference import load_checkpoint, grid, image_url, inspect, pil_image


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def serialize_frames(frames):
    return [{"t": f["t"], "image": image_url(f["x"][0]), "clean": image_url(f["clean"][0])} for f in frames]


@torch.inference_mode()
def metrics_at(model, schedule, images, noise, depth, batch_size=128):
    metrics = []
    device = next(model.parameters()).device
    for first in range(0, len(images), batch_size):
        x, eps = images[first:first + batch_size].to(device), noise[first:first + batch_size].to(device)
        t = torch.full((len(x),), depth - 1, device=device, dtype=torch.long)
        noisy = schedule.corrupt(x, t, eps)
        predicted = model(noisy, t)
        clean = schedule.clean(noisy, t, predicted)
        metrics.append(torch.stack(((predicted - eps).square().flatten(1).mean(1), (clean.clamp(-1, 1) - x).square().flatten(1).mean(1), (clean - x).square().flatten(1).mean(1)), dim=1).cpu())
    values = torch.cat(metrics).double()
    return {"depth": depth, "alpha_bar": schedule.abar[depth - 1].item(), "noise_mse": values[:, 0].mean().item(), "noise_mse_se": (values[:, 0].std() / len(values) ** 0.5).item(), "clean_mse_clipped": values[:, 1].mean().item(), "clean_mse_raw": values[:, 2].mean().item(), "samples": len(values)}


def forward_frames(x0, schedule, seed, frames=41):
    generator = torch.Generator().manual_seed(seed)
    x = x0.clone().cpu()
    saved = [{"t": 0, "image": image_url(x)}]
    depths = set(torch.linspace(1, schedule.steps, frames - 1).long().tolist())
    for i in range(schedule.steps):
        eps = torch.randn(x.shape, generator=generator)
        beta = schedule.beta[i].cpu()
        x = (1 - beta).sqrt() * x + beta.sqrt() * eps
        if i + 1 in depths:
            saved.append({"t": i + 1, "image": image_url(x)})
    return saved


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", default="checkpoints")
    p.add_argument("--output", default="results")
    p.add_argument("--data", default="data")
    p.add_argument("--models", default="models", help="Directory for compact inference weights")
    p.add_argument("--samples", type=int, default=1024)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if not 2 <= args.samples <= 10000:
        p.error("samples must be between 2 and 10000")
    torch.set_num_threads(4)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    images, labels = load_data(args.data, train=False)
    # Fixed random test subset, independent of the training seed and order.
    indices = torch.randperm(len(images), generator=torch.Generator().manual_seed(31415))[:args.samples]
    subset = images[indices]
    noise = torch.randn(subset.shape, generator=torch.Generator().manual_seed(27182))
    results = {"schema_version": 1, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "protocol": {"test_indices_seed": 31415, "noise_seed": 27182, "test_samples": args.samples, "test_indices_sha256": hashlib.sha256(indices.numpy().tobytes()).hexdigest(), "dataset_md5": FILES, "replicates": 1, "limitations": "One paired training seed; error bars reflect test-example variation, not training uncertainty. Raw timestep MSE compares different SNRs. Signal-matched comparisons are reported separately. No FID or class coverage claim."}, "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "device": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else "CPU"}, "schedules": {}}
    checkpoints = {}
    for name in ("linear", "cosine"):
        model, schedule, checkpoint = load_checkpoint(Path(args.checkpoints) / f"{name}.pt", args.device)
        checkpoints[name] = checkpoint
        models_dir = Path(args.models)
        models_dir.mkdir(parents=True, exist_ok=True)
        compact = {"config": checkpoint["config"], "ema": {k: v.detach().cpu() for k, v in checkpoint["ema"].items()}, "step": checkpoint["step"], "initial_hash": checkpoint["initial_hash"], "torch_version": str(torch.__version__)}
        torch.save(compact, models_dir / f"{name}.pt")
        depths = sorted(set([1, 5, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, schedule.steps]))
        depths = [t for t in depths if t <= schedule.steps]
        metrics = [metrics_at(model, schedule, subset, noise, t) for t in depths]
        matched = []
        for target in (0.99, 0.9, 0.5, 0.1, 0.01):
            depth = (schedule.abar - target).abs().argmin().item() + 1
            matched.append({"target_alpha_bar": target, **metrics_at(model, schedule, subset, noise, depth)})
        frames = sample(model, schedule, 42, count=32)
        grid(frames[-1]["x"]).save(output / f"{name}-samples.png")
        animation = [pil_image(f["x"][0], scale=8).convert("P") for f in frames]
        animation[0].save(output / f"{name}-denoising.gif", save_all=True, append_images=animation[1:], duration=95, loop=0)
        summary = {"config": checkpoint["config"], "step": checkpoint["step"], "parameters": sum(v.numel() for v in model.parameters()), "initial_hash": checkpoint["initial_hash"], "checkpoint_sha256": sha256(Path(args.checkpoints) / f"{name}.pt"), "inference_weights_sha256": sha256(models_dir / f"{name}.pt"), "elapsed_seconds": checkpoint["elapsed_seconds"], "training": checkpoint["history"], "by_timestep": metrics, "signal_matched": matched, "sample_grid": f"/results/{name}-samples.png", "animation": f"/results/{name}-denoising.gif"}
        results["schedules"][name] = summary
        for seed in (7, 42, 2026):
            single = sample(model, schedule, seed)
            index = 0
            payload = {"seed": seed, "schedule": name, "steps": schedule.steps, "checkpoint_step": checkpoint["step"], "frames": serialize_frames(single), "cached": True}
            (output / f"generation-{name}-{seed}.json").write_text(json.dumps(payload))
            forward = {"seed": seed, "schedule": name, "index": index, "label": LABELS[labels[index].item()], "frames": forward_frames(images[index:index + 1], schedule, seed), "kind": "markov", "steps": schedule.steps, "cached": True}
            (output / f"forward-{name}-{seed}.json").write_text(json.dumps(forward))
        for depth in (1, 20, 50, 100, 150, schedule.steps):
            if depth <= schedule.steps:
                payload = inspect(model, schedule, images[:1], depth, 42)
                payload.update({"index": 0, "label": LABELS[labels[0].item()], "schedule": name, "cached": True})
                (output / f"inspection-{name}-42-0-{depth}.json").write_text(json.dumps(payload))
        print(json.dumps({"schedule": name, "step": checkpoint["step"], "metrics": metrics}), flush=True)
        del model
    left, right = (checkpoints[n] for n in ("linear", "cosine"))
    controlled_keys = ("timesteps", "batch_size", "width", "seed", "lr", "steps")
    results["controls_verified"] = {key: left["config"][key] == right["config"][key] for key in controlled_keys}
    results["controls_verified"]["initial_weights"] = left["initial_hash"] == right["initial_hash"]
    results["controls_verified"]["completed_steps"] = left["step"] == right["step"] == left["config"]["steps"]
    if not all(results["controls_verified"].values()):
        raise ValueError("The checkpoints do not form a controlled pair")
    (output / "metrics.json").write_text(json.dumps(results, indent=2))
    lines = ["# Measured experiment results", "", f"Evaluated {args.samples} held-out examples. One paired training seed; this is an educational comparison, not a statistical ranking.", "", "| Schedule | Updates | Parameters | Training seconds | Noise MSE at alpha_bar ≈ 0.5 |", "|---|---:|---:|---:|---:|"]
    for name, result in results["schedules"].items():
        half = result["signal_matched"][2]
        lines.append(f"| {name} | {result['step']} | {result['parameters']} | {result['elapsed_seconds']:.1f} | {half['noise_mse']:.6f} ± {half['noise_mse_se']:.6f} SE |")
    lines += ["", "All paired controls verified. Full per-timestep measurements, signal-matched comparisons, dataset checksums, initial-weight hashes, checkpoint hashes and environment information are in [metrics.json](metrics.json).", "", "Actual and predicted noise share a fixed signed color scale of −3 to +3 standard deviations. Clean-image display clips to [−1, 1]; both unclipped and clipped reconstruction MSE are recorded.", "", "Sample grids and GIFs are generated by the trained EMA models. Test labels are used only for viewer captions; training is unconditional."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
