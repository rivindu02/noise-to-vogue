"""Shared checkpoint loading, rendering and held-out inspections."""
import base64
from io import BytesIO
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from .diffusion import Schedule
from .model import UNet


def load_checkpoint(path, device="cpu"):
    # Only load trusted checkpoints produced by this project's trainer.
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = checkpoint["config"]
    model = UNet(config["width"]).to(device).eval()
    model.load_state_dict(checkpoint["ema"])
    return model, Schedule(config["schedule"], config["timesteps"], device), checkpoint


def pil_image(tensor, noise=False, scale=1):
    a = tensor.detach().cpu().float().squeeze().numpy()
    if noise:
        # Fixed +/-3 standard deviations, shared by actual/predicted noise.
        a = (a / 3).clip(-1, 1)
        rgb = np.empty((*a.shape, 3), dtype=np.float32)
        negative = np.array([75, 125, 255], dtype=np.float32)
        positive = np.array([249, 136, 91], dtype=np.float32)
        neutral = np.array([18, 24, 38], dtype=np.float32)
        for c in range(3):
            rgb[..., c] = neutral[c] + np.abs(a) * (np.where(a < 0, negative[c], positive[c]) - neutral[c])
        image = Image.fromarray(rgb.astype(np.uint8))
    else:
        image = Image.fromarray(((a.clip(-1, 1) + 1) * 127.5).round().astype(np.uint8))
    if scale != 1:
        image = image.resize((28 * scale, 28 * scale), Image.Resampling.NEAREST)
    return image


def image_url(tensor, noise=False):
    stream = BytesIO()
    pil_image(tensor, noise).save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


def grid(images, columns=8, scale=4):
    rows = (len(images) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * (28 * scale + 8) + 8, rows * (28 * scale + 8) + 8), "#101724")
    for i, image in enumerate(images):
        canvas.paste(pil_image(image, scale=scale), (8 + i % columns * (28 * scale + 8), 8 + i // columns * (28 * scale + 8)))
    return canvas


@torch.inference_mode()
def inspect(model, schedule, x0, depth, seed):
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    x0 = x0.to(device)
    epsilon = torch.randn(x0.shape, generator=generator, device=device)
    t = torch.tensor([depth - 1], device=device)
    xt = schedule.corrupt(x0, t, epsilon)
    predicted = model(xt, t)
    clean = schedule.clean(xt, t, predicted)
    return {
        "original": image_url(x0), "noisy": image_url(xt),
        "actual_noise": image_url(epsilon, True), "predicted_noise": image_url(predicted, True),
        "estimated_clean": image_url(clean),
        "noise_mse": (epsilon - predicted).square().mean().item(),
        "clean_mse": (x0 - clean.clamp(-1, 1)).square().mean().item(),
        "clean_mse_raw": (x0 - clean).square().mean().item(),
        "beta": schedule.beta[depth - 1].item(), "alpha_bar": schedule.abar[depth - 1].item(),
        "depth": depth, "seed": seed,
    }
