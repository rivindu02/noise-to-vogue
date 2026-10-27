"""FastAPI viewer for live models or committed, genuine model export artifacts."""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Literal
from urllib.error import URLError
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from .data import load_data, LABELS
from .diffusion import Schedule, sample
from .evaluate import forward_frames, serialize_frames
from .inference import load_checkpoint, inspect

ScheduleName = Literal["linear", "cosine"]


def create_app(root=None, device=None):
    root = Path(root or os.environ.get("NTV_ROOT", Path(__file__).resolve().parent.parent))
    checkpoints = Path(os.environ.get("NTV_CHECKPOINTS", root / "checkpoints"))
    device = device or os.environ.get("NTV_DEVICE", "cpu")
    torch.set_num_threads(int(os.environ.get("NTV_THREADS", "4")))
    app = FastAPI(title="Noise to Vogue", version="1.0.0")
    inference_lock = threading.Lock()

    def checkpoint_path(name):
        full = checkpoints / f"{name}.pt"
        return full if full.is_file() else root / "models" / f"{name}.pt"

    @lru_cache(maxsize=2)
    def model_for(name, modified):
        return load_checkpoint(checkpoint_path(name), device)

    def get_model(name):
        path = checkpoint_path(name)
        if not path.is_file():
            raise HTTPException(503, "No trained checkpoint available. Train the pair or use exported demo seeds (7, 42, 2026).")
        return model_for(name, path.stat().st_mtime_ns)

    @lru_cache(maxsize=1)
    def test_data():
        try:
            return load_data(root / "data", train=False)
        except (OSError, URLError, ValueError) as e:
            raise HTTPException(503, "Test data unavailable. Run python -m noise_to_vogue.prepare first.") from e

    def artifact(name):
        path = root / "results" / name
        if path.is_file():
            return json.loads(path.read_text())
        return None

    @lru_cache(maxsize=4)
    def checkpoint_hash(path, modified, size):
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def model_artifact(filename, name):
        value = artifact(filename)
        if value is None:
            return None
        path = checkpoint_path(name)
        if not path.is_file():
            return value
        metrics = artifact("metrics.json")
        if not metrics or name not in metrics["schedules"]:
            return None
        key = "checkpoint_sha256" if path.parent == checkpoints else "inference_weights_sha256"
        expected = metrics["schedules"][name].get(key)
        info = path.stat()
        if expected and checkpoint_hash(str(path), info.st_mtime_ns, info.st_size) == expected:
            return value
        return None

    def steps_for(name):
        path = checkpoint_path(name)
        if path.exists():
            return get_model(name)[1].steps
        metrics = artifact("metrics.json")
        if metrics and name in metrics["schedules"]:
            return metrics["schedules"][name]["config"]["timesteps"]
        return 200

    @app.get("/api/health")
    def health():
        return {"status": "ok", "device": device, "checkpoints": {n: checkpoint_path(n).is_file() for n in ("linear", "cosine")}, "results": (root / "results" / "metrics.json").is_file(), "demo_seeds": [7, 42, 2026]}

    @app.get("/api/config")
    def config():
        curves = {}
        for name in ("linear", "cosine"):
            schedule = Schedule(name, steps_for(name))
            curves[name] = {"steps": schedule.steps, "beta": schedule.beta.tolist(), "alpha_bar": schedule.abar.tolist(), "log_snr": torch.log(schedule.abar / (1 - schedule.abar)).tolist()}
        return {"schedules": curves, "labels": LABELS, "normalization": "[-1, 1]", "noise_display_range": [-3, 3]}

    @app.get("/api/results")
    def results():
        result = artifact("metrics.json")
        if result is None:
            raise HTTPException(503, "No measured experiment yet. Run python -m noise_to_vogue.evaluate after training both schedules.")
        return result

    @app.get("/api/generate")
    def generate(schedule: ScheduleName = "cosine", seed: int = Query(42, ge=0, le=4294967295)):
        cached = model_artifact(f"generation-{schedule}-{seed}.json", schedule)
        if cached is not None:
            return cached
        with inference_lock:
            model, diffusion, checkpoint = get_model(schedule)
            frames = sample(model, diffusion, seed)
        return {"seed": seed, "schedule": schedule, "steps": diffusion.steps, "checkpoint_step": checkpoint["step"], "frames": serialize_frames(frames), "cached": False}

    @app.get("/api/forward")
    def forward(schedule: ScheduleName = "cosine", seed: int = Query(42, ge=0, le=4294967295), index: int = Query(0, ge=0, le=9999)):
        cached = model_artifact(f"forward-{schedule}-{seed}.json", schedule)
        if index == 0 and cached is not None:
            return cached
        images, labels = test_data()
        diffusion = Schedule(schedule, steps_for(schedule))
        return {"seed": seed, "schedule": schedule, "index": index, "label": LABELS[labels[index].item()], "frames": forward_frames(images[index:index + 1], diffusion, seed), "kind": "markov", "steps": diffusion.steps, "cached": False}

    @app.get("/api/inspect")
    def inspection(schedule: ScheduleName = "cosine", seed: int = Query(42, ge=0, le=4294967295), index: int = Query(0, ge=0, le=9999), depth: int = Query(100, ge=1, le=1000)):
        cached = model_artifact(f"inspection-{schedule}-{seed}-{index}-{depth}.json", schedule)
        if cached is not None:
            return cached
        with inference_lock:
            model, diffusion, _ = get_model(schedule)
            if depth > diffusion.steps:
                raise HTTPException(422, f"depth must be <= {diffusion.steps}")
            images, labels = test_data()
            result = inspect(model, diffusion, images[index:index + 1], depth, seed)
        return {**result, "schedule": schedule, "index": index, "label": LABELS[labels[index].item()], "cached": False}

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(root / "web" / "index.html")

    if (root / "web").is_dir():
        app.mount("/web", StaticFiles(directory=root / "web"), name="web")
    if (root / "results").is_dir():
        app.mount("/results", StaticFiles(directory=root / "results"), name="results")
    return app


app = create_app()
