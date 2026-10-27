import json
from pathlib import Path
import pytest
import torch
from fastapi.testclient import TestClient
from noise_to_vogue.model import UNet
from noise_to_vogue.server import create_app


@pytest.fixture
def root(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "web").mkdir()
    (tmp_path / "web" / "index.html").write_text("<h1>Noise to Vogue</h1>")
    return tmp_path


def test_health_config_static_and_missing_results(root):
    client = TestClient(create_app(root))
    assert client.get("/").status_code == 200
    health = client.get("/api/health").json()
    assert health["checkpoints"] == {"linear": False, "cosine": False}
    assert not health["results"]
    config = client.get("/api/config").json()
    assert len(config["schedules"]["cosine"]["alpha_bar"]) == 200
    for endpoint in ("/api/generate", "/api/inspect", "/api/results"):
        assert client.get(endpoint).status_code == 503


@pytest.mark.parametrize("query", ["schedule=unknown", "seed=-1", "seed=4294967296", "seed=2.5"])
def test_generation_validates_inputs(root, query):
    assert TestClient(create_app(root)).get("/api/generate?" + query).status_code == 422


@pytest.mark.parametrize("query", ["index=-1", "index=10000", "depth=0", "depth=1001"])
def test_inspector_validates_inputs(root, query):
    assert TestClient(create_app(root)).get("/api/inspect?" + query).status_code == 422


def test_exported_demo_can_run_without_checkpoint(root):
    payload = {"seed": 42, "schedule": "cosine", "frames": [], "cached": True}
    (root / "results" / "generation-cosine-42.json").write_text(json.dumps(payload))
    client = TestClient(create_app(root))
    assert client.get("/api/generate?seed=42").json() == payload
    assert client.get("/api/generate?seed=7").status_code == 503


@pytest.mark.parametrize("weights_dir", ["checkpoints", "models"])
def test_real_inference_endpoint_is_repeatable_and_encodes_images(root, weights_dir):
    folder = root / weights_dir
    folder.mkdir()
    torch.manual_seed(1)
    model = UNet(8)
    torch.save({"config": {"width": 8, "schedule": "cosine", "timesteps": 6}, "ema": model.state_dict(), "step": 0}, folder / "cosine.pt")
    client = TestClient(create_app(root))
    a = client.get("/api/generate?seed=42").json()
    b = client.get("/api/generate?seed=42").json()
    assert a == b
    assert a["frames"][0]["t"] == 6
    assert a["frames"][-1]["t"] == 0
    assert a["frames"][-1]["image"].startswith("data:image/png;base64,")
    assert client.get("/api/inspect?depth=7").status_code == 422


def test_exported_demo_is_invalidated_when_checkpoint_changes(root):
    folder = root / "checkpoints"
    folder.mkdir()
    model = UNet(8)
    torch.save({"config": {"width": 8, "schedule": "cosine", "timesteps": 6}, "ema": model.state_dict(), "step": 1}, folder / "cosine.pt")
    (root / "results" / "generation-cosine-42.json").write_text(json.dumps({"stale": True}))
    (root / "results" / "metrics.json").write_text(json.dumps({"schedules": {"cosine": {"checkpoint_sha256": "outdated"}}}))
    result = TestClient(create_app(root)).get("/api/generate?seed=42").json()
    assert "stale" not in result and result["cached"] is False
    assert result["checkpoint_step"] == 1
