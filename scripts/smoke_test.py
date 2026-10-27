"""Exercise the trained model, exported demos, inspector and results API."""
import json
import os
from pathlib import Path
from fastapi.testclient import TestClient
from noise_to_vogue.server import app, create_app


def main():
    outcomes = []
    with TestClient(app) as client:
        for path in ["/", "/web/app.js", "/web/style.css", "/api/health", "/api/config", "/api/results", "/api/generate?schedule=linear&seed=42", "/api/generate?schedule=cosine&seed=42", "/api/generate?schedule=cosine&seed=123", "/api/forward?schedule=cosine&seed=42&index=0", "/api/inspect?schedule=cosine&seed=42&index=0&depth=100", "/api/inspect?schedule=linear&seed=123&index=17&depth=50"]:
            response = client.get(path)
            assert response.status_code == 200, (path, response.text)
            outcomes.append({"path": path, "status": response.status_code})
        a = client.get("/api/generate?schedule=cosine&seed=123").json()
        b = client.get("/api/generate?schedule=cosine&seed=123").json()
        assert a == b, "Fixed-seed live generation is not reproducible"
        metrics = client.get("/api/results").json()
        assert all(metrics["controls_verified"].values()), "Paired controls differ"
        for name in ("linear", "cosine"):
            assert metrics["schedules"][name]["step"] > 0
            assert client.get(metrics["schedules"][name]["sample_grid"]).status_code == 200
    # Exercise the distributable compact weights even when full trainer files exist.
    previous = os.environ.get("NTV_CHECKPOINTS")
    os.environ["NTV_CHECKPOINTS"] = str(Path("models") / "no-full-checkpoints")
    try:
        with TestClient(create_app()) as client:
            assert all(client.get("/api/health").json()["checkpoints"].values())
            assert client.get("/api/generate?schedule=cosine&seed=123").json() == a
            response = client.get("/api/generate?schedule=linear&seed=123")
            assert response.status_code == 200 and response.json()["cached"] is False
    finally:
        if previous is None:
            os.environ.pop("NTV_CHECKPOINTS", None)
        else:
            os.environ["NTV_CHECKPOINTS"] = previous
    result = {"status": "passed", "checks": outcomes, "live_seed_reproducibility": True, "paired_controls": True, "compact_weights_fallback": True}
    Path("results/smoke-test.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
