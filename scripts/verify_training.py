"""Verify applied optimizer updates and paired RNG progress from full checkpoints."""
import json
from pathlib import Path
import torch


def main():
    root = Path(__file__).resolve().parent.parent
    pair = {
        name: torch.load(root / "checkpoints" / f"{name}.pt", map_location="cpu", weights_only=True)
        for name in ("linear", "cosine")
    }
    left, right = pair.values()
    checks = {
        "same_initial_weights": left["initial_hash"] == right["initial_hash"],
        "same_data_order_rng": torch.equal(left["order_rng"], right["order_rng"]),
        "same_noise_rng": torch.equal(left["noise_rng"], right["noise_rng"]),
        "same_data_order": torch.equal(left["order"], right["order"]),
        "same_data_cursor": left["cursor"] == right["cursor"],
    }
    updates = {}
    for name, checkpoint in pair.items():
        counts = sorted({int(state["step"].item()) for state in checkpoint["optimizer"]["state"].values()})
        checks[f"{name}_all_optimizer_updates_applied"] = counts == [checkpoint["step"]] and checkpoint["step"] == checkpoint["config"]["steps"]
        updates[name] = {"training_steps": checkpoint["step"], "optimizer_step_counts": counts, "elapsed_seconds": checkpoint["elapsed_seconds"], "scaler": checkpoint["scaler"]}
    result = {"status": "passed" if all(checks.values()) else "failed", "checks": checks, "runs": updates}
    (root / "results" / "training-verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not all(checks.values()):
        raise RuntimeError("Training progress verification failed")


if __name__ == "__main__":
    main()
