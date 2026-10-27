"""Exact resume comparison on a tiny synthetic dataset, not a quality claim."""
import sys
import torch
from noise_to_vogue import train


def test_checkpoint_resume_matches_uninterrupted_training(tmp_path, monkeypatch):
    generator = torch.Generator().manual_seed(123)
    images = torch.rand((16, 1, 28, 28), generator=generator) * 2 - 1
    monkeypatch.setattr(train, "load_data", lambda *args: (images.clone(), torch.zeros(16, dtype=torch.long)))

    def run(output, steps, resume=False):
        argv = ["train", "--schedule", "linear", "--steps", str(steps), "--timesteps", "8", "--batch-size", "4", "--width", "8", "--device", "cpu", "--output", str(output)]
        if resume:
            argv.append("--resume")
        monkeypatch.setattr(sys, "argv", argv)
        train.main()
        return torch.load(output / "linear.pt", weights_only=True)

    direct = run(tmp_path / "direct", 4)
    run(tmp_path / "resumed", 2)
    resumed = run(tmp_path / "resumed", 4, True)
    assert direct["initial_hash"] == resumed["initial_hash"]
    for key in direct["model"]:
        assert torch.equal(direct["model"][key], resumed["model"][key]), key
        assert torch.equal(direct["ema"][key], resumed["ema"][key]), key
    assert torch.equal(direct["noise_rng"], resumed["noise_rng"])
    assert direct["cursor"] == resumed["cursor"]
