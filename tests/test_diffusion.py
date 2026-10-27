import pytest
import torch
from noise_to_vogue.diffusion import Schedule, sample
from noise_to_vogue.model import UNet


@pytest.mark.parametrize("name", ["linear", "cosine"])
def test_schedule_is_a_valid_near_gaussian_terminal_process(name):
    s = Schedule(name, 200)
    assert ((s.beta > 0) & (s.beta < 1)).all()
    assert (s.abar[1:] < s.abar[:-1]).all()
    assert s.abar[-1] < 1e-3
    assert s.variance[0] == 0
    assert (s.variance >= 0).all()


@pytest.mark.parametrize("name", ["linear", "cosine"])
def test_oracle_noise_exactly_recovers_clean_images(name):
    torch.manual_seed(3)
    s = Schedule(name)
    x = torch.rand(4, 1, 28, 28) * 2 - 1
    epsilon = torch.randn_like(x)
    t = torch.tensor([0, 25, 70, 150])
    xt = s.corrupt(x, t, epsilon)
    assert torch.allclose(s.clean(xt, t, epsilon), x, atol=2e-5)


def test_final_reverse_step_has_no_added_randomness():
    s = Schedule()
    x0 = torch.rand(2, 1, 28, 28) * 2 - 1
    eps = torch.randn_like(x0)
    t = torch.zeros(2, dtype=torch.long)
    xt = s.corrupt(x0, t, eps)
    a, _ = s.reverse(xt, t, eps, torch.randn_like(x0))
    b, _ = s.reverse(xt, t, eps, torch.randn_like(x0) * 100)
    assert torch.equal(a, b)
    assert torch.allclose(a, x0, atol=2e-5)


def test_reverse_mean_matches_ddpm_epsilon_parameterization_without_clipping():
    s = Schedule()
    xt, pred = torch.randn(3, 1, 28, 28), torch.randn(3, 1, 28, 28)
    t = torch.tensor([1, 50, 100])
    actual, _ = s.reverse(xt, t, pred, torch.zeros_like(xt), clip=False)
    expected = (xt - s.at(s.beta, t) / (1 - s.at(s.abar, t)).sqrt() * pred) / s.at(s.alpha, t).sqrt()
    assert torch.allclose(actual, expected, atol=1e-5)


def test_sampling_repeats_for_a_fixed_seed_and_changes_for_another_seed():
    torch.set_num_threads(1)
    torch.manual_seed(2)
    model = UNet(8).eval()
    s = Schedule("cosine", 6)
    a, b, c = (sample(model, s, seed, frames=5) for seed in [42, 42, 7])
    assert [v["t"] for v in a] == sorted([v["t"] for v in a], reverse=True)
    assert a[0]["t"] == 6 and a[-1]["t"] == 0
    assert len(a) == 5
    for left, right in zip(a, b):
        assert torch.equal(left["x"], right["x"])
    assert not torch.equal(a[-1]["x"], c[-1]["x"])
    assert a[-1]["x"].abs().max() <= 1


def test_network_shape_gradients_and_timestep_dependence():
    torch.set_num_threads(1)
    model = UNet(8)
    x = torch.randn(2, 1, 28, 28)
    a = model(x, torch.tensor([0, 0]))
    b = model(x, torch.tensor([199, 199]))
    assert a.shape == x.shape and torch.isfinite(a).all()
    assert not torch.allclose(a, b)
    a.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_invalid_schedule_is_rejected():
    with pytest.raises(ValueError):
        Schedule("invalid")
    with pytest.raises(ValueError):
        Schedule(steps=1)
