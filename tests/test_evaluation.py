import torch
from noise_to_vogue.diffusion import Schedule
from noise_to_vogue.evaluate import metrics_at, forward_frames
from noise_to_vogue.model import UNet
from noise_to_vogue.inference import inspect, pil_image


def test_forward_markov_chain_has_correct_depths_and_fixed_seed():
    schedule = Schedule("linear", 8)
    image = torch.zeros(1, 1, 28, 28)
    a = forward_frames(image, schedule, 42, frames=5)
    assert a == forward_frames(image, schedule, 42, frames=5)
    assert a != forward_frames(image, schedule, 7, frames=5)
    assert a[0]["t"] == 0 and a[-1]["t"] == 8
    assert len(a) == 5


def test_inspection_metrics_and_batch_evaluation_agree():
    torch.manual_seed(0)
    model = UNet(8).eval()
    schedule = Schedule("cosine", 8)
    image = torch.zeros(1, 1, 28, 28)
    noise = torch.randn(image.shape, generator=torch.Generator().manual_seed(42))
    result = inspect(model, schedule, image, 4, 42)
    metric = metrics_at(model, schedule, image.repeat(2, 1, 1, 1), noise.repeat(2, 1, 1, 1), 4)
    assert abs(result["noise_mse"] - metric["noise_mse"]) < 1e-5
    assert abs(result["clean_mse"] - metric["clean_mse_clipped"]) < 1e-5
    assert result["clean_mse_raw"] >= result["clean_mse"]


def test_noise_display_uses_identical_fixed_signed_scale():
    zero = torch.zeros(1, 1, 28, 28)
    assert pil_image(zero, noise=True).getpixel((0, 0)) == (18, 24, 38)
    positive = pil_image(zero + 3, noise=True).getpixel((0, 0))
    negative = pil_image(zero - 3, noise=True).getpixel((0, 0))
    assert positive == (249, 136, 91)
    assert negative == (75, 125, 255)
