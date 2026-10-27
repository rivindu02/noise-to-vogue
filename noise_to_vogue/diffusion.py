"""DDPM forward marginals and ancestral reverse transitions, in explicit equations."""
import math
import torch


class Schedule:
    def __init__(self, name="linear", steps=200, device="cpu"):
        if steps < 2:
            raise ValueError("steps must be at least 2")
        if name == "linear":
            # Preserve the original 1000-step DDPM noise budget at shorter T.
            beta = torch.linspace(1e-4 * 1000 / steps, 0.02 * 1000 / steps, steps, dtype=torch.float64).clamp(max=0.999)
        elif name == "cosine":
            t = torch.linspace(0, 1, steps + 1, dtype=torch.float64)
            a = torch.cos((t + 0.008) / 1.008 * math.pi / 2).square()
            a = a / a[0]
            beta = (1 - a[1:] / a[:-1]).clamp(1e-8, 0.999)
        else:
            raise ValueError("schedule must be linear or cosine")
        self.name, self.steps = name, steps
        # Compute small posterior coefficients in float64 before casting.
        # This avoids cancellation near the clean end of the chain.
        alpha = 1 - beta
        abar = torch.cumprod(alpha, dim=0)
        prev = torch.cat((torch.ones(1, dtype=torch.float64), abar[:-1]))
        self.beta = beta.float().to(device)
        self.alpha = alpha.float().to(device)
        self.abar = abar.float().to(device)
        self.prev = prev.float().to(device)
        self.variance = (beta * (1 - prev) / (1 - abar)).float().to(device)
        self.coef_clean = (beta * prev.sqrt() / (1 - abar)).float().to(device)
        self.coef_noisy = ((1 - prev) * alpha.sqrt() / (1 - abar)).float().to(device)

    def at(self, values, t):
        return values[t].reshape(-1, 1, 1, 1)

    def corrupt(self, x0, t, noise):
        a = self.at(self.abar, t)
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    def clean(self, xt, t, predicted):
        a = self.at(self.abar, t)
        return (xt - (1 - a).sqrt() * predicted) / a.sqrt()

    def reverse(self, xt, t, predicted, noise, clip=True):
        x0 = self.clean(xt, t, predicted)
        if clip:
            x0 = x0.clamp(-1, 1)
        # Posterior mean q(x_{t-1}|x_t,x_0_hat), using clipped x_0_hat.
        mean = self.at(self.coef_clean, t) * x0 + self.at(self.coef_noisy, t) * xt
        result = mean + self.at(self.variance, t).sqrt() * noise
        return torch.where((t == 0).reshape(-1, 1, 1, 1), x0, result), x0


@torch.inference_mode()
def sample(model, schedule, seed=42, count=1, frames=41):
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((count, 1, 28, 28), device=device, generator=generator)
    saved = []
    depths = set(torch.linspace(1, schedule.steps, min(max(frames - 1, 1), schedule.steps)).long().tolist())
    for i in reversed(range(schedule.steps)):
        t = torch.full((count,), i, device=device, dtype=torch.long)
        predicted = model(x, t)
        next_x, clean = schedule.reverse(x, t, predicted, torch.randn(x.shape, device=device, generator=generator))
        if i + 1 in depths:
            saved.append({"t": i + 1, "x": x.cpu(), "clean": clean.cpu()})
        x = next_x
    saved.append({"t": 0, "x": x.cpu(), "clean": x.cpu()})
    return saved
