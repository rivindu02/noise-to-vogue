"""Small timestep-conditioned U-Net; no class labels enter the network."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):
        freq = torch.exp(torch.arange(self.dim // 2, device=t.device) * (-math.log(10000) / (self.dim // 2 - 1)))
        angles = t.float()[:, None] * freq[None]
        return self.net(torch.cat((angles.sin(), angles.cos()), dim=-1))


class Block(nn.Module):
    def __init__(self, cin, cout, time_dim=128):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.time = nn.Linear(time_dim, cout)
        self.norm2 = nn.GroupNorm(8, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, t):
        h = self.conv1(F.silu(self.norm1(x))) + self.time(F.silu(t))[:, :, None, None]
        return self.skip(x) + self.conv2(F.silu(self.norm2(h)))


class UNet(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.width = width
        self.time = TimeEmbedding()
        self.input = nn.Conv2d(1, width, 3, padding=1)
        self.enc1 = Block(width, width)
        self.down1 = nn.Conv2d(width, width * 2, 4, stride=2, padding=1)
        self.enc2 = Block(width * 2, width * 2)
        self.down2 = nn.Conv2d(width * 2, width * 4, 4, stride=2, padding=1)
        self.mid1 = Block(width * 4, width * 4)
        self.mid2 = Block(width * 4, width * 4)
        self.dec2 = Block(width * 6, width * 2)
        self.dec1 = Block(width * 3, width)
        self.output = nn.Sequential(nn.GroupNorm(8, width), nn.SiLU(), nn.Conv2d(width, 1, 3, padding=1))

    def forward(self, x, t):
        t = self.time(t)
        h1 = self.enc1(self.input(x), t)
        h2 = self.enc2(self.down1(h1), t)
        h = self.mid2(self.mid1(self.down2(h2), t), t)
        h = self.dec2(torch.cat((F.interpolate(h, size=h2.shape[-2:], mode="nearest"), h2), dim=1), t)
        h = self.dec1(torch.cat((F.interpolate(h, size=h1.shape[-2:], mode="nearest"), h1), dim=1), t)
        return self.output(h)
