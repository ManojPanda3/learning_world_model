"""Small shared utilities: symlog, two-hot distributions, MLPs, return norm."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric log: compresses large magnitudes, keeps sign and zero."""
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(x.abs())


class TwoHot(nn.Module):
    """Two-hot encoding of scalar targets over a fixed set of bins.

    Used for reward and value prediction. Targets are expected in symlog space.
    """

    def __init__(self, lo: float = -20.0, hi: float = 20.0, bins: int = 255):
        super().__init__()
        self.register_buffer("bins", torch.linspace(lo, hi, bins))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        bins = self.bins
        x = x.clamp(bins[0], bins[-1])
        idx = torch.searchsorted(bins, x.contiguous(), right=True) - 1
        idx = idx.clamp(0, bins.numel() - 2)
        lo, hi = bins[idx], bins[idx + 1]
        w = ((x - lo) / (hi - lo)).clamp(0.0, 1.0)
        target = torch.zeros(*x.shape, bins.numel(), device=x.device, dtype=x.dtype)
        target.scatter_(-1, idx.unsqueeze(-1), (1.0 - w).unsqueeze(-1))
        target.scatter_(-1, (idx + 1).unsqueeze(-1), w.unsqueeze(-1))
        return target

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        return (logits.softmax(-1) * self.bins).sum(-1)

    def loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Cross entropy against a two-hot encoded symlog target."""
        return -(self.encode(symlog(target)) * logits.log_softmax(-1)).sum(-1)


class MLP(nn.Module):
    """LayerNorm + SiLU MLP, the default building block in DreamerV3."""

    def __init__(
        self,
        inp: int,
        hidden: int,
        out: int,
        layers: int = 2,
        norm: bool = True,
        act: type[nn.Module] = nn.SiLU,
    ):
        super().__init__()
        mods: list[nn.Module] = []
        for i in range(layers):
            mods.append(nn.Linear(inp if i == 0 else hidden, hidden))
            if norm:
                mods.append(nn.LayerNorm(hidden))
            mods.append(act())
        mods.append(nn.Linear(hidden, out))
        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReturnNorm:
    """Running 5-95 percentile scale for returns / advantages."""

    def __init__(self, decay: float = 0.99):
        self.decay = decay
        self.scale = 1.0

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() > 1:
            flat = x.detach().float().flatten()
            lo = torch.quantile(flat, 0.05)
            hi = torch.quantile(flat, 0.95)
            scale = max(1.0, (hi - lo).item())
            self.scale = self.decay * self.scale + (1.0 - self.decay) * scale
        return x / self.scale


def kl_categorical(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """KL(p || q) for factorized categoricals.

    Shapes (..., classes, dims); returns (...), summed over classes and dims.
    """
    p = p.clamp_min(1e-8)
    q = q.clamp_min(1e-8)
    return (p * (p.log() - q.log())).sum(dim=(-1, -2))


def grad_clip(parameters, max_norm: float) -> None:
    nn.utils.clip_grad_norm_(parameters, max_norm)
