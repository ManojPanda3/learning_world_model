"""World model, actor and critic networks for the lightweight DreamerV3."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from .utils import MLP


class Encoder(nn.Module):
    """Observation -> embedding."""

    def __init__(self, obs_shape: tuple[int, ...], embed: int, hidden: int, pixels: bool):
        super().__init__()
        self.pixels = pixels
        self.obs_shape = obs_shape
        if pixels:
            c, h, w = obs_shape
            assert h == w, "expected square images"
            self.base = h // 16
            self.cnn = nn.Sequential(
                nn.Conv2d(c, 32, 4, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(32, 64, 4, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(64, 64, 4, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(64, 128, 4, stride=2, padding=1),
                nn.SiLU(),
                nn.Flatten(),
            )
            self.fc = nn.Sequential(
                nn.Linear(128 * self.base * self.base, embed),
                nn.LayerNorm(embed),
                nn.SiLU(),
            )
        else:
            self.mlp = MLP(obs_shape[0], hidden, embed)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.pixels:
            lead = obs.shape[:-3]
            x = obs.reshape(-1, *self.obs_shape)
            x = self.fc(self.cnn(x))
            return x.reshape(*lead, -1)
        return self.mlp(obs)


class Decoder(nn.Module):
    """Feature -> observation (symlog space for vectors, [0, 1] for pixels)."""

    def __init__(self, feat: int, obs_shape: tuple[int, ...], hidden: int, pixels: bool):
        super().__init__()
        self.pixels = pixels
        self.obs_shape = obs_shape
        if pixels:
            c, h, w = obs_shape
            self.base = h // 16
            self.fc = nn.Linear(feat, 128 * self.base * self.base)
            self.cnn = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                nn.SiLU(),
                nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
                nn.SiLU(),
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
                nn.SiLU(),
                nn.ConvTranspose2d(32, c, 4, stride=2, padding=1),
                nn.Sigmoid(),
            )
        else:
            self.mlp = MLP(feat, hidden, obs_shape[0])

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if self.pixels:
            lead = feat.shape[:-1]
            x = self.fc(feat).reshape(-1, 128, self.base, self.base)
            x = self.cnn(x)
            return x.reshape(*lead, *self.obs_shape)
        return self.mlp(feat)


class RSSM(nn.Module):
    """Recurrent state-space model with categorical latents.

    State = deterministic GRU hidden `h` + stochastic `z` (factorized categoricals).
    """

    def __init__(
        self,
        action_dim: int,
        embed: int,
        deter: int = 128,
        hidden: int = 128,
        classes: int = 16,
        dims: int = 8,
        unimix: float = 0.01,
    ):
        super().__init__()
        self.deter = deter
        self.classes = classes
        self.dims = dims
        self.z_dim = classes * dims
        self.unimix = unimix
        self.gru = nn.GRUCell(self.z_dim + action_dim, deter)
        self.prior = MLP(deter, hidden, self.z_dim)
        self.post = MLP(deter + embed, hidden, self.z_dim)

    @property
    def feat_dim(self) -> int:
        return self.deter + self.z_dim

    def initial(self, batch: int, device: torch.device, dtype=torch.float32):
        return (
            torch.zeros(batch, self.deter, device=device, dtype=dtype),
            torch.zeros(batch, self.z_dim, device=device, dtype=dtype),
        )

    def _sample(self, logits: torch.Tensor):
        logits = logits.view(-1, self.dims, self.classes)
        probs = logits.softmax(-1)
        if self.unimix:
            probs = probs * (1.0 - self.unimix) + self.unimix / self.classes
        idx = Categorical(probs=probs).sample()
        onehot = F.one_hot(idx, self.classes).to(probs.dtype)
        # Straight-through: gradients flow through the probabilities.
        z = (onehot + probs - probs.detach()).flatten(1)
        return z, probs

    def prior_step(self, h, z, action):
        h = self.gru(torch.cat([z, action], dim=-1), h)
        z, probs = self._sample(self.prior(h))
        return h, z, probs

    def post_step(self, h, embed):
        z, probs = self._sample(self.post(torch.cat([h, embed], dim=-1)))
        return z, probs

    def observe(self, embed: torch.Tensor, action: torch.Tensor):
        """Run the posterior over a sequence.

        `action[t]` is the action taken at time t, so it is used to move from
        state t to state t+1 (time 0 starts from the initial state).
        """
        batch, length = embed.shape[:2]
        h, z = self.initial(batch, embed.device, embed.dtype)
        hs, zs, priors, posts = [], [], [], []
        for t in range(length):
            a = action[:, t - 1] if t > 0 else torch.zeros_like(action[:, 0])
            h, z_prior, prior_probs = self.prior_step(h, z, a)
            z, post_probs = self.post_step(h, embed[:, t])
            hs.append(h)
            zs.append(z)
            priors.append(prior_probs)
            posts.append(post_probs)
        return (
            torch.stack(hs, 1),
            torch.stack(zs, 1),
            torch.stack(priors, 1),
            torch.stack(posts, 1),
        )


class Actor(nn.Module):
    def __init__(self, feat: int, hidden: int, action_dim: int, discrete: bool):
        super().__init__()
        self.discrete = discrete
        self.action_dim = action_dim
        self.trunk = MLP(feat, hidden, hidden)
        if discrete:
            self.logits = nn.Linear(hidden, action_dim)
        else:
            self.mean = nn.Linear(hidden, action_dim)
            self.logstd = nn.Linear(hidden, action_dim)

    def forward(self, feat: torch.Tensor):
        x = self.trunk(feat)
        if self.discrete:
            return Categorical(logits=self.logits(x))
        std = F.softplus(self.logstd(x)) + 0.1
        return Normal(self.mean(x), std)

    def sample(self, dist, deterministic: bool = False):
        """Returns action and log-prob (continuous actions are squashed to [-1, 1])."""
        if self.discrete:
            a = dist.probs.argmax(-1) if deterministic else dist.sample()
            return a, dist.log_prob(a)
        if deterministic:
            a = torch.tanh(dist.mean)
        else:
            a = torch.tanh(dist.rsample())
        logp = dist.log_prob(atanh(a)) - torch.log(1.0 - a.pow(2) + 1e-6)
        return a, logp.sum(-1)

    def log_prob(self, dist, action: torch.Tensor):
        if self.discrete:
            return dist.log_prob(action.long())
        logp = dist.log_prob(atanh(action)) - torch.log(1.0 - action.pow(2) + 1e-6)
        return logp.sum(-1)

    def entropy(self, dist):
        if self.discrete:
            return dist.entropy()
        return dist.entropy().sum(-1)


def atanh(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(-0.999999, 0.999999)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


class Critic(nn.Module):
    def __init__(self, feat: int, hidden: int, bins: int = 255):
        super().__init__()
        self.net = MLP(feat, hidden, bins)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)
