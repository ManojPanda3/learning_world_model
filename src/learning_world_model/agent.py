"""DreamerV3 agent: world-model learning, latent imagination, actor-critic."""

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from .config import Config
from .models import Actor, Critic, Decoder, Encoder, RSSM
from .utils import MLP, ReturnNorm, TwoHot, grad_clip, kl_categorical, symexp


class Dreamer:
    def __init__(self, cfg: Config, obs_shape: tuple[int, ...], action_dim: int, discrete: bool):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.discrete = discrete
        self.action_dim = action_dim

        self.encoder = Encoder(obs_shape, cfg.embed, cfg.hidden, cfg.pixels).to(self.device)
        self.rssm = RSSM(
            action_dim,
            cfg.embed,
            deter=cfg.deter,
            hidden=cfg.hidden,
            classes=cfg.classes,
            dims=cfg.dims,
            unimix=cfg.unimix,
        ).to(self.device)
        feat = self.rssm.feat_dim
        self.decoder = Decoder(feat, obs_shape, cfg.hidden, cfg.pixels).to(self.device)
        self.reward_head = MLP(feat, cfg.hidden, 255).to(self.device)
        self.cont_head = MLP(feat, cfg.hidden, 1).to(self.device)
        self.actor = Actor(feat, cfg.hidden, action_dim, discrete).to(self.device)
        self.critic = Critic(feat, cfg.hidden, 255).to(self.device)
        self.slow_critic = copy.deepcopy(self.critic).to(self.device)
        for p in self.slow_critic.parameters():
            p.requires_grad_(False)

        self.twohot = TwoHot().to(self.device)
        self.retnorm = ReturnNorm()

        wm_params = (
            list(self.encoder.parameters())
            + list(self.rssm.parameters())
            + list(self.decoder.parameters())
            + list(self.reward_head.parameters())
            + list(self.cont_head.parameters())
        )
        self.wm_opt = Adam(wm_params, lr=cfg.model_lr)
        self.actor_opt = Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = Adam(self.critic.parameters(), lr=cfg.critic_lr)

    # ------------------------------------------------------------------ acting
    def _obs_tensor(self, obs) -> torch.Tensor:
        x = torch.as_tensor(np.asarray(obs), device=self.device)
        if self.cfg.pixels:
            return x.float() / 255.0
        return x.float()

    def _encode_action(self, action) -> torch.Tensor:
        a = torch.as_tensor(action, device=self.device)
        if self.discrete:
            if a.dim() == 0:
                a = a.unsqueeze(0)
            return F.one_hot(a.long(), self.action_dim).float()
        if a.dim() == 1:
            a = a.unsqueeze(0)
        return a.float()

    def init_state(self, obs):
        h, z = self.rssm.initial(1, self.device)
        if self.discrete:
            zero = torch.zeros(1, dtype=torch.long, device=self.device)
        else:
            zero = torch.zeros(self.action_dim, device=self.device)
        return self.observe_state(obs, (h, z), zero)

    @torch.no_grad()
    def observe_state(self, obs, state, action):
        h, z = state
        h, _, _ = self.rssm.prior_step(h, z, self._encode_action(action))
        embed = self.encoder(self._obs_tensor(obs).unsqueeze(0))
        z, _ = self.rssm.post_step(h, embed)
        return h, z

    @torch.no_grad()
    def act(self, state, deterministic: bool = False):
        h, z = state
        dist = self.actor(torch.cat([h, z], dim=-1))
        action, _ = self.actor.sample(dist, deterministic=deterministic)
        if self.discrete:
            return int(action[0])
        return action[0].cpu().numpy()

    # ------------------------------------------------------------- world model
    def world_model_loss(self, batch: dict[str, torch.Tensor]):
        obs = batch["obs"]
        mask = batch["mask"]
        denom = mask.sum().clamp_min(1.0)

        embed = self.encoder(obs)
        action = self._encode_action(batch["action"])
        h, z, prior, post = self.rssm.observe(embed, action)
        feat = torch.cat([h, z], dim=-1)

        rec = self.decoder(feat)
        rec_loss = F.mse_loss(rec, obs, reduction="none")
        rec_loss = rec_loss.flatten(2).mean(-1)

        reward_logits = self.reward_head(feat)
        rew_loss = self.twohot.loss(reward_logits, batch["reward"])
        cont_logits = self.cont_head(feat).squeeze(-1)
        cont_loss = F.binary_cross_entropy_with_logits(
            cont_logits, batch["cont"], reduction="none"
        )

        # KL balancing: train the prior toward the posterior and vice versa.
        dyn = kl_categorical(post.detach(), prior)
        rep = kl_categorical(post, prior.detach())
        kl = (self.cfg.dyn_scale * dyn + self.cfg.rep_scale * rep).clamp_min(self.cfg.kl_free)

        rec_loss = (rec_loss * mask).sum() / denom
        rew_loss = (rew_loss * mask).sum() / denom
        cont_loss = (cont_loss * mask).sum() / denom
        kl_loss = (kl * mask).sum() / denom

        loss = (
            self.cfg.recon_scale * rec_loss
            + self.cfg.reward_scale * rew_loss
            + self.cfg.cont_scale * cont_loss
            + kl_loss
        )

        self.wm_opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_clip(
            list(self.encoder.parameters())
            + list(self.rssm.parameters())
            + list(self.decoder.parameters())
            + list(self.reward_head.parameters())
            + list(self.cont_head.parameters()),
            self.cfg.grad_clip,
        )
        self.wm_opt.step()

        metrics = {
            "wm_loss": loss.item(),
            "rec": rec_loss.item(),
            "reward": rew_loss.item(),
            "cont": cont_loss.item(),
            "kl": kl_loss.item(),
        }
        return h.detach(), z.detach(), metrics

    # --------------------------------------------------------------- imagination
    @torch.no_grad()
    def _imagine(self, h, z):
        cfg = self.cfg
        feats, after, actions, logps, ents = [], [], [], [], []
        for _ in range(cfg.horizon):
            feat = torch.cat([h, z], dim=-1)
            dist = self.actor(feat)
            action, logp = self.actor.sample(dist)
            feats.append(feat)
            actions.append(action)
            logps.append(logp)
            ents.append(self.actor.entropy(dist))
            h, z, _ = self.rssm.prior_step(h, z, self._encode_action(action))
            after.append(torch.cat([h, z], dim=-1))

        final = torch.cat([h, z], dim=-1)
        feats = torch.stack(feats, 1)  # (B, H, F)
        after = torch.stack(after, 1)  # (B, H, F)
        all_feats = torch.cat([feats, final.unsqueeze(1)], 1)  # (B, H+1, F)
        actions = torch.stack(actions, 1)
        logps = torch.stack(logps, 1)
        ents = torch.stack(ents, 1)

        rewards = symexp(self.twohot.decode(self.reward_head(after)))
        conts = torch.sigmoid(self.cont_head(after).squeeze(-1))
        slow_values = symexp(self.twohot.decode(self.slow_critic(all_feats)))

        rets = []
        last = slow_values[:, -1]
        for t in reversed(range(cfg.horizon)):
            disc = cfg.gamma * conts[:, t]
            last = rewards[:, t] + disc * ((1.0 - cfg.lam) * slow_values[:, t + 1] + cfg.lam * last)
            rets.append(last)
        rets = torch.stack(rets[::-1], 1)
        return feats, all_feats, actions, logps, ents, rewards, conts, slow_values, rets

    def rl_update(self, h, z):
        cfg = self.cfg
        feats, all_feats, actions, _, _, rewards, conts, slow_values, rets = self._imagine(h, z)

        # Actor: REINFORCE with percentile-normalized advantages.
        dist = self.actor(feats)
        logp = self.actor.log_prob(dist, actions)
        ent = self.actor.entropy(dist)
        adv = rets - slow_values[:, :-1]
        adv = self.retnorm(adv.detach())
        actor_loss = -(logp * adv).mean() - cfg.entropy_scale * ent.mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        grad_clip(self.actor.parameters(), cfg.grad_clip)
        self.actor_opt.step()

        # Critic: two-hot regression to lambda-returns from the slow critic.
        logits = self.critic(all_feats[:, :-1])
        critic_loss = self.twohot.loss(logits, rets.detach()).mean()

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        grad_clip(self.critic.parameters(), cfg.grad_clip)
        self.critic_opt.step()

        with torch.no_grad():
            for slow, fast in zip(self.slow_critic.parameters(), self.critic.parameters()):
                slow.mul_(1.0 - cfg.slow_tau).add_(fast, alpha=cfg.slow_tau)

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": ent.mean().item(),
            "ret": rets.mean().item(),
            "reward_pred": rewards.mean().item(),
            "cont_pred": conts.mean().item(),
        }

    def update(self, batch: dict[str, torch.Tensor]):
        h, z, metrics = self.world_model_loss(batch)
        mask = batch["mask"]
        lengths = mask.sum(1).long().clamp_min(1)
        idx = (lengths - 1).clamp(0, h.shape[1] - 1)
        rows = torch.arange(h.shape[0], device=self.device)
        metrics.update(self.rl_update(h[rows, idx], z[rows, idx]))
        return metrics

    # ---------------------------------------------------------------- checkpoint
    def save(self, path: str, step: int) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "step": step,
                "encoder": self.encoder.state_dict(),
                "rssm": self.rssm.state_dict(),
                "decoder": self.decoder.state_dict(),
                "reward_head": self.reward_head.state_dict(),
                "cont_head": self.cont_head.state_dict(),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "slow_critic": self.slow_critic.state_dict(),
                "wm_opt": self.wm_opt.state_dict(),
                "actor_opt": self.actor_opt.state_dict(),
                "critic_opt": self.critic_opt.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.rssm.load_state_dict(ckpt["rssm"])
        self.decoder.load_state_dict(ckpt["decoder"])
        self.reward_head.load_state_dict(ckpt["reward_head"])
        self.cont_head.load_state_dict(ckpt["cont_head"])
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.slow_critic.load_state_dict(ckpt["slow_critic"])
        self.wm_opt.load_state_dict(ckpt["wm_opt"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])
        return int(ckpt.get("step", 0))
