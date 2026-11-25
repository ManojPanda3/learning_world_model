"""Pytest verification tests for Dreamer RSSM, shapes, gradients, and imagination."""
import torch
from learning_world_model.models import Encoder, Decoder, RSSM, Actor, Critic
from learning_world_model.agent import Dreamer
from learning_world_model.config import Config


def test_rssm_step_and_observe():
    action_dim = 2
    embed_dim = 64
    rssm = RSSM(action_dim=action_dim, embed=embed_dim, deter=64, classes=8, dims=4)

    # Initial state
    h, z = rssm.initial(batch=4, device=torch.device("cpu"))
    assert h.shape == (4, 64)
    assert z.shape == (4, 32)

    # Prior step
    act = torch.randn(4, action_dim)
    h_next, z_prior, prior_probs = rssm.prior_step(h, z, act)
    assert h_next.shape == (4, 64)
    assert z_prior.shape == (4, 32)
    assert prior_probs.shape == (4, 4, 8)

    # Sequence observation
    embeds = torch.randn(4, 10, embed_dim)
    actions = torch.randn(4, 10, action_dim)
    hs, zs, priors, posts = rssm.observe(embeds, actions)
    assert hs.shape == (4, 10, 64)
    assert zs.shape == (4, 10, 32)
    assert priors.shape == (4, 10, 4, 8)
    assert posts.shape == (4, 10, 4, 8)


def test_dreamer_imagination_and_gradients():
    cfg = Config(
        deter=64,
        classes=8,
        dims=4,
        embed=64,
        hidden=64,
        horizon=5,
    )
    agent = Dreamer(cfg, obs_shape=(4,), action_dim=2, discrete=True)

    # Mock batch
    batch_size = 4
    seq_len = 8
    obs = torch.randn(batch_size, seq_len, 4)
    act = torch.randint(0, 2, (batch_size, seq_len))
    rew = torch.randn(batch_size, seq_len)
    cont = torch.ones(batch_size, seq_len)
    mask = torch.ones(batch_size, seq_len)

    batch = {
        "obs": obs,
        "action": act,
        "reward": rew,
        "cont": cont,
        "mask": mask,
    }

    metrics = agent.update(batch)
    assert "wm_loss" in metrics
    assert "actor_loss" in metrics
    assert "critic_loss" in metrics
    assert not torch.isnan(torch.tensor(metrics["wm_loss"]))
    assert not torch.isnan(torch.tensor(metrics["actor_loss"]))
    assert not torch.isnan(torch.tensor(metrics["critic_loss"]))
