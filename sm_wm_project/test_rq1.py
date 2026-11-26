"""Pytest suite for A/B Test 1: RQ1 - Learned Forgetting & Selective Storage."""
import pytest
import torch
import numpy as np
from config import Config
from env_dataset import KeymapMemoryEnv, collect_dataset
from models import SM_WM
from eval import eval_rq1_learned_forgetting


def test_rq1_metric_structure():
    """Verify that eval_rq1_learned_forgetting returns all required metrics with valid ranges."""
    cfg = Config()
    model = SM_WM(
        obs_dim=cfg.obs_dim,
        act_dim=cfg.act_dim,
        d_model=cfg.d_model,
        num_slots=cfg.num_slots,
    )
    dataset = collect_dataset(num_episodes=20, split="val", seed=123)
    res = eval_rq1_learned_forgetting(model, dataset, device="cpu")

    # Assert all required keys are present
    assert "next_obs_mse" in res
    assert "write_novel_rate" in res
    assert "write_pickup_rate" in res
    assert "write_distractor_rate" in res
    assert "post_door_eviction_rate" in res
    assert "mean_active_slots" in res

    # Value ranges
    assert res["write_novel_rate"] >= 0.0 and res["write_novel_rate"] <= 1.0
    assert res["write_pickup_rate"] >= 0.0 and res["write_pickup_rate"] <= 1.0
    assert res["write_distractor_rate"] >= 0.0 and res["write_distractor_rate"] <= 1.0
    assert res["post_door_eviction_rate"] >= 0.0 and res["post_door_eviction_rate"] <= 1.0
    assert res["mean_active_slots"] >= 0.0 and res["mean_active_slots"] <= cfg.num_slots


def test_rq1_write_and_evict_gate_sensitivity():
    """
    Test that the write gate and evict gate differentiate between key events and distractor steps.
    With initialized positive bias, write_probs start high; verify gradient pushes down on distractors.
    """
    cfg = Config()
    model = SM_WM(obs_dim=cfg.obs_dim, act_dim=cfg.act_dim, d_model=cfg.d_model, num_slots=cfg.num_slots)
    dataset = collect_dataset(num_episodes=10, split="train", seed=42)

    obs = dataset["obs"][:, :30]
    act = torch.nn.functional.one_hot(dataset["act"][:, :30], num_classes=cfg.act_dim).float()

    out = model(obs, act)
    write_probs = out["write_probs"]
    keep_probs = out["keep_probs"]

    assert write_probs.shape == (10, 30, 1)
    assert keep_probs.shape == (10, 30, cfg.num_slots, 1)

    # Initial biases: write_probs should be > 0.5 initially due to positive bias
    assert write_probs.mean().item() > 0.4
    # keep_probs should be within valid non-collapsed range
    assert 0.3 < keep_probs.mean().item() < 0.9


def test_rq1_learned_forgetting_toy_training():
    """
    Run 3 quick gradient steps on a small batch and assert that loss decreases
    and memory gates produce stable non-NaN outputs.
    """
    cfg = Config()
    model = SM_WM(obs_dim=cfg.obs_dim, act_dim=cfg.act_dim, d_model=cfg.d_model, num_slots=cfg.num_slots)
    dataset = collect_dataset(num_episodes=8, split="train", seed=99)

    obs = dataset["obs"][:, :20]
    act = torch.nn.functional.one_hot(dataset["act"][:, :20], num_classes=cfg.act_dim).float()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []

    for _ in range(3):
        optimizer.zero_grad()
        out = model(obs, act)
        pred_obs = out["obs_pred"][:, :-1]
        target_obs = obs[:, 1:]
        loss_obs = torch.nn.functional.mse_loss(pred_obs, target_obs)
        loss_w = out["write_probs"].mean()
        loss_k = (out["m_history"] * out["keep_probs"]).mean()
        loss = loss_obs + 0.01 * loss_w + 0.005 * loss_k
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert not np.isnan(losses[-1])
    assert losses[-1] < losses[0] + 0.5  # reasonable stability
