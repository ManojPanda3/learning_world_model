"""Unit, Law, and Smoke Tests for Suite A."""
import torch
import pytest
from gymnasium.utils.env_checker import check_env

from suite_a.config import SuiteAConfig
from suite_a.envs import TASK_SPECS, make_env, collect_task_dataset
from suite_a.models import SM_WM, GRU_WM, SlidingTF_WM, FullTF_WM
from suite_a.train import ABLATION_CHOICES, build_model, run_single
import argparse


def test_suite_a_envs_gym_and_ood_isolation():
    for task_name in TASK_SPECS:
        env = make_env(task_name)
        check_env(env, skip_render_check=True)

        train_perms, ood_perms = set(), set()
        for i in range(30):
            _, info_tr = env.reset(seed=i, options={"split": "train"})
            _, info_od = env.reset(seed=1000 + i, options={"split": "ood"})
            train_perms.add(info_tr["perm_id"])
            ood_perms.add(info_od["perm_id"])
        assert train_perms.isdisjoint(ood_perms), f"OOD leakage in {task_name}"


def test_suite_a_collector_shapes_and_balance():
    for task_name, spec in TASK_SPECS.items():
        ds = collect_task_dataset(task_name, num_episodes=20, split="train", seed=42)
        T = spec["episode_len"]
        assert ds["obs"].shape == (20, T, 16)
        assert ds["act"].shape == (20, T)
        assert ds["rew"].shape == (20, T)
        if spec["kind"] in ("key_door", "multistage"):
            assert ds["is_key_pickup"].any()
            assert (ds["is_door_open"] | ds["is_door_decoy"]).any()


def test_suite_a_param_parity_and_laws():
    cfg = SuiteAConfig()
    models = {
        "sm_wm": build_model("sm_wm", cfg),
        "gru_wm": build_model("gru_wm", cfg),
        "sliding_tf": build_model("sliding_tf", cfg),
        "full_tf": build_model("full_tf", cfg),
    }
    counts = {k: sum(p.numel() for p in m.parameters()) for k, m in models.items()}
    assert max(counts.values()) / min(counts.values()) <= 1.35, f"Param imbalance: {counts}"
    for k, c in counts.items():
        assert 600_000 <= c <= 950_000, f"{k} param count out of ~0.8M range: {c}"

    # Gradient flow (must be finite and > 0 for SM_WM gates including at M=0)
    sm = models["sm_wm"]
    obs = torch.randn(4, 24, 16)
    act = torch.nn.functional.one_hot(torch.randint(0, 4, (4, 24)), 4).float()
    out = sm(obs, act)
    loss = out["obs_pred"].pow(2).mean() + 0.01 * out["write_probs"].mean() + 0.01 * out["keep_probs"].mean()
    loss.backward()
    for name in ["write_gate", "evict_gate", "seq_emb", "trans_encoder", "transformer"]:
        mod = getattr(sm, name)
        gsum = sum(p.grad.abs().sum().item() for p in mod.parameters() if p.grad is not None)
        assert gsum > 0.0 and torch.isfinite(torch.tensor(gsum)), f"Bad grad in {name}: {gsum}"


def test_full_tf_forward_matches_step():
    model = FullTF_WM(obs_dim=16, act_dim=4, d_model=128, max_seq_len=64, num_layers=3, dim_feedforward=512)
    model.eval()
    obs = torch.randn(2, 20, 16)
    act = torch.nn.functional.one_hot(torch.randint(0, 4, (2, 20)), 4).float()
    with torch.no_grad():
        out_par = model(obs, act)["obs_pred"]
        state = model.init_state(2, "cpu")
        step_preds = []
        for t in range(20):
            (p, _), state, _ = model.step(obs[:, t], act[:, t], state)
            step_preds.append(p)
        out_seq = torch.stack(step_preds, dim=1)
    assert (out_par - out_seq).abs().max().item() < 1e-4


def test_sm_wm_causality_and_memory_cap():
    sm = SM_WM(obs_dim=16, act_dim=4, d_model=128, num_slots=16, recency_len=4, num_layers=3, dim_feedforward=512)
    sm.eval()
    obs1 = torch.randn(2, 24, 16)
    act1 = torch.nn.functional.one_hot(torch.randint(0, 4, (2, 24)), 4).float()
    obs2 = obs1.clone()
    obs2[:, 12:, :] += 3.0
    with torch.no_grad():
        p1 = sm(obs1, act1)["obs_pred"]
        p2 = sm(obs2, act1)["obs_pred"]
    assert (p1[:, :11, :] - p2[:, :11, :]).abs().max().item() < 1e-5

    state = sm.init_state(2, "cpu")
    for _ in range(40):
        _, state, _ = sm.step(torch.randn(2, 16), act1[:, 0], state, x_curr=torch.randn(2, 16))
        assert state["M"].shape == (2, 16, 128)
        assert state["recency"].shape == (2, 4, 128)


def test_ablations_smoke(tmp_path):
    for ab in ABLATION_CHOICES:
        m = build_model("sm_wm", SuiteAConfig(), ablation=ab)
        obs = torch.randn(2, 12, 16)
        act = torch.nn.functional.one_hot(torch.randint(0, 4, (2, 12)), 4).float()
        out = m(obs, act)
        assert out["obs_pred"].shape == (2, 12, 16)

    # End-to-end 1-epoch tiny smoke run
    args = argparse.Namespace(
        task="kd_L16",
        model="sm_wm",
        ablation="none",
        seed=42,
        epochs=1,
        batch_size=8,
        train_episodes=16,
        val_episodes=8,
        eval_episodes=4,
        device="cpu",
        log_every=1,
        out_dir=str(tmp_path),
        profile_latency=True,
    )
    res = run_single(args)
    assert res["memory"]["door_match_acc"] is not None
    assert "ms_per_step_T128" in res["latency"]
