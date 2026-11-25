"""Evaluation harness for A/B Tests: RQ1 (Learned Forgetting & Selective Storage),
RQ2 (Open-Loop Imagination), and RQ3 (Zero-Gradient Online Adaptation).
LOCKED / READ-ONLY file.
"""
import torch
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from env_dataset import KeymapMemoryEnv, collect_dataset

def eval_rq1_learned_forgetting(model, dataset, device="cpu"):
    """
    A/B Test 1: RQ1 - Learned Forgetting & Selective Storage
    Metrics on validation dataset:
    1. Write Rate by Event Type: avg w_t on novel_probe, key_pickup, distractor.
    2. Eviction Trigger Rate: avg drop in k_i for the slot written on key_pickup post door-open.
    3. Memory Efficiency: mean active slots vs Next-Obs MSE.
    """
    model.eval()
    obs = dataset["obs"].to(device)  # [N, 80, 16]
    act_idx = dataset["act"].to(device)  # [N, 80]
    act = F.one_hot(act_idx, num_classes=4).float()
    rew = dataset["rew"].to(device)  # [N, 80]
    is_novel = dataset["is_novel_probe"].cpu().numpy()
    is_pickup = dataset["is_key_pickup"].cpu().numpy()
    is_door = dataset["is_door_open"].cpu().numpy()
    is_dist = dataset["is_distractor"].cpu().numpy()

    N, T, _ = obs.shape
    with torch.no_grad():
        out = model(obs, act)

    obs_pred = out["obs_pred"]  # [N, T, 16]
    # Prediction at t is predicting obs at t+1 (from 0..T-2 to 1..T-1)
    obs_target = obs[:, 1:]
    pred_aligned = obs_pred[:, :-1]
    mse = F.mse_loss(pred_aligned, obs_target).item()

    if "write_probs" not in out:
        # Baseline model without explicit memory gates
        return {
            "model_type": type(model).__name__,
            "next_obs_mse": mse,
            "write_novel_rate": None,
            "write_pickup_rate": None,
            "write_distractor_rate": None,
            "post_door_eviction_rate": None,
            "mean_active_slots": None,
        }

    write_probs = out["write_probs"].squeeze(-1).cpu().numpy()  # [N, T]
    keep_probs = out["keep_probs"].squeeze(-1).cpu().numpy()    # [N, T, K]
    m_history = out["m_history"].squeeze(-1).cpu().numpy()      # [N, T, K]

    novel_writes = write_probs[is_novel]
    pickup_writes = write_probs[is_pickup]
    dist_writes = write_probs[is_dist]

    write_novel_rate = float(np.mean(novel_writes)) if len(novel_writes) > 0 else 0.0
    write_pickup_rate = float(np.mean(pickup_writes)) if len(pickup_writes) > 0 else 0.0
    write_dist_rate = float(np.mean(dist_writes)) if len(dist_writes) > 0 else 0.0

    # Eviction analysis:
    # On pickup step, find which slot had the highest write probability / occupancy change
    # Then inspect keep_probs or occupancy for that slot after door-open step
    eviction_drops = []
    for ep in range(min(N, 200)):
        pickup_indices = np.where(is_pickup[ep])[0]
        door_indices = np.where(is_door[ep])[0]
        if len(pickup_indices) > 0 and len(door_indices) > 0:
            t_pick = pickup_indices[0]
            t_door = door_indices[0]
            if t_door > t_pick and t_door + 1 < T:
                # Find slot that became occupied at t_pick
                slot_idx = int(np.argmax(m_history[ep, t_pick]))
                # Keep probability at door opening / right after door
                k_after = keep_probs[ep, t_door + 1, slot_idx]
                evicted = float(k_after < 0.5)
                eviction_drops.append(evicted)

    eviction_rate = float(np.mean(eviction_drops)) if len(eviction_drops) > 0 else 0.0
    mean_active_slots = float(np.mean(np.sum(m_history > 0.5, axis=-1)))

    return {
        "model_type": type(model).__name__,
        "next_obs_mse": mse,
        "write_novel_rate": write_novel_rate,
        "write_pickup_rate": write_pickup_rate,
        "write_distractor_rate": write_dist_rate,
        "post_door_eviction_rate": eviction_rate,
        "mean_active_slots": mean_active_slots,
    }


def eval_rq2_imagination(model, num_episodes=50, horizons=(10, 25, 40), device="cpu"):
    """
    A/B Test 2: RQ2 - Dreaming (Open-Loop Imagination) Consistency
    1. Prime model with real observations for t = 0..19.
    2. Switch to Dreaming Mode for H in {10, 25, 40} steps:
       Feed predicted x_{t-1} as input along with actual action a_{t-1}.
    """
    model.eval()
    env = KeymapMemoryEnv()
    results = {h: {"pos_mse": [], "rew_f1": []} for h in horizons}
    max_h = max(horizons)

    for ep in range(num_episodes):
        obs, info = env.reset(seed=2000 + ep, options={"split": "val"})
        traj_obs = [obs]
        traj_act = []
        traj_rew = []
        for t in range(20 + max_h):
            a = int(np.random.randint(0, 4))
            n_obs, r, term, trunc, _ = env.step(a)
            traj_obs.append(n_obs)
            traj_act.append(a)
            traj_rew.append(r)

        traj_obs = torch.tensor(np.array(traj_obs), dtype=torch.float32, device=device)
        traj_act = F.one_hot(torch.tensor(traj_act, dtype=torch.long, device=device), num_classes=4).float()

        for H in horizons:
            # Prime model up to t=19
            state = model.init_state(batch_size=1, device=device)
            for t in range(20):
                x = traj_obs[t:t+1]
                a = traj_act[t:t+1]
                (pred_o, pred_r), state, _ = model.step(x, a, state)

            # Dream open-loop for H steps
            dreamed_obs = []
            curr_pred_o = pred_o
            for step_h in range(H):
                t_idx = 20 + step_h
                a = traj_act[t_idx:t_idx+1]
                (curr_pred_o, curr_pred_r), state, _ = model.step(curr_pred_o, a, state)
                dreamed_obs.append(curr_pred_o.detach())

            dreamed_stack = torch.cat(dreamed_obs, dim=0)  # [H, 16]
            real_stack = traj_obs[21:21+H]                 # [H, 16]

            # Measure position MSE (first 2 dims)
            pos_mse = F.mse_loss(dreamed_stack[:, :2], real_stack[:, :2]).item()
            results[H]["pos_mse"].append(pos_mse)

    summary = {}
    for h in horizons:
        summary[f"pos_mse_H{h}"] = float(np.mean(results[h]["pos_mse"]))
    return summary


def eval_rq3_online_adaptation(model, num_episodes=50, device="cpu"):
    """
    A/B Test 3: RQ3 - Zero-Gradient Online Adaptation on OOD Keymaps (perms 18..23).
    Measures Next-Position Prediction Error across 3 windows:
    - Discovery (t = 1..8)
    - Short-Term Recall (t = 9..25)
    - Long-Term Recall (t = 50..80)
    """
    model.eval()
    env = KeymapMemoryEnv()
    errs_discovery = []
    errs_short = []
    errs_long = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=9000 + ep, options={"split": "ood"})
        state = model.init_state(batch_size=1, device=device)
        ep_obs = [obs]
        ep_act = []

        for t in range(80):
            # Probe 0..3 early, then walk
            a = t if t < 4 else int(np.random.randint(0, 4))
            n_obs, r, term, trunc, _ = env.step(a)
            ep_obs.append(n_obs)
            ep_act.append(a)

            x = torch.tensor(ep_obs[t], dtype=torch.float32, device=device).unsqueeze(0)
            a_t = F.one_hot(torch.tensor([a], dtype=torch.long, device=device), num_classes=4).float()
            with torch.no_grad():
                (pred_o, pred_r), state, _ = model.step(x, a_t, state)

            actual_next_pos = torch.tensor(n_obs[:2], dtype=torch.float32, device=device)
            pred_pos = pred_o[0, :2]
            pos_err = F.mse_loss(pred_pos, actual_next_pos).item()

            if 1 <= t <= 8:
                errs_discovery.append(pos_err)
            elif 9 <= t <= 25:
                errs_short.append(pos_err)
            elif 50 <= t < 80:
                errs_long.append(pos_err)

    return {
        "discovery_err": float(np.mean(errs_discovery)) if errs_discovery else 0.0,
        "short_recall_err": float(np.mean(errs_short)) if errs_short else 0.0,
        "long_recall_err": float(np.mean(errs_long)) if errs_long else 0.0,
    }
