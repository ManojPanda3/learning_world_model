"""Suite A Evaluation Harness:
- Horizon Cliff & Door-Match Accuracy (Key-Door L=16..128 & Multi-Stage)
- Learned Forgetting & Selective Storage (RQ1 + multi-stage eviction)
- Scrambled Controller Online Adaptation (pre-flip, discovery, short, long recall)
- Open-Loop Imagination & Trajectory Variance (RQ2, anti-Frozen-Goalie check)
- Per-step Wall-Clock Latency Profiling (O(1) vs O(T))
"""
import time
import numpy as np
import torch
import torch.nn.functional as F

from .envs import TASK_SPECS, make_env


def eval_suite_a_memory(model, dataset, device="cpu"):
    """Evaluate next-obs MSE, door-match accuracy, and SM_WM slot write/eviction metrics."""
    model.eval()
    obs = dataset["obs"].to(device)
    act = F.one_hot(dataset["act"].to(device), num_classes=4).float()
    rew = dataset["rew"].to(device)

    is_novel = dataset["is_novel_probe"].cpu().numpy()
    is_pickup = dataset["is_key_pickup"].cpu().numpy()
    is_door_open = dataset["is_door_open"].cpu().numpy()
    is_door_decoy = dataset["is_door_decoy"].cpu().numpy()
    is_dist = dataset["is_distractor"].cpu().numpy()

    N, T, _ = obs.shape
    with torch.no_grad():
        out = model(obs, act)

    obs_pred = out["obs_pred"]
    rew_pred = out["rew_pred"].squeeze(-1)  # [N, T]

    next_obs_mse = F.mse_loss(obs_pred[:, :-1], obs[:, 1:]).item()
    rew_mse = F.mse_loss(rew_pred[:, :-1], rew[:, :-1]).item()

    # Door-match accuracy on steps where a door outcome occurs (is_door_open or is_door_decoy)
    door_mask = is_door_open | is_door_decoy
    door_match_acc = None
    door_rew_mse = None
    if door_mask.any():
        r_pred_np = rew_pred.cpu().numpy()[door_mask]
        r_true_np = rew.cpu().numpy()[door_mask]
        door_match_acc = float(np.mean((r_pred_np > 0.5) == (r_true_np > 0.5)))
        door_rew_mse = float(np.mean((r_pred_np - r_true_np) ** 2))

    res = {
        "model_type": type(model).__name__,
        "next_obs_mse": next_obs_mse,
        "rew_mse": rew_mse,
        "door_match_acc": door_match_acc,
        "door_rew_mse": door_rew_mse,
        "write_novel_rate": None,
        "write_pickup_rate": None,
        "write_distractor_rate": None,
        "carry_retention_rate": None,
        "post_door_eviction_rate": None,
        "mean_active_slots": None,
    }

    if "write_probs" not in out:
        return res

    write_probs = out["write_probs"].squeeze(-1).cpu().numpy()  # [N, T]
    keep_probs = out["keep_probs"].squeeze(-1).cpu().numpy()    # [N, T, K]
    m_history = out["m_history"].squeeze(-1).cpu().numpy()      # [N, T, K]

    res["write_novel_rate"] = float(np.mean(write_probs[is_novel])) if is_novel.any() else 0.0
    res["write_pickup_rate"] = float(np.mean(write_probs[is_pickup])) if is_pickup.any() else 0.0
    res["write_distractor_rate"] = float(np.mean(write_probs[is_dist])) if is_dist.any() else 0.0
    res["mean_active_slots"] = float(np.mean(np.sum(m_history > 0.5, axis=-1)))

    # Per-stage carry retention & post-door eviction (works for both 1-stage and 3-stage)
    eviction_drops = []
    during_carry_keeps = []
    for ep in range(min(N, 250)):
        pick_steps = np.where(is_pickup[ep])[0]
        door_steps = np.where(is_door_open[ep] | is_door_decoy[ep])[0]
        for t_pick in pick_steps:
            # Find first door step after t_pick
            later_doors = door_steps[door_steps > t_pick]
            if len(later_doors) == 0:
                continue
            t_door = int(later_doors[0])
            if t_door + 1 >= T:
                continue
            # Identify slot whose occupancy increased most at pickup step t_pick
            prev_m = m_history[ep, t_pick - 1] if t_pick > 0 else np.zeros_like(m_history[ep, t_pick])
            slot_idx = int(np.argmax(m_history[ep, t_pick] - prev_m))
            if t_door > t_pick + 1:
                during_k = keep_probs[ep, t_pick + 1:t_door, slot_idx]
                during_carry_keeps.extend(during_k.tolist())
            k_after = min(
                float(keep_probs[ep, t_door, slot_idx]),
                float(keep_probs[ep, min(t_door + 1, T - 1), slot_idx]),
            )
            eviction_drops.append(float(k_after < 0.5))

    res["carry_retention_rate"] = float(np.mean(np.array(during_carry_keeps) >= 0.5)) if during_carry_keeps else 0.0
    res["post_door_eviction_rate"] = float(np.mean(eviction_drops)) if eviction_drops else 0.0
    return res


def eval_scrambled_adaptation(model, num_episodes=100, split="ood", device="cpu"):
    """
    Zero-gradient online adaptation on ScrambledControllerEnv (mid-episode flip at t_flip).
    Strictly causal: predicts x_{t+1} BEFORE observing x_{t+1}, then passes x_curr=x_{t+1}.
    """
    model.eval()
    env = make_env("scrambled")
    T = TASK_SPECS["scrambled"]["episode_len"]
    rng = np.random.default_rng(7000)

    errs_pre, errs_disc, errs_short, errs_long = [], [], [], []
    for ep in range(num_episodes):
        obs, info = env.reset(seed=9000 + ep, options={"split": split})
        t_flip = info["t_flip"]
        state = model.init_state(batch_size=1, device=device)

        curr_obs = obs
        for t in range(T - 1):
            a = t if t < 4 else int(rng.integers(0, 4))
            next_obs, _, _, _, _ = env.step(a)

            x_t = torch.tensor(curr_obs, dtype=torch.float32, device=device).unsqueeze(0)
            x_next = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
            a_t = F.one_hot(torch.tensor([a], dtype=torch.long, device=device), num_classes=4).float()

            with torch.no_grad():
                # First predict (without seeing x_next) to score prediction error, then update state with x_next
                if hasattr(model, "predict_step"):
                    (pred_o, _), _ = model.predict_step(x_t, a_t, state)
                    _, state, _ = model.step(x_t, a_t, state, x_curr=x_next)
                else:
                    (pred_o, _), state, _ = model.step(x_t, a_t, state, x_curr=x_next)

            pos_err = F.mse_loss(pred_o[0, :2], x_next[0, :2]).item()
            if 6 <= t < t_flip:
                errs_pre.append(pos_err)
            elif t_flip <= t < t_flip + 4:
                errs_disc.append(pos_err)
            elif t_flip + 4 <= t < t_flip + 16:
                errs_short.append(pos_err)
            elif t_flip + 16 <= t < T - 1:
                errs_long.append(pos_err)

            curr_obs = next_obs

    return {
        "pre_flip_mse": float(np.mean(errs_pre)) if errs_pre else 0.0,
        "discovery_mse": float(np.mean(errs_disc)) if errs_disc else 0.0,
        "short_recall_mse": float(np.mean(errs_short)) if errs_short else 0.0,
        "long_recall_mse": float(np.mean(errs_long)) if errs_long else 0.0,
    }


def eval_dreaming_and_variance(model, task_name="kd_L32", num_episodes=50, horizons=(10, 25, 40), device="cpu"):
    """Open-loop imagination rollout (RQ2) + trajectory variance check."""
    model.eval()
    env = make_env(task_name)
    max_h = max(horizons)
    prime_steps = 20
    rng = np.random.default_rng(3000)

    results = {h: [] for h in horizons}
    dream_vars = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=2000 + ep, options={"split": "val"})
        traj_obs = [obs]
        traj_act = []
        for t in range(prime_steps + max_h):
            a = t if t < 4 else int(rng.integers(0, 4))
            n_obs, _, _, _, _ = env.step(a)
            traj_obs.append(n_obs)
            traj_act.append(a)

        obs_t = torch.tensor(np.array(traj_obs), dtype=torch.float32, device=device)
        act_t = F.one_hot(torch.tensor(traj_act, dtype=torch.long, device=device), num_classes=4).float()

        with torch.no_grad():
            state = model.init_state(batch_size=1, device=device)
            for t in range(prime_steps):
                (pred_o, _), state, _ = model.step(
                    obs_t[t:t + 1], act_t[t:t + 1], state, x_curr=obs_t[t + 1:t + 2]
                )

            dreamed = []
            curr_o = pred_o
            for step_h in range(max_h):
                t_idx = prime_steps + step_h
                (curr_o, _), state, _ = model.step(curr_o, act_t[t_idx:t_idx + 1], state, x_curr=None)
                dreamed.append(curr_o)

            dreamed_stack = torch.cat(dreamed, dim=0)  # [max_h, 16]
            real_stack = obs_t[prime_steps + 1:prime_steps + 1 + max_h]
            dream_vars.append(dreamed_stack[:, :2].var(dim=0).sum().item())

            for h in horizons:
                results[h].append(F.mse_loss(dreamed_stack[:h, :2], real_stack[:h, :2]).item())

    out = {f"pos_mse_H{h}": float(np.mean(results[h])) for h in horizons}
    out["dream_pos_var"] = float(np.mean(dream_vars))
    return out


def eval_step_latency(model, horizons=(16, 32, 64, 128), device="cpu"):
    """Measure average ms/step across horizon lengths T to demonstrate O(1) vs O(T) scaling."""
    model.eval()
    latencies = {}
    for T in horizons:
        x = torch.randn(1, 16, device=device)
        a = torch.zeros(1, 4, device=device)
        a[0, 0] = 1.0

        with torch.no_grad():
            # Warmup
            state = model.init_state(1, device)
            for _ in range(min(T, 16)):
                _, state, _ = model.step(x, a, state, x_curr=x)

            if "cuda" in str(device):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            state = model.init_state(1, device)
            for _ in range(T):
                _, state, _ = model.step(x, a, state, x_curr=x)
            if "cuda" in str(device):
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies[f"ms_per_step_T{T}"] = elapsed_ms / T
    return latencies
