"""Training & evaluation script for a single Suite A (task, model, ablation, seed) run."""
import os
import time
import argparse
import traceback
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .config import SuiteAConfig
from .envs import TASK_SPECS, collect_task_dataset
from .models import SM_WM, GRU_WM, SlidingTF_WM, FullTF_WM
from .eval import (
    eval_suite_a_memory,
    eval_scrambled_adaptation,
    eval_dreaming_and_variance,
    eval_step_latency,
)


ABLATION_CHOICES = [
    "none",
    "no_trans_tokens",
    "no_surprise_gate",
    "no_eviction",
    "no_target_margin",
    "no_rollout_loss",
    "no_recency",
]


def build_model(model_name: str, cfg: SuiteAConfig, ablation: str = "none"):
    if model_name == "sm_wm":
        return SM_WM(
            obs_dim=cfg.obs_dim,
            act_dim=cfg.act_dim,
            d_model=cfg.d_model,
            num_slots=cfg.num_slots,
            recency_len=0 if ablation == "no_recency" else cfg.recency_len,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            dim_feedforward=cfg.dim_feedforward,
            write_bias_init=cfg.write_gate_bias_init,
            evict_bias_init=cfg.evict_gate_bias_init,
            use_transition_tokens=(ablation != "no_trans_tokens"),
            use_surprise_gate=(ablation != "no_surprise_gate"),
            use_eviction=(ablation != "no_eviction"),
        )
    elif model_name == "gru_wm":
        return GRU_WM(
            obs_dim=cfg.obs_dim,
            act_dim=cfg.act_dim,
            d_model=cfg.d_model,
            hidden_size=cfg.gru_hidden,
        )
    elif model_name == "sliding_tf":
        return SlidingTF_WM(
            obs_dim=cfg.obs_dim,
            act_dim=cfg.act_dim,
            d_model=cfg.d_model,
            window_size=cfg.num_slots,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            dim_feedforward=cfg.dim_feedforward,
        )
    elif model_name == "full_tf":
        return FullTF_WM(
            obs_dim=cfg.obs_dim,
            act_dim=cfg.act_dim,
            d_model=cfg.d_model,
            max_seq_len=cfg.max_seq_len,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            dim_feedforward=cfg.dim_feedforward,
        )
    raise ValueError(f"Unknown model: {model_name}")


def run_single(args):
    cfg = SuiteAConfig()
    cfg.seed = args.seed
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    torch.manual_seed(cfg.seed)

    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    spec = TASK_SPECS[args.task]
    T = spec["episode_len"]
    num_train_eps = args.train_episodes if args.train_episodes > 0 else max(100, cfg.target_train_transitions // T)

    tag = f"{args.task}__{args.model}__{args.ablation}__seed{args.seed}"
    print(f"[{tag}] device={device} | train_eps={num_train_eps} (T={T}) | epochs={cfg.epochs}")

    train_data = collect_task_dataset(args.task, num_episodes=num_train_eps, split="train", seed=args.seed)
    val_data = collect_task_dataset(args.task, num_episodes=args.val_episodes, split="val", seed=args.seed + 1000)

    train_obs = train_data["obs"]
    train_act = F.one_hot(train_data["act"], num_classes=cfg.act_dim).float()
    train_rew = train_data["rew"].unsqueeze(-1)

    val_obs = val_data["obs"].to(device)
    val_act = F.one_hot(val_data["act"], num_classes=cfg.act_dim).float().to(device)
    val_rew = val_data["rew"].unsqueeze(-1).to(device)

    loader = DataLoader(
        TensorDataset(train_obs, train_act, train_rew),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=(num_train_eps >= cfg.batch_size),
        num_workers=2 if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(args.model, cfg, ablation=args.ablation).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs), eta_min=1e-5)

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(args.out_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_path = os.path.join(args.out_dir, f"ckpt_{tag}_best.pt")

    t0 = time.time()
    try:
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            if epoch <= cfg.warmup_epochs:
                lambda_w, lambda_k = 0.0, 0.0
            elif epoch <= cfg.ramp_epochs:
                prog = (epoch - cfg.warmup_epochs) / max(1, cfg.ramp_epochs - cfg.warmup_epochs)
                lambda_w = prog * cfg.lambda_w_max
                lambda_k = prog * cfg.lambda_k_max
            else:
                lambda_w, lambda_k = cfg.lambda_w_max, cfg.lambda_k_max

            if args.ablation == "no_target_margin":
                lambda_k = 0.0

            roll_weight = 0.0 if args.ablation == "no_rollout_loss" else cfg.rollout_loss_weight

            total_loss = 0.0
            for b_obs, b_act, b_rew in loader:
                b_obs = b_obs.to(device, non_blocking=True)
                b_act = b_act.to(device, non_blocking=True)
                b_rew = b_rew.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                    out = model(b_obs, b_act)
                    loss_obs_tf = F.mse_loss(out["obs_pred"][:, :-1], b_obs[:, 1:])
                    loss_rew = F.mse_loss(out["rew_pred"][:, :-1], b_rew[:, :-1])

                    # H=4 Unrolled Rollout Loss (primed with real x_curr, unrolled open-loop)
                    loss_roll = torch.tensor(0.0, device=device)
                    if roll_weight > 0.0 and T > cfg.rollout_horizon + 8:
                        t_start = int(torch.randint(6, T - cfg.rollout_horizon - 2, (1,)).item())
                        roll_state = model.init_state(b_obs.shape[0], device)
                        for step_p in range(t_start):
                            (_, _), roll_state, _ = model.step(
                                b_obs[:, step_p], b_act[:, step_p], roll_state, x_curr=b_obs[:, step_p + 1]
                            )
                        roll_pred_o = b_obs[:, t_start]
                        roll_losses = []
                        for step_r in range(cfg.rollout_horizon):
                            tc = t_start + step_r
                            (roll_pred_o, _), roll_state, _ = model.step(
                                roll_pred_o, b_act[:, tc], roll_state, x_curr=None
                            )
                            roll_losses.append(F.mse_loss(roll_pred_o[:, :2], b_obs[:, tc + 1, :2]))
                        loss_roll = torch.stack(roll_losses).mean()

                    loss = loss_obs_tf + (roll_weight * loss_roll) + (cfg.reward_loss_weight * loss_rew)

                    if "write_probs" in out:
                        loss_w = out["write_probs"].mean()
                        mean_active = out["m_history"].sum(dim=2).mean()
                        excess_slots = F.relu(mean_active - cfg.target_slots)
                        loss_k = (excess_slots ** 2) / cfg.num_slots
                        loss = loss + (lambda_w * loss_w) + (lambda_k * loss_k)

                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    optimizer.step()

                total_loss += loss.item()

            scheduler.step()

            model.eval()
            with torch.no_grad():
                with torch.amp.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                    v_out = model(val_obs, val_act)
                    v_obs = F.mse_loss(v_out["obs_pred"][:, :-1], val_obs[:, 1:]).item()
                    v_rew = F.mse_loss(v_out["rew_pred"][:, :-1], val_rew[:, :-1]).item()
                    val_loss = v_obs + cfg.reward_loss_weight * v_rew

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_loss": val_loss,
                    "params": param_count,
                }, best_path)

            if epoch % args.log_every == 0 or epoch == cfg.epochs:
                print(
                    f"  [{tag}] Ep {epoch:2d}/{cfg.epochs} | "
                    f"Train {total_loss / max(1, len(loader)):.4f} | Val {val_loss:.4f} (rew {v_rew:.5f})"
                )

    except Exception as e:
        traceback.print_exc()
        raise e

    elapsed_s = time.time() - t0
    ckpt = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])

    mem_res = eval_suite_a_memory(model, val_data, device=device)
    scrambled_res = eval_scrambled_adaptation(model, num_episodes=args.eval_episodes, split="ood", device=device) if args.task == "scrambled" else {}
    dream_res = eval_dreaming_and_variance(model, task_name=args.task if args.task != "kd_L128" else "kd_L32", num_episodes=min(50, args.eval_episodes), device=device)
    lat_res = eval_step_latency(model, horizons=(16, 32, 64, 128), device=device) if args.profile_latency else {}

    summary = {
        "task": args.task,
        "model": args.model,
        "ablation": args.ablation,
        "seed": args.seed,
        "params": param_count,
        "train_seconds": elapsed_s,
        "best_val_loss": best_val_loss,
        "memory": mem_res,
        "scrambled": scrambled_res,
        "dreaming": dream_res,
        "latency": lat_res,
    }
    res_path = os.path.join(args.out_dir, f"res_{tag}.pt")
    torch.save(summary, res_path)
    print(f"[{tag}] Done in {elapsed_s:.1f}s | door_acc={mem_res['door_match_acc']} | active={mem_res['mean_active_slots']}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="kd_L16", choices=list(TASK_SPECS.keys()))
    parser.add_argument("--model", type=str, default="sm_wm", choices=["sm_wm", "gru_wm", "sliding_tf", "full_tf"])
    parser.add_argument("--ablation", type=str, default="none", choices=ABLATION_CHOICES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_episodes", type=int, default=0)  # 0 = auto 200k transitions
    parser.add_argument("--val_episodes", type=int, default=400)
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--out_dir", type=str, default="results_suite_a")
    parser.add_argument("--profile_latency", action="store_true")
    args = parser.parse_args()
    run_single(args)


if __name__ == "__main__":
    main()
