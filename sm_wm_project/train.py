"""Unified training script with:
- Anti-collapse curriculum & AMP
- Offline W&B logging for metrics & loss curves
- Top-level try-except with emergency checkpointing (_epoch_num)
- Best checkpoint saving (lowest val loss)
- 1 model/seed per GPU process execution for 2x T4 utilization
"""
import os
import sys
import argparse
import time
import traceback
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from config import Config
from env_dataset import collect_dataset
from models import SM_WM, GRU_WM, SlidingTF_WM
from eval import eval_rq1_learned_forgetting, eval_rq2_imagination, eval_rq3_online_adaptation

# Try importing wandb; if not available, run in dummy mode
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def build_model(model_name: str, cfg: Config):
    if model_name == "sm_wm":
        return SM_WM(
            obs_dim=cfg.obs_dim,
            act_dim=cfg.act_dim,
            d_model=cfg.d_model,
            num_slots=cfg.num_slots,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            write_bias_init=cfg.write_gate_bias_init,
            evict_bias_init=cfg.evict_gate_bias_init,
        )
    elif model_name == "gru_wm":
        return GRU_WM(
            obs_dim=cfg.obs_dim,
            act_dim=cfg.act_dim,
            d_model=cfg.d_model,
        )
    elif model_name == "sliding_tf":
        return SlidingTF_WM(
            obs_dim=cfg.obs_dim,
            act_dim=cfg.act_dim,
            d_model=cfg.d_model,
            window_size=cfg.num_slots,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def save_checkpoint(model, optimizer, epoch, val_loss, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_loss": val_loss,
    }, path)
    print(f"[Checkpoint] Saved {path} (epoch {epoch})")


def train(args):
    cfg = Config()
    cfg.seed = args.seed
    torch.manual_seed(cfg.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    print(f"[{args.model} | seed {args.seed}] Starting on device: {device}")

    # Initialize offline W&B
    wb_run = None
    if HAS_WANDB and args.use_wandb:
        os.environ["WANDB_MODE"] = "offline"
        wb_run = wandb.init(
            project=args.wandb_project,
            group=args.model,
            name=f"{args.model}_seed{args.seed}",
            config={**vars(args), **vars(cfg)},
            dir=args.out_dir,
            reinit=True,
        )

    # Generate datasets
    print(f"Collecting datasets (train: {args.train_episodes}, val: {args.val_episodes})...")
    train_data = collect_dataset(num_episodes=args.train_episodes, split="train", seed=args.seed)
    val_data = collect_dataset(num_episodes=args.val_episodes, split="val", seed=args.seed + 100)

    train_obs = train_data["obs"][:, :cfg.seq_len]  # [N, T, 16]
    train_act = F.one_hot(train_data["act"][:, :cfg.seq_len], num_classes=cfg.act_dim).float()
    train_rew = train_data["rew"][:, :cfg.seq_len].unsqueeze(-1)  # [N, T, 1]

    val_obs = val_data["obs"][:, :cfg.seq_len].to(device)
    val_act = F.one_hot(val_data["act"][:, :cfg.seq_len], num_classes=cfg.act_dim).float().to(device)
    val_rew = val_data["rew"][:, :cfg.seq_len].unsqueeze(-1).to(device)

    dataset = TensorDataset(train_obs, train_act, train_rew)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2 if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(args.model, cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(args.out_dir, exist_ok=True)
    best_val_loss = float("inf")
    current_epoch = 0

    try:
        total_start = time.time()
        for epoch in range(1, args.epochs + 1):
            current_epoch = epoch
            model.train()
            total_loss = 0.0
            total_obs_loss = 0.0
            total_rew_loss = 0.0
            total_w_pen = 0.0
            total_k_pen = 0.0

            # Curriculum penalty ramp
            if epoch <= cfg.warmup_epochs:
                lambda_w = 0.0
                lambda_k = 0.0
            elif epoch <= cfg.ramp_epochs:
                prog = (epoch - cfg.warmup_epochs) / (cfg.ramp_epochs - cfg.warmup_epochs)
                lambda_w = prog * cfg.lambda_w_max
                lambda_k = prog * cfg.lambda_k_max
            else:
                lambda_w = cfg.lambda_w_max
                lambda_k = cfg.lambda_k_max

            for b_obs, b_act, b_rew in loader:
                b_obs = b_obs.to(device, non_blocking=True)
                b_act = b_act.to(device, non_blocking=True)
                b_rew = b_rew.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                    out = model(b_obs, b_act)
                    pred_obs = out["obs_pred"][:, :-1]
                    target_obs = b_obs[:, 1:]
                    loss_obs = F.mse_loss(pred_obs, target_obs)

                    pred_rew = out["rew_pred"][:, :-1]
                    target_rew = b_rew[:, :-1]
                    loss_rew = F.mse_loss(pred_rew, target_rew)

                    loss = loss_obs + loss_rew

                    loss_w = torch.tensor(0.0, device=device)
                    loss_k = torch.tensor(0.0, device=device)
                    if "write_probs" in out:
                        loss_w = out["write_probs"].mean()
                        # Target-margin penalty: free budget up to cfg.target_slots (e.g. 5 slots)
                        # Penalizes only when mean active slots exceed target_slots
                        mean_active = out["m_history"].sum(dim=2).mean() # [B, T, K, 1] -> active slots per step
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
                total_obs_loss += loss_obs.item()
                total_rew_loss += loss_rew.item()
                total_w_pen += loss_w.item()
                total_k_pen += loss_k.item()

            scheduler.step()
            num_b = len(loader)
            train_obs_loss_avg = total_obs_loss / num_b
            train_rew_loss_avg = total_rew_loss / num_b
            train_loss_avg = total_loss / num_b

            # Validation step
            model.eval()
            with torch.no_grad():
                with torch.amp.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                    v_out = model(val_obs, val_act)
                    v_loss_obs = F.mse_loss(v_out["obs_pred"][:, :-1], val_obs[:, 1:]).item()
                    v_loss_rew = F.mse_loss(v_out["rew_pred"][:, :-1], val_rew[:, :-1]).item()
                    val_loss = v_loss_obs + v_loss_rew

            # Check for best checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                best_path = os.path.join(args.out_dir, f"{args.model}_seed{args.seed}_best.pt")
                save_checkpoint(model, optimizer, epoch, val_loss, best_path)

            # Wandb logging
            if wb_run:
                log_dict = {
                    "epoch": epoch,
                    "train/loss": train_loss_avg,
                    "train/obs_loss": train_obs_loss_avg,
                    "train/rew_loss": train_rew_loss_avg,
                    "train/pen_write": total_w_pen / num_b,
                    "train/pen_keep": total_k_pen / num_b,
                    "val/loss": val_loss,
                    "val/obs_loss": v_loss_obs,
                    "val/rew_loss": v_loss_rew,
                    "lr": scheduler.get_last_lr()[0],
                }
                wb_run.log(log_dict)

            if epoch % args.log_every == 0 or epoch == args.epochs:
                print(
                    f"Epoch {epoch:2d}/{args.epochs} | "
                    f"Train Loss: {train_loss_avg:.4f} (obs: {train_obs_loss_avg:.4f}, rew: {train_rew_loss_avg:.4f}) | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"lr: {scheduler.get_last_lr()[0]:.2e}"
                )

        elapsed = time.time() - total_start
        print(f"[{args.model} | seed {args.seed}] Training complete in {elapsed:.1f}s.")

        # Save final checkpoint
        final_ckpt = os.path.join(args.out_dir, f"{args.model}_seed{args.seed}_epoch_{current_epoch}.pt")
        save_checkpoint(model, optimizer, current_epoch, val_loss, final_ckpt)

    except Exception as e:
        print(f"\n[FATAL ERROR in Epoch {current_epoch}] Exception encountered: {e}")
        traceback.print_exc()
        emergency_ckpt = os.path.join(args.out_dir, f"{args.model}_seed{args.seed}_epoch_{current_epoch}_error.pt")
        save_checkpoint(model, optimizer, current_epoch, -1.0, emergency_ckpt)
        print(f"Emergency checkpoint saved to {emergency_ckpt}")
        if wb_run:
            wb_run.alert(title="Run Failed", text=f"Error at epoch {current_epoch}: {e}")
            wb_run.finish(exit_code=1)
        raise e

    # Run evaluations
    print(f"\nRunning A/B Test 1 (RQ1: Learned Forgetting)...")
    rq1_res = eval_rq1_learned_forgetting(model, val_data, device=device)
    print(f"RQ1 Results: {rq1_res}")

    print(f"Running A/B Test 2 (RQ2: Open-Loop Dreaming)...")
    rq2_res = eval_rq2_imagination(model, num_episodes=args.eval_episodes, device=device)
    print(f"RQ2 Results: {rq2_res}")

    print(f"Running A/B Test 3 (RQ3: OOD Adaptation)...")
    rq3_res = eval_rq3_online_adaptation(model, num_episodes=args.eval_episodes, device=device)
    print(f"RQ3 Results: {rq3_res}")

    if wb_run:
        wb_run.log({
            "rq1/next_obs_mse": rq1_res["next_obs_mse"],
            "rq1/write_novel_rate": rq1_res["write_novel_rate"] or 0.0,
            "rq1/write_pickup_rate": rq1_res["write_pickup_rate"] or 0.0,
            "rq1/write_distractor_rate": rq1_res["write_distractor_rate"] or 0.0,
            "rq1/post_door_eviction_rate": rq1_res["post_door_eviction_rate"] or 0.0,
            "rq1/mean_active_slots": rq1_res["mean_active_slots"] or 0.0,
            "rq2/pos_mse_H10": rq2_res["pos_mse_H10"],
            "rq2/pos_mse_H25": rq2_res["pos_mse_H25"],
            "rq2/pos_mse_H40": rq2_res["pos_mse_H40"],
            "rq3/discovery_err": rq3_res["discovery_err"],
            "rq3/short_recall_err": rq3_res["short_recall_err"],
            "rq3/long_recall_err": rq3_res["long_recall_err"],
        })
        wb_run.finish()

    results = {
        "model": args.model,
        "seed": args.seed,
        "rq1": rq1_res,
        "rq2": rq2_res,
        "rq3": rq3_res,
    }
    torch.save(results, os.path.join(args.out_dir, f"results_{args.model}_seed{args.seed}.pt"))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="sm_wm", choices=["sm_wm", "gru_wm", "sliding_tf"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_episodes", type=int, default=5000)
    parser.add_argument("--val_episodes", type=int, default=500)
    parser.add_argument("--eval_episodes", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--use_wandb", type=bool, default=True)
    parser.add_argument("--wandb_project", type=str, default="sm_wm_ab_test")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
