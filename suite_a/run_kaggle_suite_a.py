"""Kaggle 2x T4 Orchestrator for Suite A (Phase 1: Parity & Horizon Cliff + Phase 2: Ablations)."""
import os
import sys
import json
import glob
import time
import subprocess
import numpy as np
import torch

from suite_a.config import SuiteAConfig
from suite_a.train import build_model


SEEDS = [42, 43, 44, 45, 46]
PHASE1_TASKS = ["kd_L16", "kd_L32", "kd_L64", "kd_L128", "scrambled", "multistage"]
PHASE1_MODELS = ["sm_wm", "gru_wm", "sliding_tf", "full_tf"]

PHASE2_TASKS = ["kd_L64", "scrambled", "multistage"]
PHASE2_ABLATIONS = [
    "no_trans_tokens",
    "no_surprise_gate",
    "no_eviction",
    "no_target_margin",
    "no_rollout_loss",
    "no_recency",
]


def run_job_queue(jobs, out_dir="results_suite_a", epochs=25):
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device_str = "cuda" if num_gpus > 0 else "cpu"
    workers = max(1, min(2, num_gpus))

    i = 0
    total = len(jobs)
    while i < total:
        batch = jobs[i:i + workers]
        procs = []
        for idx, job in enumerate(batch):
            task, model, ablation, seed, prof = job
            gpu_id = str(idx % num_gpus) if num_gpus > 0 else "0"
            cmd = [
                sys.executable, "-m", "suite_a.train",
                "--task", task,
                "--model", model,
                "--ablation", ablation,
                "--seed", str(seed),
                "--epochs", str(epochs),
                "--device", device_str,
                "--out_dir", out_dir,
                "--log_every", "10",
            ]
            if prof:
                cmd.append("--profile_latency")
            env = os.environ.copy()
            if num_gpus > 0:
                env["CUDA_VISIBLE_DEVICES"] = gpu_id
            print(f"[Queue {i + idx + 1}/{total}] GPU {gpu_id} -> {task} | {model} | {ablation} | seed={seed}")
            procs.append((job, subprocess.Popen(cmd, env=env)))

        for job, p in procs:
            rc = p.wait()
            if rc != 0:
                print(f"[WARN] Job failed (rc={rc}): {job}")
        i += workers


def aggregate_results(out_dir="results_suite_a"):
    files = sorted(glob.glob(os.path.join(out_dir, "res_*.pt")))
    grouped = {}
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        key = (d["task"], d["model"], d["ablation"])
        grouped.setdefault(key, []).append(d)

    summary = {}
    print("\n" + "=" * 78)
    print("SUITE A AGGREGATED RESULTS (Mean +/- Std across seeds)")
    print("=" * 78)
    for (task, model, ab), runs in sorted(grouped.items()):
        n = len(runs)
        obs_mse = [r["memory"]["next_obs_mse"] for r in runs]
        door_acc = [r["memory"]["door_match_acc"] for r in runs if r["memory"]["door_match_acc"] is not None]
        active = [r["memory"]["mean_active_slots"] for r in runs if r["memory"]["mean_active_slots"] is not None]
        evict = [r["memory"]["post_door_eviction_rate"] for r in runs if r["memory"]["post_door_eviction_rate"] is not None]
        carry = [r["memory"]["carry_retention_rate"] for r in runs if r["memory"]["carry_retention_rate"] is not None]
        d40 = [r["dreaming"]["pos_mse_H40"] for r in runs]
        dvar = [r["dreaming"]["dream_pos_var"] for r in runs]

        entry = {
            "n_seeds": n,
            "params": runs[0]["params"],
            "next_obs_mse_mean": float(np.mean(obs_mse)),
            "next_obs_mse_std": float(np.std(obs_mse)),
            "dream_H40_mse_mean": float(np.mean(d40)),
            "dream_H40_mse_std": float(np.std(d40)),
            "dream_pos_var_mean": float(np.mean(dvar)),
        }
        line = f"{task:10s} | {model:10s} | {ab:16s} (n={n}) | obs_mse={entry['next_obs_mse_mean']:.4f}"

        if door_acc:
            entry["door_match_acc_mean"] = float(np.mean(door_acc))
            entry["door_match_acc_std"] = float(np.std(door_acc))
            line += f" | door_acc={entry['door_match_acc_mean']:.3f}+/-{entry['door_match_acc_std']:.3f}"
        if active:
            entry["active_slots_mean"] = float(np.mean(active))
            entry["carry_retention_mean"] = float(np.mean(carry))
            entry["post_door_eviction_mean"] = float(np.mean(evict))
            line += f" | slots={entry['active_slots_mean']:.2f} carry={entry['carry_retention_mean']:.2f} evict={entry['post_door_eviction_mean']:.2f}"
        if runs[0]["scrambled"]:
            for k in ("pre_flip_mse", "discovery_mse", "short_recall_mse", "long_recall_mse"):
                vals = [r["scrambled"][k] for r in runs]
                entry[f"{k}_mean"] = float(np.mean(vals))
                entry[f"{k}_std"] = float(np.std(vals))
            line += (
                f" | scr_disc={entry['discovery_mse_mean']:.4f}"
                f" scr_short={entry['short_recall_mse_mean']:.4f}"
                f" scr_long={entry['long_recall_mse_mean']:.4f}"
            )
        lat_runs = [r["latency"] for r in runs if r["latency"]]
        if lat_runs:
            entry["latency"] = lat_runs[0]
            line += f" | lat_T128={lat_runs[0].get('ms_per_step_T128', 0):.3f}ms/step"

        summary[f"{task}__{model}__{ab}"] = entry
        print(line)

    out_json = os.path.join(out_dir, "suite_a_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {out_json}")


def main():
    t_start = time.time()
    print("=== Suite A Kaggle Benchmark (Phase 1 + Phase 2) ===")
    print("PyTorch:", torch.__version__, "| CUDA:", torch.cuda.is_available(), "| GPUs:", torch.cuda.device_count())

    cfg = SuiteAConfig()
    for m in PHASE1_MODELS:
        mod = build_model(m, cfg)
        print(f"Model {m:12s}: {sum(p.numel() for p in mod.parameters()):,} params")

    jobs = []
    # Phase 1: Parity & Horizon Cliff + Scrambled + MultiStage (4 models x 6 tasks x 5 seeds)
    for task in PHASE1_TASKS:
        for model in PHASE1_MODELS:
            for seed in SEEDS:
                prof = (task == "kd_L128" and seed == 42)
                jobs.append((task, model, "none", seed, prof))

    # Phase 2: Component Ablations on SM_WM (6 ablations x 3 tasks x 5 seeds)
    for task in PHASE2_TASKS:
        for ab in PHASE2_ABLATIONS:
            for seed in SEEDS:
                jobs.append((task, "sm_wm", ab, seed, False))

    print(f"Total scheduled runs: {len(jobs)}")
    run_job_queue(jobs, out_dir="results_suite_a", epochs=cfg.epochs)
    aggregate_results("results_suite_a")
    print(f"=== Suite A Finished in {(time.time() - t_start) / 60.0:.1f} min ===")


if __name__ == "__main__":
    main()
