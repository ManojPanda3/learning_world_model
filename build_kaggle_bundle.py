import os
import sys
import base64

project_dir = "/home/manoj/Projects/Research/learning_world_model/sm_wm_project"
files = ["config.py", "env_dataset.py", "models.py", "eval.py", "test_laws.py", "test_rq1.py", "train.py"]

packed = {}
for fname in files:
    with open(os.path.join(project_dir, fname), "rb") as f:
        packed[fname] = base64.b64encode(f.read()).decode("ascii")

template = '''# Auto-generated self-contained benchmark suite
import os
import sys
import base64
import subprocess
import glob
import torch
import numpy as np

print("=== Starting SM-WM Kaggle Benchmark Suite ===")
print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}:", torch.cuda.get_device_name(i))

# Unpack embedded project files
EMBEDDED_FILES = {packed_repr}

for fname, b64_str in EMBEDDED_FILES.items():
    with open(fname, "wb") as f:
        f.write(base64.b64decode(b64_str))
    print(f"Unpacked: {fname}")

# 1. Install gymnasium, pytest, and wandb
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gymnasium", "pytest", "wandb"], check=True)

# 2. Run Test Laws & RQ1 tests
print("\\n--- Running Law-Enforced Verification & RQ1 Test Suite ---")
test_res = subprocess.run([sys.executable, "-m", "pytest", "test_laws.py", "test_rq1.py", "-v"])
if test_res.returncode != 0:
    print("Verification failed! Exiting.")
    sys.exit(1)
print("Verification passed successfully!")

# 3. Process-level GPU orchestration for 2x T4
runs = [
    # Wave 1
    ("sm_wm", 42, "0"),
    ("gru_wm", 42, "1"),
    # Wave 2
    ("sliding_tf", 42, "0"),
    ("sm_wm", 43, "1"),
    # Wave 3
    ("gru_wm", 43, "0"),
    ("sliding_tf", 43, "1"),
    # Wave 4
    ("sm_wm", 44, "0"),
    ("gru_wm", 44, "1"),
    # Wave 5
    ("sliding_tf", 44, "0"),
]

os.makedirs("results", exist_ok=True)

i = 0
while i < len(runs):
    pair = runs[i:i+2]
    procs = []
    for model, seed, gpu_id in pair:
        cmd = [
            sys.executable, "train.py",
            "--model", model,
            "--seed", str(seed),
            "--epochs", "30",
            "--batch_size", "64",
            "--train_episodes", "5000",
            "--val_episodes", "500",
            "--eval_episodes", "50",
            "--device", "cuda" if torch.cuda.is_available() else "cpu",
            "--out_dir", "results",
            "--use_wandb", "True",
            "--wandb_project", "sm_wm_ab_test",
        ]
        env = os.environ.copy()
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
        elif torch.cuda.is_available():
            env["CUDA_VISIBLE_DEVICES"] = "0"
            
        print(f"\\nLaunching {model} (seed {seed}) on GPU {gpu_id}...")
        p = subprocess.Popen(cmd, env=env)
        procs.append((model, seed, p))

    for model, seed, p in procs:
        p.wait()
        print(f"Finished {model} (seed {seed}) with exit code {p.returncode}")
    i += 2

print("\\n=== Aggregating A/B Experiment Results ===")
result_files = glob.glob("results/results_*.pt")
all_results = {}
for rf in result_files:
    data = torch.load(rf)
    m = data["model"]
    s = data["seed"]
    if m not in all_results:
        all_results[m] = []
    all_results[m].append(data)

print(f"\\nFound results for {len(all_results)} models.")
for m, runs_data in all_results.items():
    print(f"\\nModel: {m} ({len(runs_data)} seeds)")
    pickups = [r["rq1"]["write_pickup_rate"] for r in runs_data if r["rq1"]["write_pickup_rate"] is not None]
    distractors = [r["rq1"]["write_distractor_rate"] for r in runs_data if r["rq1"]["write_distractor_rate"] is not None]
    evictions = [r["rq1"]["post_door_eviction_rate"] for r in runs_data if r["rq1"]["post_door_eviction_rate"] is not None]
    mses = [r["rq1"]["next_obs_mse"] for r in runs_data]
    if pickups:
        print(f"  RQ1 Write Pickup Rate:     {np.mean(pickups):.4f} +/- {np.std(pickups):.4f}")
        print(f"  RQ1 Write Distractor Rate: {np.mean(distractors):.4f} +/- {np.std(distractors):.4f}")
        print(f"  RQ1 Post-Door Eviction:    {np.mean(evictions):.4f} +/- {np.std(evictions):.4f}")
    print(f"  Next-Obs MSE:              {np.mean(mses):.4f} +/- {np.std(mses):.4f}")

    pos_h10 = [r["rq2"]["pos_mse_H10"] for r in runs_data]
    pos_h25 = [r["rq2"]["pos_mse_H25"] for r in runs_data]
    pos_h40 = [r["rq2"]["pos_mse_H40"] for r in runs_data]
    print(f"  RQ2 Imagination Pos MSE:   H10={np.mean(pos_h10):.4f}, H25={np.mean(pos_h25):.4f}, H40={np.mean(pos_h40):.4f}")

    disc = [r["rq3"]["discovery_err"] for r in runs_data]
    short = [r["rq3"]["short_recall_err"] for r in runs_data]
    long = [r["rq3"]["long_recall_err"] for r in runs_data]
    print(f"  RQ3 OOD Adaptation Error:  Discovery={np.mean(disc):.4f}, Short={np.mean(short):.4f}, Long={np.mean(long):.4f}")

print("\\n=== SM-WM Benchmark Complete ===")
'''

final_code = template.replace("{packed_repr}", repr(packed))
with open("/home/manoj/Projects/Research/learning_world_model/kaggle_sm_wm/run_kaggle_suite.py", "w") as f:
    f.write(final_code)
print("Successfully generated valid self-contained runner!")
