# Project Blueprint: Un-Diluted Sparse Memory World Model (SM-WM)

**Target Hardware:** Kaggle Free Tier (2x NVIDIA T4, 32GB total VRAM, ~15–20 min for full 3-model × 3-seed A/B suite)  
**Tech Stack:** Python 3.10+, PyTorch 2.x, Gymnasium (`pip install gymnasium`), NumPy, Pytest  
**Goal:** Validate whether separating fast continuous state ($h$) from an un-diluted, state-evicted episodic memory bank ($M$) enables learned forgetting (RQ1), stable imagination rollouts (RQ2), and zero-gradient online adaptation (RQ3).

---

## 1. Repository Structure & AI Agent Rules

To prevent an AI coding assistant from silently altering your architecture or rigging evaluations, enforce strict file ownership:

```text
sm_wm_project/
├── config.py          # Hyperparameters & experiment flags (Editable)
├── env_dataset.py     # Gymnasium Env (KeymapMemoryEnv) + Vectorized Dataset Collector (LOCKED after creation)
├── models.py          # Baseline A (GRU), Baseline B (Sliding-TF), Ours (SM-WM) (Editable)
├── train.py           # Unified training pipeline with AMP, curriculum & GPU pinning (Editable)
├── eval.py            # Standardized RQ1, RQ2, RQ3 Gymnasium evaluation harness (LOCKED / READ-ONLY)
└── test_laws.py       # Pytest architectural, gradient & Gymnasium invariants (LOCKED / READ-ONLY)
```

**Strict Workflow Rule:** After *every* edit to `models.py` or `train.py`, the coding agent MUST run:
```bash
pytest test_laws.py -v
```
If any law fails, the code is rejected immediately before spending GPU time.

---

## 2. Gymnasium Benchmark & Dataset (`env_dataset.py`)

Using **Gymnasium** gives you a clean, standardized `reset()` / `step()` API, built-in vectorized collection (`SyncVectorEnv`), and makes it effortless to swap in harder Gymnasium benchmarks (like `MiniGrid-Memory` or `Crafter`) in Phase 2 without rewriting your training or eval code.

### Custom Gymnasium Environment: `KeymapMemoryEnv(gym.Env)`
* **Action Space:** `gym.spaces.Discrete(4)` (actions $\{0, 1, 2, 3\}$).
* **Observation Space:** `gym.spaces.Box(low=-1.0, high=1.0, shape=(16,), dtype=np.float32)`.
* **Episode Length:** Fixed $T = 80$ steps (`truncated = True` at $t = 80$).

#### Environment Mechanics (Per Episode)
1. **Grid:** $5 \times 5$ grid. Agent has position $(x, y) \in \{0..4\}^2$.
2. **Permanent Episode Rule (Hidden Keymap):** At `reset(seed, options={"split": "train" | "val" | "ood"})`, the 4 movement actions $\{0, 1, 2, 3\}$ are assigned a **random permutation** of $\{\text{Up, Down, Left, Right}\}$ drawn strictly from that split's allowed permutations. This permutation stays fixed for the entire episode (tests **Permanent Rule Retention**).
3. **Temporary State (Key-Door Pair):**
   * At $t \in [5, 15]$, the agent encounters a **Key** of color $c \in \{0, 1, 2, 3\}$ and picks it up.
   * After a **distractor delay** of $20$ to $40$ steps, the agent reaches a **Door** matching color $c$.
   * Once the door opens (`reward = +1.0`), the key is consumed and becomes **obsolete** (tests **Learned Eviction / Forgetting**).
4. **Observation Vector ($x_t \in \mathbb{R}^{16}$):**
   * Agent $(x, y)$ normalized to $[-1, 1]$ (2 dims)
   * Current tile type one-hot: `{empty, wall, key, door, distractor}` (5 dims)
   * Tile color one-hot (4 dims)
   * Carrying-key indicator (1 dim — *masked to $0$* starting 1 step after pickup so the model MUST retain the key color in memory!)
   * Door-open event flag (1 dim)
   * Step noise / distractor features (3 dims)
5. **Structured `info` Dict (Returned by `step()` for RQ1–RQ3 Eval):**
   ```python
   info = {
       "perm_id": int,            # Permutation index (0..23)
       "is_novel_probe": bool,    # True on the first time each action 0..3 is tried in this episode
       "is_key_pickup": bool,     # True on the exact step the key is picked up
       "is_door_open": bool,      # True on the exact step the matching door opens
       "is_distractor": bool,     # True on routine movement/noise steps
   }
   ```

### Fast Vectorized Dataset Collection (via Gymnasium)
Use `gymnasium.vector.SyncVectorEnv` (e.g., 32 parallel envs) with a simple exploratory policy (tries all 4 actions early, then random-walks toward key/door) to collect offline datasets in $< 5$ seconds on CPU and save them as `.pt` or `.npz` files:
* **Train Set:** $10,000$ episodes using **permutation IDs `0..17`** (18 of 24 permutations).
* **Validation Set (In-Distribution):** $1,000$ episodes using **permutation IDs `0..17`**.
* **OOD Adaptation Test Set (RQ3):** $1,000$ episodes using **permutation IDs `18..23`** (6 unseen permutations, never seen during training).

---

## 3. Exact Model Architecture (`models.py`)

All models share the exact same observation encoder, prediction heads, and parameter budget ($\approx 400\text{k} - 600\text{k}$ parameters, `d_model = 128`).

### Shared Modules
* `obs_encoder`: $\text{MLP}(16 \to 128 \to 128)$ with LayerNorm + SiLU. Outputs latent $z_t \in \mathbb{R}^{128}$.
* `seq_emb`: $\text{MLP}(128 + 4 \to 128)$ fusing $(z_{t-1}, a_{t-1}) \to M_{\text{cand}} \in \mathbb{R}^{128}$.
* `pred_head`: $\text{MLP}(128 \to 128 \to 16 + 1)$ predicting next observation $\hat{x}_t$ and reward $\hat{r}_t$ from $h_{\text{new}}$.

### Ours: Sparse Memory World Model (`SM_WM`)
**Dimensions:** Batch $B$, Memory Capacity $K = 16$, Embedding Dim $D = 128$.

**State carried across timesteps:**
* Fast continuous state: $h \in \mathbb{R}^{B \times D}$ (initialized to zeros)
* Memory bank: $M \in \mathbb{R}^{B \times K \times D}$ (initialized to zeros)
* Memory occupancy mask: $m \in [0, 1]^{B \times K \times 1}$ (initialized to zeros; $0 = \text{empty}$, $1 = \text{occupied}$)

**Single Step Forward Pass ($t-1 \to t$):**
1. **Candidate Token:**
   $$M_{\text{cand}} = \text{seq\_emb}(\text{concat}(z_{t-1}, a_{t-1})) \quad \in \mathbb{R}^{B \times 1 \times D}$$
2. **Transformer Sequence Construction:**
   Concatenate along sequence dimension (total length $K + 2$):
   $$S = \text{concat}([h_{\text{old}}, M, M_{\text{cand}}]) \quad \in \mathbb{R}^{B \times (K+2) \times D}$$
   *Note:* Do NOT apply sinusoidal positional encodings to $M$ slots (slots are an unordered set); add a learned `type_emb` for `{h_old, memory_slot, cand}`.
3. **Differentiable Attention Masking (Crucial for Gradients):**
   Construct per-token attention bias $b \in \mathbb{R}^{B \times 1 \times 1 \times (K+2)}$:
   * For $h_{\text{old}}$ and $M_{\text{cand}}$: bias $= 0.0$ (always visible).
   * For $M$ slots: bias $= (1.0 - m) \times (-10^4)$.
   Pass $S$ with additive attention bias $b$ through a **2-layer, 4-head Pre-LN Transformer Encoder**.
4. **Extract Next Fast State:**
   $$h_{\text{new}} = \text{Linear}(\text{TransformerOut}[:, -1, :]) \quad \in \mathbb{R}^{B \times D}$$
5. **Parallel Eviction Gate (Check ALL slots simultaneously):**
   Expand $h_{\text{new}}$ to $[B, K, D]$ and compute keep probability for every slot $i \in \{1 \dots K\}$:
   $$k_i = \sigma\left(\text{evict\_gate}(\text{concat}(M_i, h_{\text{new}}))\right) \quad \in [0, 1]^{B \times K \times 1}$$
   Apply Straight-Through Estimator (STE):
   $$k_{\text{ste}} = \mathbb{I}(k > 0.5) - \text{detach}(k) + k$$
   Update existing occupancy mask:
   $$m_{\text{kept}} = m \odot k_{\text{ste}}$$
6. **Write Gate (Decide whether to store $M_{\text{cand}}$):**
   $$w = \sigma\left(\text{write\_gate}(\text{concat}(M_{\text{cand}}, h_{\text{new}}))\right) \quad \in [0, 1]^{B \times 1 \times 1}$$
   Apply STE:
   $$w_{\text{ste}} = \mathbb{I}(w > 0.5) - \text{detach}(w) + w$$
7. **Differentiable Slot Write (No In-Place Operations):**
   Find the lowest-occupancy slot index per batch item:
   $$\text{idx} = \arg\min_{i \in \{1..K\}} (m_{\text{kept}}[:, i, 0])$$
   Convert $\text{idx}$ to one-hot $o \in \{0, 1\}^{B \times K \times 1}$. Compute write mask $u = o \odot w_{\text{ste}}$:
   $$M_{\text{next}} = M \odot (1 - u) + M_{\text{cand}} \odot u$$
   $$m_{\text{next}} = m_{\text{kept}} \odot (1 - u) + 1.0 \odot u$$

### Baselines (For Fair A/B Comparison)
* **Baseline A (`GRU_WM`):** Replaces the Transformer + $M$ with a 2-layer GRU (`hidden_size=256`, matched parameter count). Inputs $M_{\text{cand}}$, outputs $h_{\text{new}}$.
* **Baseline B (`SlidingTF_WM`):** Exact same 2-layer Transformer and `seq_emb` as `SM_WM`, but replaces learned `write_gate`/`evict_gate` with a fixed FIFO sliding window of the last $K=16$ tokens.

---

## 4. Loss Function, Training Pipeline & 2x T4 Strategy (`train.py`)

### Total Loss
For a sequence of length $T = 64$:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{obs}} + \mathcal{L}_{\text{rew}} + \lambda_w \mathcal{L}_{\text{write}} + \lambda_k \mathcal{L}_{\text{keep}}$$

* **Next-Observation Loss:** $\mathcal{L}_{\text{obs}} = \frac{1}{T}\sum_{t=1}^T \text{MSE}(\hat{x}_t, x_t)$
* **Reward Loss:** $\mathcal{L}_{\text{rew}} = \frac{1}{T}\sum_{t=1}^T \text{MSE}(\hat{r}_t, r_t)$
* **Write Sparsity Penalty:** $\mathcal{L}_{\text{write}} = \frac{1}{T}\sum_{t=1}^T w_t$ (use raw soft $w_t$, not STE)
* **Memory Occupancy Penalty:** $\mathcal{L}_{\text{keep}} = \frac{1}{T K}\sum_{t=1}^T \sum_{i=1}^K (m_{t,i} \cdot k_{t,i})$

### Anti-Collapse Curriculum (Critical!)
1. **Gate Bias Init:** Initialize the final linear layer bias of `write_gate` to `+1.0` and `evict_gate` to `+2.0` so the model starts with memory open.
2. **Warmup (Epochs 1–5):** Set $\lambda_w = 0.0, \lambda_k = 0.0$. Let the Transformer learn basic gridworld physics first.
3. **Linear Ramp (Epochs 6–15):** Linearly ramp $\lambda_w \to 0.01$ and $\lambda_k \to 0.005$.
4. **Consolidation (Epochs 16–30):** Hold $\lambda_w, \lambda_k$ constant with Cosine LR decay.

### Kaggle 2x T4 GPU Strategy (Important!)
**DO NOT use `nn.DataParallel` or `DistributedDataParallel (DDP)`** for a 500k-parameter model—synchronizing tiny tensors across PCIe between two T4s is *slower* than running on 1 GPU and invites wrapper bugs.

Instead, use **Process-Level Parallelism** (1 independent experiment per GPU) so your 2x T4s give you a true **2x wall-clock speedup**:
```bash
# Run two models or seeds simultaneously in a Kaggle notebook cell:
CUDA_VISIBLE_DEVICES=0 python train.py --model sm_wm --seed 42 &
CUDA_VISIBLE_DEVICES=1 python train.py --model gru_wm --seed 42 &
wait

CUDA_VISIBLE_DEVICES=0 python train.py --model sliding_tf --seed 42 &
CUDA_VISIBLE_DEVICES=1 python train.py --model sm_wm --seed 43 &
wait
```

### Training Hyperparameters (Per GPU)
* Optimizer: `AdamW(lr=3e-4, weight_decay=1e-4)`
* Gradient Clipping: `max_norm = 1.0`
* Batch Size: `64`, Sequence Length: `T = 64`
* Mixed Precision: `torch.amp.autocast("cuda", dtype=torch.float16)`
* Total Epochs: `30` (~4–6 minutes per run on a single T4; full 9-run suite finishes in ~20–25 min on 2x T4s).

---

## 5. A/B Testing Plans for RQ1, RQ2, and RQ3 (`eval.py`)

Run all three models (`GRU_WM`, `SlidingTF_WM`, `SM_WM`) across **3 random seeds** (`42, 43, 44`). Evaluate directly using `KeymapMemoryEnv` (or its saved Gymnasium rollout trajectories with `info` flags).

### A/B Test 1: RQ1 — Learned Forgetting & Selective Storage
* **Hypothesis:** `SM_WM` will retain *Permanent Rule* and *Active Key* tokens at significantly higher rates than *Distractor* or *Obsolete (Post-Door)* tokens.
* **Metrics (using `info` flags on Validation Set):**
  1. **Write Rate by Event Type:** Average $w_t$ grouped by `info["is_novel_probe"]`, `info["is_key_pickup"]`, and `info["is_distractor"]`.
  2. **Eviction Trigger Rate:** Average drop in $k_i$ for the slot written on `is_key_pickup` immediately after `is_door_open == True`.
  3. **Memory Efficiency:** Mean active slots $\frac{1}{T}\sum m_t$ vs. Next-Obs MSE.
* **Success Criterion:** Write rate on rule/key events $> 2\times$ distractor write rate, and post-door eviction of key token $> 60\%$.

### A/B Test 2: RQ2 — Dreaming (Open-Loop Imagination) Consistency
* **Hypothesis:** During open-loop rollouts (feeding predicted $\hat{x}_t$ back as input instead of real $x_t$), `SM_WM` maintains lower state error than `GRU_WM` and `SlidingTF_WM`.
* **Protocol:**
  1. Prime each model with ground-truth Gymnasium observations for $t = 1 \dots 20$ (covers keymap discovery + key pickup).
  2. Switch to **Dreaming Mode** for $H \in \{10, 25, 40\}$ steps: feed $\hat{x}_{t-1}$ into `obs_encoder` along with planned actions $a_{t-1}$.
* **Metrics:** Open-loop Position MSE and Door-Opening Reward Accuracy at $H = 10, 25, 40$.
* **Success Criterion:** `SM_WM` achieves statistically lower MSE and higher Door-Opening F1 at $H = 40$ than both baselines.

### A/B Test 3: RQ3 — Zero-Gradient Online Adaptation (OOD Keymaps)
* **Hypothesis:** With all weights strictly frozen (`model.eval()` + `torch.no_grad()`), `SM_WM` adapts online inside `KeymapMemoryEnv(options={"split": "ood"})` to the **6 unseen action permutations** faster and retains them longer than baselines.
* **Protocol:**
  1. Roll out live in `SyncVectorEnv` on 1,000 OOD episodes (`perm_id` in `18..23`).
  2. Measure Next-Position Prediction Accuracy in three windows:
     * **Discovery Phase ($t = 1 \dots 8$):** Agent is probing keys.
     * **Short-Term Recall ($t = 9 \dots 25$):** Immediately after probing.
     * **Long-Term Recall ($t = 50 \dots 80$):** Long after probing, past `SlidingTF_WM`'s window of $K=16$.
* **Success Criterion:** `SM_WM` matches or beats baselines in Short-Term Recall and significantly outperforms `SlidingTF_WM` and `GRU_WM` in Long-Term Recall ($t = 50 \dots 80$) with zero gradient updates.

---

## 6. Law-Enforced Verification Suite (`test_laws.py`)

This replaces formal verification tools (like Bend 2) with fast, unbreakable PyTorch + Gymnasium invariants. **Never allow the AI coder to edit this file.**

```python
import pytest
import torch
from gymnasium.utils.env_checker import check_env
from env_dataset import KeymapMemoryEnv
from models import SM_WM, GRU_WM, SlidingTF_WM

B, T, K, D = 4, 30, 16, 128
OBS_DIM, ACT_DIM = 16, 4

def make_dummy_batch():
    obs = torch.randn(B, T, OBS_DIM)
    act = torch.nn.functional.one_hot(torch.randint(0, ACT_DIM, (B, T)), ACT_DIM).float()
    return obs, act

# LAW 1: Architecture Lock (Prevents AI from deleting gates or swapping to a plain GRU)
def test_law_1_architecture_lock():
    model = SM_WM(obs_dim=OBS_DIM, act_dim=ACT_DIM, d_model=D, num_slots=K)
    assert hasattr(model, "write_gate"), "VIOLATION: write_gate missing!"
    assert hasattr(model, "evict_gate"), "VIOLATION: evict_gate missing!"
    assert hasattr(model, "transformer"), "VIOLATION: transformer missing!"
    assert hasattr(model, "seq_emb"), "VIOLATION: seq_emb missing!"

# LAW 2: Gradient Flow Law (Catches broken if/else or detached STE bugs)
def test_law_2_gradient_flow():
    model = SM_WM(obs_dim=OBS_DIM, act_dim=ACT_DIM, d_model=D, num_slots=K)
    obs, act = make_dummy_batch()
    out = model(obs, act)
    loss = out["obs_pred"].pow(2).mean() + 0.01 * out["write_probs"].mean() + 0.01 * out["keep_probs"].mean()
    loss.backward()

    for name in ["write_gate", "evict_gate", "seq_emb", "transformer"]:
        module = getattr(model, name)
        grad_sum = sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None)
        assert grad_sum > 0.0, f"VIOLATION: Zero gradient flowing to {name}!"

# LAW 3: Memory Cap Law (Ensures O(K) bounded memory over long horizons)
def test_law_3_memory_cap():
    model = SM_WM(obs_dim=OBS_DIM, act_dim=ACT_DIM, d_model=D, num_slots=K)
    state = model.init_state(batch_size=B, device="cpu")
    for _ in range(100):  # Step way past K=16
        x = torch.randn(B, OBS_DIM)
        a = torch.zeros(B, ACT_DIM)
        _, state, _ = model.step(x, a, state)
        assert state["M"].shape == (B, K, D), f"VIOLATION: Memory shape mutated to {state['M'].shape}"
        assert state["m"].shape == (B, K, 1), f"VIOLATION: Mask shape mutated to {state['m'].shape}"
        assert (state["m"] >= 0.0).all() and (state["m"] <= 1.0).all(), "VIOLATION: Mask out of [0,1]"

# LAW 4: Strict Causality Law (Model at time t must NOT see future inputs t+1..T)
def test_law_4_causality():
    model = SM_WM(obs_dim=OBS_DIM, act_dim=ACT_DIM, d_model=D, num_slots=K)
    model.eval()
    obs1, act1 = make_dummy_batch()
    obs2, act2 = obs1.clone(), act1.clone()
    obs2[:, 15:, :] += 5.0  # Perturb future only

    with torch.no_grad():
        pred1 = model(obs1, act1)["obs_pred"]
        pred2 = model(obs2, act2)["obs_pred"]

    diff = (pred1[:, :15, :] - pred2[:, :15, :]).abs().max().item()
    assert diff < 1e-5, f"VIOLATION: Future leakage detected! Max diff = {diff}"

# LAW 5: Parameter Budget Parity Law (Ensures fair A/B comparison within 35% params)
def test_law_5_param_parity():
    m_sm = SM_WM(obs_dim=OBS_DIM, act_dim=ACT_DIM, d_model=D, num_slots=K)
    m_gru = GRU_WM(obs_dim=OBS_DIM, act_dim=ACT_DIM, d_model=D)
    m_sl = SlidingTF_WM(obs_dim=OBS_DIM, act_dim=ACT_DIM, d_model=D, window_size=K)

    counts = {
        "SM_WM": sum(p.numel() for p in m_sm.parameters()),
        "GRU_WM": sum(p.numel() for p in m_gru.parameters()),
        "SlidingTF_WM": sum(p.numel() for p in m_sl.parameters()),
    }
    max_p, min_p = max(counts.values()), min(counts.values())
    assert max_p / min_p <= 1.35, f"VIOLATION: Unfair parameter imbalance: {counts}"

# LAW 6: Gymnasium API Compliance & Zero OOD Split Leakage
def test_law_6_gym_and_ood_isolation():
    env = KeymapMemoryEnv()
    check_env(env, skip_render_check=True)

    train_perms = set()
    for i in range(50):
        _, info = env.reset(seed=i, options={"split": "train"})
        train_perms.add(info["perm_id"])

    ood_perms = set()
    for i in range(50):
        _, info = env.reset(seed=1000 + i, options={"split": "ood"})
        ood_perms.add(info["perm_id"])

    assert train_perms.isdisjoint(ood_perms), f"VIOLATION: OOD leakage! Overlap: {train_perms & ood_perms}"
```

---

## 7. Copy-Paste Prompt Contract for Your Cheap AI Coding Model

Give this exact prompt alongside `sm_wm_blueprint.md` when generating code:

> **System Instructions for Implementation:**  
> You are implementing `sm_wm_blueprint.md` strictly as specified.  
> 1. Implement one file at a time in this order: `config.py` -> `env_dataset.py` -> `models.py` -> `test_laws.py` -> `train.py` -> `eval.py`.  
> 2. **NEVER** modify `test_laws.py` or `eval.py` once created.  
> 3. Use standard `gymnasium` (`gym.Env`, `gym.vector.SyncVectorEnv`) for `env_dataset.py` and `eval.py`.  
> 4. **NEVER** replace `write_gate`, `evict_gate`, or the STE slot-update math with Python `if/else` branches, `.append()`, or a plain GRU.  
> 5. Every tensor operation in `SM_WM.step()` must be out-of-place (no `M[:, idx] = ...`) to avoid PyTorch autograd version errors.  
> 6. **DO NOT** use `nn.DataParallel` or `DDP`. Support single-GPU execution via `--model` and `--seed` CLI args so Kaggle's 2x T4s can run two independent jobs concurrently with `CUDA_VISIBLE_DEVICES=0` and `CUDA_VISIBLE_DEVICES=1`.  
> 7. After implementing or editing `env_dataset.py` or `models.py`, run `pytest test_laws.py -v` and fix any failures before moving on.