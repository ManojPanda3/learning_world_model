You do **not** need heavy 3D or pixel-based benchmarks (like 3D Memory Maze or high-res video) to publish a strong paper. With **Kaggle's free 2x T4 GPUs** (~30 hours/week, 12-hour max per run) and a **Ryzen 3250U**, the winning strategy is to stay strictly in **vector-observation (state-based) environments** with **compact ~0.8M-parameter models**.

In state-based world models, you can run a full 5-seed experiment in **15–30 minutes** on a T4—meaning your entire publishable paper fits inside **one single week of free Kaggle quota**.

---

## 1. Hardware Strategy: How to Stretch Your Compute

* **Local Ryzen 3250U (CPU only):** Use this strictly for writing code, unit-testing shapes, 30-second smoke tests (100 steps), analyzing saved slot logs, and generating Matplotlib/LaTeX figures. Never run training locally.
* **Kaggle 2x T4 Trick (2x Throughput for 1x Quota):** Kaggle counts wall-clock time as 1 quota hour even when both T4s are active. Do **not** use Distributed Data Parallel (DDP) for small models—instead, launch **2 independent seeds simultaneously** (`CUDA_VISIBLE_DEVICES=0` for Seed 1, `CUDA_VISIBLE_DEVICES=1` for Seed 2).
* **Use `Save & Run All (Commit)`:** Interactive sessions die after 20–60 minutes of inactivity. Always run training via background commits and enable `torch.cuda.amp.autocast(dtype=torch.float16)` to leverage T4 Tensor Cores.
* **Bonus Free Compute:** Kaggle also provides ~20 hours/week of free **TPU v5e-8**, which is ideal if you later want to run JAX-based POPGym Arcade.

---

## 2. Exact Model Parameters (~0.8M Parameter Budget)

To guarantee fair, reviewer-proof comparisons, lock **every model** to roughly **0.7M – 0.9M trainable parameters** and train all world models with identical **$H=4$ unrolled rollout loss**.

| Component | `SM_WM` (V8 + Recency Buffer) | Matched Baselines |
| --- | --- | --- |
| **Latent / Model Dim ($d$)** | `128` | `GRU`: 2-layer, `hidden=256` (~0.8M) |
| **Backbone** | 2-layer Transformer (`4 heads`, `FFN=256`) | `SlidingTF`: 2-layer, `d=128`, 16-step window (~0.8M) |
| **Working Memory Bank** | `16 slots` × `128d` (Full Transition Tokens) | `FullTF`: Same as SlidingTF, full episode context (~0.8M) |
| **Local Recency Buffer** | `4 steps` × `128d` (fixes Scrambled Controller gap) | `SSM / Mamba`: 2-layer S4D/Mamba, `d=128` (~0.8M) |
| **Actor & Critic Heads** | 2-layer MLP (`128 -> 128 -> | A |
| **Training Hyperparams** | AdamW (`lr=3e-4`), batch `64`, seq len `64`, `fp16` | Identical optimizer, batch size, and $H=4$ unroll |

---

## 3. Datasets & Environments (Zero Pixel Overhead)

In world-model research, you use **two types of data**: fast **offline transition datasets** (for cheap, controlled world-model ablations) and **online environments** (for closed-loop RL and mistake recovery).

1. **Suite A — Mechanistic Probes (Offline + Online, Custom Vector Envs):**
* **Data:** Collect 200k transitions per task using a mixed random + heuristic policy (takes under 1 minute on CPU).
* **Tasks:** (1) **Key-Door** with scaling delays $L \in \{16, 32, 64, 128\}$; (2) **Scrambled Controller** (mid-episode action flip); (3) **Multi-Stage Key-Door** (3 sequential keys/doors in one episode to prove event eviction).


2. **Suite B — Public Memory Benchmark (Classic Vector POPGym):**
* Use the standard **NumPy/Gymnasium POPGym** (`pip install popgym`), *not* the heavy pixel version.
* **Tasks (3 core tasks):** `RepeatPrevious-Medium`, `Autoencode-Easy`, and `Concentration-Medium` (500k–1M env steps each).


3. **Suite C — Non-Stationary Mistake-Recovery (Closed-Loop Online):**
* **Tasks (2 tasks):** (1) **Dynamic Hazard Gridworld** (mid-episode unseen wall/trap injected on the optimal path); (2) **Control-Flip Navigation** (controls permute at $t_{\text{shift}} \in [20, 40]$).
* **Execution:** Evaluate with **frozen weights** over 100 episodes using your **Surprise-Triggered $H=4$ MPC** (discrete beam search, beam width 8–16—takes <2 ms/step on a T4).



---

## 4. Complete Evaluation Plan & 1-Week Kaggle Budget

You need **5 seeds** per model for every main table. Running 2 seeds in parallel on Kaggle's 2x T4s, the entire paper takes **~24 GPU hours** (under your 30-hour weekly limit):

| Phase | Exact Runs to Execute | Paper Deliverable | Est. 2x T4 Time |
| --- | --- | --- | --- |
| **Phase 1: Parity & Horizon Cliff** | Train `SM_WM`, `GRU`, `SlidingTF`, `FullTF` on Suite A ($L = 16, 32, 64, 128$), 5 seeds each. | **Figure 1:** Accuracy & ms/step latency vs. horizon $L$ (shows SlidingTF cliff at $L>16$ and $O(1)$ speed). | **~3.5 hrs** |
| **Phase 2: V8 Component Ablations** | Run 5 ablations of `SM_WM` on Suite A (remove: Full Transition Tokens, Surprise Gate, Event Eviction, Target-Margin Budget, $H=4$ unroll), 5 seeds each. | **Table 1:** Ablation table proving every V8 design choice is strictly necessary. | **~4.0 hrs** |
| **Phase 3: Closed-Loop Mistake Recovery** | Train Actor-Critic on Suite C, then evaluate **frozen weights** with Surprise-Triggered $H=4$ MPC vs. `GRU`, `SlidingTF`, `Heuristic 16-Slot`, `SSM`, 5 seeds each. | **Table 2 & Figure 2:** Repeat-Error Rate (RER), Steps-to-Recovery ($T_{\text{rec}}$), and slot-write/eviction trace plots. | **~5.5 hrs** |
| **Phase 4: Public POPGym Benchmark** | Train `SM_WM`, `GRU`, `SlidingTF`, `SSM`, `Heuristic 16-Slot` on the 3 vector POPGym tasks (500k–1M steps), 5 seeds each. | **Table 3:** Standardized POPGym returns (Median + IQM with 95% CI) against public baselines. | **~9.5 hrs** |
| **Phase 5: Dreaming & Diagnostics** | Run $H=40$ open-loop imagination rollouts + movement variance check (anti-Frozen-Goalie proof) + latency profiling ($T=16 \to 1024$). | **Table 4:** $H=40$ MSE, trajectory variance, active slot counts, and wall-clock ms/step. | **~1.5 hrs** |
| **Total** | **Full publishable evaluation suite** | **4 Tables + 2–3 Core Figures** | **~24.0 hrs** |

### What to Run First Tonight

Start with **Phase 1 and Phase 2** (~7.5 hours total). They use your existing Key-Door / Scrambled Controller codebase, require zero new environment wrappers, and will immediately confirm whether adding the **4-step recency buffer** closes the Scrambled Controller gap against SlidingTF before you spend a single GPU hour on POPGym.
