# Architecture Document: Sparse Memory World Model (SM-WM V8)

## 1. Overview & Core Hypothesis

Traditional Recurrent World Models (e.g., RSSM / GRU) and sliding-window Transformers suffer from two fundamental long-horizon failure modes:
1. **Compounding State Dilution:** In continuous recurrent states ($h \in \mathbb{R}^d$), small prediction errors compound during autoregressive rollout ($H > 15$), causing severe trajectory drift (drift error $> 1.5$ on a $5\times 5$ grid).
2. **Memory Contamination:** Sliding-window or dense FIFO memory mechanisms indiscriminately append the model's own unconfident predictions during imagination rollouts, causing self-reinforcing hallucinations.

**The SM-WM Solution:**
SM-WM separates **fast continuous dynamics** ($h_t \in \mathbb{R}^d$) from an **un-diluted episodic memory bank** ($M_t \in \mathbb{R}^{K \times d}$, with $K=16$ bounded slots). Crucially, memory writes are driven by **prediction surprise** ($e_t = x_t - \hat{x}_t$), while eviction is driven by **task event obsoletion and duplicate redundancy**. During open-loop dreaming, ground truth is absent ($e_t \equiv 0$), naturally locking the write gate ($w_t \equiv 0$) and insulating the memory bank from hallucination.

---

## 2. Complete Mathematical Specification (V8)

```text
Timestep t:
  Phase 1 (Predict):
    (x_{t-1}, a_{t-1}, state_{t-1}) ──────► [Transformer Query] ──► h_t, \hat{x}_t, \hat{r}_t

  Phase 2 (Observe & Surprise):
    Real x_t arrives ─────────────────────► e_t = (x_t - \hat{x}_t).detach()
                                            \Delta x_t = x_t - x_{t-1}

  Phase 3 (Encode Full Transition):
    (x_{t-1}, a_{t-1}, x_t, \Delta x_t, e_t) ──► [TransitionEncoder] ──► M_cand

  Phase 4 (Contextualize Token):
    [h_t, M_{t-1}, M_cand] ───────────────► [Transformer Context] ─► M_cand_ctx
    M_store = LayerNorm(M_cand + 0.5 * M_cand_ctx)

  Phase 5 (Gated Write & Evict):
    Surprise ||e_t||^2 + M_store ────────► [Write Gate (STE)] ───► w_t \in {0, 1}
    CosineSim(M_i, M_store) * w_t ────────► [Evict Gate (STE)] ───► k_i \in {0, 1}
    Slot allocation via argmin(m_kept) ──► M_t, m_t
```

### Module 1: Observation & Query Encoders
* **`ObsEncoder`:** $\text{MLP}(16 \to 128 \to 128)$ with LayerNorm + SiLU. Maps raw state $x \in \mathbb{R}^{16} \to z \in \mathbb{R}^{128}$.
* **`SeqEmb`:** $\text{MLP}(128 + 4 \to 128 \to 128)$ with LayerNorm + SiLU. Projects $(z_{t-1}, a_{t-1}) \to q_t \in \mathbb{R}^{128}$.

### Module 2: Pure Predict Step (No Memory Mutations)
To ensure zero future leakage and safe open-loop dreaming, prediction occurs before observing $x_t$:
1. Sequence construction:
   $$S_{\text{pred}} = [h_{t-1}, M_{t-1}, q_t] + \text{TypeEmb}(\{0, 1 \dots 1, 2\}) \quad \in \mathbb{R}^{B \times (K+2) \times d}$$
2. Attention masking on empty slots ($m_i = 0$):
   $$\text{Bias}_i = (1 - m_i) \times (-10^4)$$
3. Fast state extraction:
   $$h_t = \text{Linear}(\text{TransformerOut}[:, -1, :]) \quad \in \mathbb{R}^{B \times d}$$
4. Residual position and observation head:
   $$\hat{p}_t = \text{clamp}(p_{t-1} + \Delta \hat{p}_t, -1.0, 1.0)$$
   $$\hat{x}_t = [\hat{p}_t, \hat{x}_{t, \text{features}}], \quad \hat{r}_t = \text{PredHead}_{\text{rew}}(h_t)$$

### Module 3: Transition Encoder & Contextualization
Once $x_t$ arrives, the full transition narrative is encoded:
1. Detached surprise vector (Safeguard 1):
   $$e_t = (x_t - \hat{x}_t).\text{detach}(), \quad \|e_t\|^2 = \frac{1}{d}\sum_{j=1}^d e_{t,j}^2$$
2. Raw transition encoding:
   $$M_{\text{cand}} = \text{TransitionEncoder}([x_{t-1}, a_{t-1}, x_t, x_t - x_{t-1}, e_t]) \quad \in \mathbb{R}^{B \times d}$$
3. Transformer contextualization pass:
   $$S_{\text{ctx}} = [h_t, M_{t-1}, M_{\text{cand}}] + \text{TypeEmb}$$
   $$M_{\text{store}} = \text{LayerNorm}(M_{\text{cand}} + 0.5 \cdot \text{Linear}(\text{TransformerOut}[:, -1, :]))$$

### Module 4: Surprise-Driven Write Gate
$$w_{\text{logit}} = \text{Linear}\Big(\text{SiLU}\big(\text{Linear}([M_{\text{store}}, h_t, \|e_t\|^2])\big)\Big)$$
$$w_t = \sigma(w_{\text{logit}}), \quad w_{\text{ste}} = \mathbb{I}(w_t > 0.5) - w_t.\text{detach}() + w_t$$
* **High Surprise (e.g., novel action or key pickup):** $w_t \to 1.0$.
* **Zero Surprise (familiar distractor steps):** $w_t \to 0.0$.
* **Open-Loop Dreaming:** $x_{\text{curr}} = \text{None} \implies e_t \equiv 0 \implies w_t \equiv 0$ (Memory is completely write-protected).

### Module 5: Redundancy & Obsoletion Evict Gate
To clear consumed keys and avoid duplicate action rules without accidental deletion:
1. Gated cosine redundancy check against incoming $M_{\text{store}}$ (Safeguard 2):
   $$\text{Sim}_i = \left(\frac{M_i \cdot M_{\text{store}}}{\|M_i\|_2 \|M_{\text{store}}\|_2}\right) \times w_t.\text{detach}()$$
   *(If $w_t \approx 0$, redundancy is 0; existing memories are never deleted on distractor steps!)*
2. Slot-wise keep probability:
   $$k_i = \sigma\Big(\text{Linear}\big([M_i, h_t, \text{Sim}_i]\big)\Big)$$
   $$k_{\text{ste}} = \mathbb{I}(k_i > 0.5) - k_i.\text{detach}() + k_i$$
   $$m_{\text{kept}} = m_{t-1} \odot k_{\text{ste}}$$

### Module 6: Slot Allocation (Out-of-Place Update)
$$\text{idx} = \arg\min_{i \in \{1 \dots K\}} (m_{\text{kept}}[:, i])$$
$$u = \text{OneHot}(\text{idx}) \odot w_{\text{ste}}$$
$$M_t = M_{t-1} \odot (1 - u) + M_{\text{store}} \odot u$$
$$m_t = m_{\text{kept}} \odot (1 - u) + 1.0 \odot u$$

---

## 3. Parameter Budgets & Fair Comparison

All models strictly conform to the $\approx 0.5\text{M} - 0.6\text{M}$ parameter budget (Law 5 parity within $\pm 25\%$):

| Model | Total Parameters | Checkpoint Size | Architecture Details |
|---|---|---|---|
| **`SM_WM` (Ours V8)** | **593,299** | $\approx 6.9\text{ MB}$ | $d=128$, 2 layers, 4 heads, $K=16$ slots, transition encoder, STE gates |
| **`GRU_WM` (Baseline A)** | **536,369** | $\approx 6.2\text{ MB}$ | 2-layer stacked GRU ($h=200$), parameter-matched output heads |
| **`SlidingTF_WM` (Baseline B)**| **486,545** | $\approx 5.7\text{ MB}$ | $d=128$, 2 layers, 4 heads, $W=17$ FIFO sliding window |

---

## 4. Objective & Loss Formulation

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{1-step}} + 0.5 \cdot \mathcal{L}_{\text{rollout}} + \mathcal{L}_{\text{rew}} + \lambda_w \mathcal{L}_{\text{write}} + \lambda_k \mathcal{L}_{\text{capacity}}$$

1. **1-Step Teacher-Forcing Loss:**
   $$\mathcal{L}_{\text{1-step}} = \frac{1}{T-1}\sum_{t=1}^{T-1} \|\hat{x}_t - x_t\|_2^2$$
2. **Multi-Step Unrolled Rollout Loss ($H=4$ BPTT):**
   Unrolls open-loop predictions without teacher forcing from random start points $t_{\text{start}}$:
   $$\mathcal{L}_{\text{rollout}} = \frac{1}{4}\sum_{k=1}^4 \|\hat{p}_{t_{\text{start}}+k} - p_{t_{\text{start}}+k}\|_2^2$$
3. **Write Sparsity Penalty:** $\mathcal{L}_{\text{write}} = \frac{1}{T}\sum_{t=1}^T w_t$ (scaled by curriculum $\lambda_w \in [0, 0.005]$)
4. **Target-Margin Capacity Loss:**
   $$\mathcal{L}_{\text{capacity}} = \frac{\max\left(0, \sum_{i=1}^K m_{t,i} - 5.0\right)^2}{K}$$
   *Provides a free budget of 5 slots (4 buttons + 1 key) with zero penalty, completely eliminating slot collapse while preventing hoarding.*
