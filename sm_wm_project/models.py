"""Models for SM-WM (V8 Surprise-Driven Transition Memory), GRU baseline, and Sliding-TF baseline."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, use_ln=True, act=nn.SiLU):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim)]
        if use_ln:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(act())
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ObsEncoder(nn.Module):
    def __init__(self, obs_dim=16, d_model=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x):
        return self.net(x)


class SeqEmb(nn.Module):
    """Query embedding: projects (z_{t-1}, a_{t-1}) -> query vector in R^D."""
    def __init__(self, d_model=128, act_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model + act_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, z_prev, a_prev):
        return self.net(torch.cat([z_prev, a_prev], dim=-1))


class TransitionEncoder(nn.Module):
    """
    V8 Transition Encoder: encodes the complete story of the transition
    (x_{t-1}, a_{t-1}, x_t, Delta x_t, surprise e_t) -> M_cand in R^D.
    Input dims: obs_dim (16) + act_dim (4) + obs_dim (16) + obs_dim (16) + obs_dim (16) = 68 dims
    """
    def __init__(self, obs_dim=16, act_dim=4, d_model=128):
        super().__init__()
        in_dim = (obs_dim * 4) + act_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x_prev, a_prev, x_curr, delta_x, surprise):
        feat = torch.cat([x_prev, a_prev, x_curr, delta_x, surprise], dim=-1)
        return self.net(feat)


class PredHead(nn.Module):
    def __init__(self, d_model=128, obs_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, obs_dim + 1),  # obs_pred (16) + rew_pred (1)
        )
        self.obs_dim = obs_dim

    def forward(self, h):
        out = self.net(h)
        obs_pred = out[..., :self.obs_dim]
        rew_pred = out[..., self.obs_dim:self.obs_dim + 1]
        return obs_pred, rew_pred


class SM_WM(nn.Module):
    """
    V8 Sparse Memory World Model (SM-WM) with:
    - Phase 1: Prediction pass [h_old, M, Query] -> h_pred -> (\hat{x}_t, \hat{r}_t)
    - Phase 2: Observe outcome x_t -> surprise e_t = ||x_t - \hat{x}_t||_2^2 (detached)
    - Phase 3: TransitionEncoder(x_{t-1}, a_{t-1}, x_t, Delta x_t, e_t) -> M_cand
    - Phase 4: Contextual token blend M_store = M_cand + M_cand_ctx
    - Phase 5: Surprise-driven Write Gate & Duplicate-Gated Eviction Gate
    """
    def __init__(
        self,
        obs_dim=16,
        act_dim=4,
        d_model=128,
        num_slots=16,
        nhead=4,
        num_layers=2,
        write_bias_init=0.5,
        evict_bias_init=0.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.d_model = d_model
        self.num_slots = num_slots

        self.obs_encoder = ObsEncoder(obs_dim, d_model)
        self.seq_emb = SeqEmb(d_model, act_dim)
        self.trans_encoder = TransitionEncoder(obs_dim, act_dim, d_model)

        # Type embeddings: 0: h, 1: M slots, 2: cand/query
        self.type_emb = nn.Embedding(3, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.h_out_proj = nn.Linear(d_model, d_model)
        self.cand_ctx_proj = nn.Linear(d_model, d_model)

        # V8 Gates
        # Write gate: decides whether transition is novel based on (M_cand, h_new, surprise_mag)
        self.write_gate = nn.Sequential(
            nn.Linear(d_model * 2 + 1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )
        
        # Evict gate: checks each slot M_i for obsoletion and redundancy
        # Input: [M_i, h_new, similarity(M_i, M_store) * w_t] -> dim: 2D + 1
        self.evict_gate = nn.Sequential(
            nn.Linear(d_model * 2 + 1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )

        nn.init.constant_(self.write_gate[-1].bias, write_bias_init)
        nn.init.constant_(self.evict_gate[-1].bias, evict_bias_init)

        self.pred_head = PredHead(d_model, obs_dim)

    def init_state(self, batch_size, device):
        return {
            "h": torch.zeros(batch_size, self.d_model, device=device),
            "M": torch.zeros(batch_size, self.num_slots, self.d_model, device=device),
            "m": torch.zeros(batch_size, self.num_slots, 1, device=device),
        }

    def _build_attn_mask(self, m, B, device):
        K = self.num_slots
        nhead = self.transformer.layers[0].self_attn.num_heads
        # Detach m for attention routing so recursive softmax backward doesn't explode
        slot_bias = (1.0 - m.squeeze(-1).detach()) * (-10000.0)  # [B, K]
        h_bias = torch.zeros(B, 1, device=device)
        cand_bias = torch.zeros(B, 1, device=device)
        key_bias = torch.cat([h_bias, slot_bias, cand_bias], dim=1)  # [B, K+2]
        key_bias_expanded = key_bias.unsqueeze(1).unsqueeze(1).repeat(1, nhead, K + 2, 1)
        return key_bias_expanded.view(B * nhead, K + 2, K + 2)

    def predict_step(self, x_prev, a_prev, state):
        """
        Safeguard 3: Pure Predict step.
        Queries memory M with (x_{t-1}, a_{t-1}) BEFORE x_t arrives.
        Used for 1-step prediction and open-loop dreaming (NO memory writes).
        """
        B = x_prev.shape[0]
        K = self.num_slots
        h_old = state["h"]
        M = state["M"]
        m = state["m"]

        z_prev = self.obs_encoder(x_prev)
        q_emb = self.seq_emb(z_prev, a_prev).unsqueeze(1)  # [B, 1, D]

        S = torch.cat([h_old.unsqueeze(1), M, q_emb], dim=1)  # [B, K+2, D]
        type_ids = torch.tensor([0] + [1] * K + [2], device=x_prev.device, dtype=torch.long)
        S = S + self.type_emb(type_ids).unsqueeze(0)

        attn_mask = self._build_attn_mask(m, B, x_prev.device)
        trans_out = self.transformer(S, mask=attn_mask)
        h_new = self.h_out_proj(trans_out[:, -1, :])  # [B, D]

        obs_pred, rew_pred = self.pred_head(h_new)
        next_pos = torch.clamp(x_prev[..., :2] + obs_pred[..., :2], -1.0, 1.0)
        full_obs_pred = torch.cat([next_pos, obs_pred[..., 2:]], dim=-1)

        new_state = {
            "h": h_new,
            "M": M,
            "m": m,
        }
        return (full_obs_pred, rew_pred), new_state

    def step(self, x_prev, a_prev, state, x_curr=None):
        """
        Full V8 Transition Step:
        1. Predict \hat{x}_t using memory
        2. If x_curr is provided, observe real outcome, compute surprise, and update memory
        3. If x_curr is None (dreaming), perform pure predict_step with zero memory updates.
        """
        (full_obs_pred, rew_pred), pred_state = self.predict_step(x_prev, a_prev, state)

        # Open-loop dreaming safeguard: no ground-truth observation available
        if x_curr is None:
            B = x_prev.shape[0]
            K = self.num_slots
            aux = {
                "write_prob": torch.zeros(B, 1, device=x_prev.device),
                "keep_probs": torch.ones(B, K, 1, device=x_prev.device),
                "write_mask": torch.zeros(B, K, 1, device=x_prev.device),
                "m_kept": pred_state["m"],
                "surprise": torch.zeros(B, 1, device=x_prev.device),
            }
            return (full_obs_pred, rew_pred), pred_state, aux

        B = x_prev.shape[0]
        K = self.num_slots
        D = self.d_model
        h_new = pred_state["h"]
        M = pred_state["M"]
        m = pred_state["m"]

        # Safeguard 1: Detach surprise so prediction gradients don't fight with the write gate
        delta_x = x_curr - x_prev
        surprise_vec = (x_curr - full_obs_pred).detach()
        surprise_mag = surprise_vec.pow(2).mean(dim=-1, keepdim=True)  # [B, 1]

        # 3. Transition candidate token
        m_cand_raw = self.trans_encoder(x_prev, a_prev, x_curr, delta_x, surprise_vec)  # [B, D]

        # 4. Contextual sequence pass to get contextualized token M_cand_ctx AND post-observation state h_post
        S = torch.cat([h_new.unsqueeze(1), M, m_cand_raw.unsqueeze(1)], dim=1)
        type_ids = torch.tensor([0] + [1] * K + [2], device=x_prev.device, dtype=torch.long)
        S = S + self.type_emb(type_ids).unsqueeze(0)
        attn_mask = self._build_attn_mask(m, B, x_prev.device)
        trans_out = self.transformer(S, mask=attn_mask)
        
        # h_post is index 0 (updated fast state knowing true outcome x_curr), cand_ctx is index -1
        h_post = self.h_out_proj(trans_out[:, 0, :])
        m_cand_ctx = self.cand_ctx_proj(trans_out[:, -1, :])  # [B, D]
        # Residual blend of pure transition + contextualized embedding
        M_store = (m_cand_raw + 0.5 * m_cand_ctx).unsqueeze(1)  # [B, 1, D]

        # 5. Write Gate: powered by surprise magnitude
        write_in = torch.cat([M_store.squeeze(1), h_post, surprise_mag], dim=-1)  # [B, 2D + 1]
        w_logits = self.write_gate(write_in)
        w = torch.sigmoid(w_logits).unsqueeze(-1)  # [B, 1, 1]
        w_ste = (w > 0.5).float() - w.detach() + w

        # Safeguard 2: Duplicate redundancy check gated by write probability w
        # cosine similarity between existing slots M and incoming candidate M_store
        M_norm = F.normalize(M, p=2, dim=-1)            # [B, K, D]
        store_norm = F.normalize(M_store, p=2, dim=-1)  # [B, 1, D]
        cos_sim = torch.sum(M_norm * store_norm, dim=-1, keepdim=True)  # [B, K, 1] in [-1, 1]
        gated_sim = cos_sim * w.detach()  # Gated by write probability: 0 if not writing!

        # 6. Evict Gate: checks slots for obsoletion (via h_post seeing real x_curr) and redundancy (via gated_sim)
        h_post_exp = h_post.unsqueeze(1).expand(B, K, D)
        evict_in = torch.cat([M, h_post_exp, gated_sim], dim=-1)  # [B, K, 2D + 1]
        k_logits = self.evict_gate(evict_in)
        k = torch.sigmoid(k_logits)
        k_ste = (k > 0.5).float() - k.detach() + k
        m_kept = m * k_ste

        # 7. Slot allocation (least occupied slot)
        occ_scores = m_kept.squeeze(-1)  # [B, K]
        idx = torch.argmin(occ_scores, dim=-1)  # [B]
        o = F.one_hot(idx, num_classes=K).float().unsqueeze(-1)  # [B, K, 1]
        u = o * w_ste  # [B, K, 1]

        # Out-of-place update
        M_next = M * (1.0 - u) + M_store * u
        m_next = m_kept * (1.0 - u) + 1.0 * u

        next_state = {
            "h": h_post,
            "M": M_next,
            "m": m_next,
        }
        aux = {
            "write_prob": w.squeeze(-1),
            "keep_probs": k,
            "write_mask": u,
            "m_kept": m_kept,
            "surprise": surprise_mag,
        }
        return (full_obs_pred, rew_pred), next_state, aux

    def forward(self, obs, act):
        """
        Full sequence forward pass.
        obs: [B, T, obs_dim]
        act: [B, T, act_dim]
        """
        B, T, _ = obs.shape
        device = obs.device
        state = self.init_state(B, device)

        obs_preds = []
        rew_preds = []
        write_probs = []
        keep_probs = []
        m_history = []

        for t in range(T):
            x_t = obs[:, t]
            a_t = act[:, t]
            # Next real observation for transition memory
            x_next = obs[:, t + 1] if t + 1 < T else None
            (o_pred, r_pred), state, aux = self.step(x_t, a_t, state, x_curr=x_next)
            obs_preds.append(o_pred)
            rew_preds.append(r_pred)
            write_probs.append(aux["write_prob"])
            keep_probs.append(aux["keep_probs"])
            m_history.append(state["m"])

        return {
            "obs_pred": torch.stack(obs_preds, dim=1),
            "rew_pred": torch.stack(rew_preds, dim=1),
            "write_probs": torch.stack(write_probs, dim=1),
            "keep_probs": torch.stack(keep_probs, dim=1),
            "m_history": torch.stack(m_history, dim=1),
        }


class GRU_WM(nn.Module):
    """
    Baseline A: 2-layer GRU with parameter count matched to SM_WM.
    """
    def __init__(self, obs_dim=16, act_dim=4, d_model=128, hidden_size=200):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.d_model = d_model
        self.hidden_size = hidden_size

        self.obs_encoder = ObsEncoder(obs_dim, d_model)
        self.seq_emb = SeqEmb(d_model, act_dim)
        self.gru = nn.GRU(d_model, hidden_size, num_layers=2, batch_first=True)
        self.h_out_proj = nn.Linear(hidden_size, d_model)
        self.pred_head = PredHead(d_model, obs_dim)

    def init_state(self, batch_size, device):
        return {
            "h": torch.zeros(2, batch_size, self.hidden_size, device=device)
        }

    def step(self, x_prev, a_prev, state, x_curr=None):
        z_prev = self.obs_encoder(x_prev)
        m_cand = self.seq_emb(z_prev, a_prev).unsqueeze(1)
        out, h_next = self.gru(m_cand, state["h"])
        h_feat = self.h_out_proj(out.squeeze(1))
        obs_pred, rew_pred = self.pred_head(h_feat)
        next_pos = torch.clamp(x_prev[..., :2] + obs_pred[..., :2], -1.0, 1.0)
        full_obs_pred = torch.cat([next_pos, obs_pred[..., 2:]], dim=-1)
        return (full_obs_pred, rew_pred), {"h": h_next}, {}

    def forward(self, obs, act):
        B, T, _ = obs.shape
        device = obs.device
        state = self.init_state(B, device)

        obs_preds = []
        rew_preds = []
        for t in range(T):
            x_t = obs[:, t]
            a_t = act[:, t]
            (o_pred, r_pred), state, _ = self.step(x_t, a_t, state)
            obs_preds.append(o_pred)
            rew_preds.append(r_pred)

        return {
            "obs_pred": torch.stack(obs_preds, dim=1),
            "rew_pred": torch.stack(rew_preds, dim=1),
        }


class SlidingTF_WM(nn.Module):
    """
    Baseline B: 2-layer Pre-LN Transformer with a fixed FIFO sliding window of K=16 tokens.
    """
    def __init__(self, obs_dim=16, act_dim=4, d_model=128, window_size=16, nhead=4, num_layers=2):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.d_model = d_model
        self.window_size = window_size

        self.obs_encoder = ObsEncoder(obs_dim, d_model)
        self.seq_emb = SeqEmb(d_model, act_dim)

        self.pos_emb = nn.Parameter(torch.randn(1, window_size, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.h_out_proj = nn.Linear(d_model, d_model)
        self.pred_head = PredHead(d_model, obs_dim)

    def init_state(self, batch_size, device):
        return {
            "buf": torch.zeros(batch_size, 0, self.d_model, device=device)
        }

    def step(self, x_prev, a_prev, state, x_curr=None):
        B = x_prev.shape[0]
        z_prev = self.obs_encoder(x_prev)
        m_cand = self.seq_emb(z_prev, a_prev).unsqueeze(1)

        buf = state["buf"]
        if buf.shape[1] < self.window_size:
            buf = torch.cat([buf, m_cand], dim=1)
        else:
            buf = torch.cat([buf[:, 1:], m_cand], dim=1)

        L = buf.shape[1]
        seq = buf + self.pos_emb[:, :L, :]
        trans_out = self.transformer(seq)
        h_feat = self.h_out_proj(trans_out[:, -1, :])
        obs_pred, rew_pred = self.pred_head(h_feat)
        next_pos = torch.clamp(x_prev[..., :2] + obs_pred[..., :2], -1.0, 1.0)
        full_obs_pred = torch.cat([next_pos, obs_pred[..., 2:]], dim=-1)

        return (full_obs_pred, rew_pred), {"buf": buf}, {}

    def forward(self, obs, act):
        B, T, _ = obs.shape
        device = obs.device
        state = self.init_state(B, device)

        obs_preds = []
        rew_preds = []
        for t in range(T):
            x_t = obs[:, t]
            a_t = act[:, t]
            (o_pred, r_pred), state, _ = self.step(x_t, a_t, state)
            obs_preds.append(o_pred)
            rew_preds.append(r_pred)

        return {
            "obs_pred": torch.stack(obs_preds, dim=1),
            "rew_pred": torch.stack(rew_preds, dim=1),
        }
