"""Models for SM-WM, GRU baseline, and Sliding-TF baseline."""
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
    Sparse Memory World Model (SM-WM).
    Separates fast continuous state h from un-diluted memory bank M.
    """
    def __init__(
        self,
        obs_dim=16,
        act_dim=4,
        d_model=128,
        num_slots=16,
        nhead=4,
        num_layers=2,
        write_bias_init=1.0,
        evict_bias_init=2.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.d_model = d_model
        self.num_slots = num_slots

        self.obs_encoder = ObsEncoder(obs_dim, d_model)
        self.seq_emb = SeqEmb(d_model, act_dim)

        # Learned type embeddings for: 0: h_old, 1: memory_slot, 2: cand
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

        # Gates
        self.evict_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )
        self.write_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )
        # Initialize gate bias to avoid collapse
        nn.init.constant_(self.write_gate[-1].bias, write_bias_init)
        nn.init.constant_(self.evict_gate[-1].bias, evict_bias_init)

        self.pred_head = PredHead(d_model, obs_dim)

    def init_state(self, batch_size, device):
        return {
            "h": torch.zeros(batch_size, self.d_model, device=device),
            "M": torch.zeros(batch_size, self.num_slots, self.d_model, device=device),
            "m": torch.zeros(batch_size, self.num_slots, 1, device=device),
        }

    def step(self, x_prev, a_prev, state):
        """
        Single step forward pass:
        x_prev: [B, obs_dim]
        a_prev: [B, act_dim] (one-hot or continuous)
        state: dict with 'h' [B, D], 'M' [B, K, D], 'm' [B, K, 1]
        """
        B = x_prev.shape[0]
        K = self.num_slots
        D = self.d_model
        h_old = state["h"]
        M = state["M"]
        m = state["m"]

        # 1. Candidate token
        z_prev = self.obs_encoder(x_prev)  # [B, D]
        m_cand = self.seq_emb(z_prev, a_prev).unsqueeze(1)  # [B, 1, D]

        # 2. Sequence construction: concat([h_old, M, m_cand])
        h_seq = h_old.unsqueeze(1)  # [B, 1, D]
        S = torch.cat([h_seq, M, m_cand], dim=1)  # [B, K+2, D]

        # Add learned type embeddings: h_old=0, M=1, cand=2
        type_ids = torch.tensor([0] + [1] * K + [2], device=x_prev.device, dtype=torch.long)
        S = S + self.type_emb(type_ids).unsqueeze(0)

        # 3. Differentiable attention masking
        # Per token mask bias: h_old: 0, cand: 0, M slots: (1 - m) * -10^4
        # PyTorch TransformerEncoder accepts src_key_padding_mask [B, S_len] (True means masked out / -inf)
        # Or src_mask [S_len, S_len] or [B*nhead, S_len, S_len].
        # For full differentiability with m, we construct additive mask [B * nhead, K+2, K+2]
        # or expand attention bias:
        slot_bias = (1.0 - m.squeeze(-1)) * (-10000.0)  # [B, K]
        h_bias = torch.zeros(B, 1, device=x_prev.device)
        cand_bias = torch.zeros(B, 1, device=x_prev.device)
        key_bias = torch.cat([h_bias, slot_bias, cand_bias], dim=1)  # [B, K+2]

        # Expand key_bias for all heads and query positions: [B * nhead, K+2, K+2]
        nhead = self.transformer.layers[0].self_attn.num_heads
        key_bias_expanded = key_bias.unsqueeze(1).unsqueeze(1).repeat(1, nhead, K + 2, 1)  # [B, nhead, K+2, K+2]
        attn_mask = key_bias_expanded.view(B * nhead, K + 2, K + 2)

        trans_out = self.transformer(S, mask=attn_mask)  # [B, K+2, D]

        # 4. Extract next fast state
        h_new = self.h_out_proj(trans_out[:, -1, :])  # [B, D]

        # 5. Parallel eviction gate on ALL slots simultaneously
        h_new_exp = h_new.unsqueeze(1).expand(B, K, D)  # [B, K, D]
        evict_in = torch.cat([M, h_new_exp], dim=-1)     # [B, K, 2D]
        k_logits = self.evict_gate(evict_in)            # [B, K, 1]
        k = torch.sigmoid(k_logits)                     # [B, K, 1]
        # Straight-Through Estimator (STE)
        k_ste = (k > 0.5).float() - k.detach() + k
        m_kept = m * k_ste                              # [B, K, 1]

        # 6. Write gate: decide whether to store M_cand
        write_in = torch.cat([m_cand.squeeze(1), h_new], dim=-1)  # [B, 2D]
        w_logits = self.write_gate(write_in)                     # [B, 1]
        w = torch.sigmoid(w_logits).unsqueeze(-1)                # [B, 1, 1]
        w_ste = (w > 0.5).float() - w.detach() + w

        # 7. Differentiable slot write (lowest occupancy slot)
        # idx = argmin(m_kept[:, i, 0])
        occ_scores = m_kept.squeeze(-1)  # [B, K]
        idx = torch.argmin(occ_scores, dim=-1)  # [B]
        o = F.one_hot(idx, num_classes=K).float().unsqueeze(-1)  # [B, K, 1]
        u = o * w_ste  # [B, K, 1]

        # Out-of-place state update
        M_next = M * (1.0 - u) + m_cand * u
        m_next = m_kept * (1.0 - u) + 1.0 * u

        obs_pred, rew_pred = self.pred_head(h_new)

        next_state = {
            "h": h_new,
            "M": M_next,
            "m": m_next,
        }
        aux = {
            "write_prob": w.squeeze(-1),     # [B, 1]
            "keep_probs": k,                 # [B, K, 1]
            "write_mask": u,                 # [B, K, 1]
            "m_kept": m_kept,
        }
        return (obs_pred, rew_pred), next_state, aux

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

        # At t=0, model receives obs[:, 0] and act[:, 0] to predict next step obs[:, 1], etc.
        for t in range(T):
            x_t = obs[:, t]
            a_t = act[:, t]
            (o_pred, r_pred), state, aux = self.step(x_t, a_t, state)
            obs_preds.append(o_pred)
            rew_preds.append(r_pred)
            write_probs.append(aux["write_prob"])
            keep_probs.append(aux["keep_probs"])
            m_history.append(state["m"])

        return {
            "obs_pred": torch.stack(obs_preds, dim=1),       # [B, T, obs_dim]
            "rew_pred": torch.stack(rew_preds, dim=1),       # [B, T, 1]
            "write_probs": torch.stack(write_probs, dim=1),   # [B, T, 1]
            "keep_probs": torch.stack(keep_probs, dim=1),     # [B, T, K, 1]
            "m_history": torch.stack(m_history, dim=1),       # [B, T, K, 1]
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

    def step(self, x_prev, a_prev, state):
        z_prev = self.obs_encoder(x_prev)
        m_cand = self.seq_emb(z_prev, a_prev).unsqueeze(1)  # [B, 1, D]
        out, h_next = self.gru(m_cand, state["h"])
        h_feat = self.h_out_proj(out.squeeze(1))
        obs_pred, rew_pred = self.pred_head(h_feat)
        return (obs_pred, rew_pred), {"h": h_next}, {}

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
        self.pos_emb = nn.Parameter(torch.randn(1, window_size + 1, d_model) * 0.02)
        self.h_out_proj = nn.Linear(d_model, d_model)
        self.pred_head = PredHead(d_model, obs_dim)

    def init_state(self, batch_size, device):
        return {
            "buf": torch.zeros(batch_size, 0, self.d_model, device=device)
        }

    def step(self, x_prev, a_prev, state):
        B = x_prev.shape[0]
        z_prev = self.obs_encoder(x_prev)
        m_cand = self.seq_emb(z_prev, a_prev).unsqueeze(1)  # [B, 1, D]

        buf = state["buf"]
        if buf.shape[1] >= self.window_size:
            buf = buf[:, 1:, :]  # FIFO slide
        buf = torch.cat([buf, m_cand], dim=1)  # [B, L, D]

        L = buf.shape[1]
        seq = buf + self.pos_emb[:, :L, :]
        trans_out = self.transformer(seq)
        h_feat = self.h_out_proj(trans_out[:, -1, :])
        obs_pred, rew_pred = self.pred_head(h_feat)

        return (obs_pred, rew_pred), {"buf": buf}, {}

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
