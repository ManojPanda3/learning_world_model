"""Suite A World Models (~0.8M Parameter Budget):
- SM_WM: V8 Surprise-Driven Transition Memory + 4-step Local Recency Buffer + ablation flags
- GRU_WM: 2-layer GRU (hidden=256, ~0.80M params)
- SlidingTF_WM: 3-layer Pre-LN Transformer with fixed 16-step sliding window (~0.69M params)
- FullTF_WM: 3-layer Pre-LN Transformer with full causal episode context (~0.71M params)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalize without NaN backward gradients on zero vectors."""
    norm = torch.sqrt(torch.sum(x * x, dim=dim, keepdim=True) + eps)
    return x / norm


class ObsEncoder(nn.Module):
    def __init__(self, obs_dim: int = 16, d_model: int = 128):
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
    def __init__(self, d_model: int = 128, act_dim: int = 4):
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
    V8 Transition Encoder: encodes (x_{t-1}, a_{t-1}, x_t, Delta x_t, surprise e_t) -> M_cand in R^D.
    Input dims: obs_dim * 4 + act_dim = 68 dims.
    """
    def __init__(self, obs_dim: int = 16, act_dim: int = 4, d_model: int = 128):
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
    def __init__(self, d_model: int = 128, obs_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, obs_dim + 1),  # obs_pred (16) + rew_pred (1)
        )
        self.obs_dim = obs_dim

    def forward(self, h):
        out = self.net(h)
        return out[..., :self.obs_dim], out[..., self.obs_dim:self.obs_dim + 1]


class SM_WM(nn.Module):
    """
    V8 Sparse Memory World Model + 4-step Local Recency Buffer.

    Tokens seen by the Transformer at each step:
      [h_old (1), M_slots (K=16), Recency_buf (R=4), query/cand (1)]
    Supports Phase-2 component ablations cleanly:
      - use_transition_tokens=False: encodes only (x_curr, a_prev) instead of full 5-tuple
      - use_surprise_gate=False: zeros surprise input to write_gate & trans_encoder
      - use_eviction=False: disables learned eviction (k_ste = 1.0)
      - recency_len=0: disables 4-step recency buffer
    """
    def __init__(
        self,
        obs_dim: int = 16,
        act_dim: int = 4,
        d_model: int = 128,
        num_slots: int = 16,
        recency_len: int = 4,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        write_bias_init: float = 0.5,
        evict_bias_init: float = 0.5,
        use_transition_tokens: bool = True,
        use_surprise_gate: bool = True,
        use_eviction: bool = True,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.d_model = d_model
        self.num_slots = num_slots
        self.recency_len = recency_len
        self.use_transition_tokens = use_transition_tokens
        self.use_surprise_gate = use_surprise_gate
        self.use_eviction = use_eviction

        self.obs_encoder = ObsEncoder(obs_dim, d_model)
        self.seq_emb = SeqEmb(d_model, act_dim)
        self.trans_encoder = TransitionEncoder(obs_dim, act_dim, d_model)
        self.simple_cand_proj = nn.Sequential(
            nn.Linear(obs_dim + act_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Type embeddings: 0: h_old, 1: long-term M slots, 2: recency slots, 3: query/cand
        self.type_emb = nn.Embedding(4, d_model)
        if recency_len > 0:
            self.rec_pos_emb = nn.Parameter(torch.randn(1, recency_len, d_model) * 0.02)
        else:
            self.register_parameter("rec_pos_emb", None)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)

        self.h_out_proj = nn.Linear(d_model, d_model)
        self.h_norm = nn.LayerNorm(d_model)
        self.cand_ctx_proj = nn.Linear(d_model, d_model)
        self.store_norm = nn.LayerNorm(d_model)

        self.write_gate = nn.Sequential(
            nn.Linear(d_model * 2 + 1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )
        self.evict_gate = nn.Sequential(
            nn.Linear(d_model * 2 + 1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )

        nn.init.constant_(self.write_gate[-1].bias, write_bias_init)
        nn.init.constant_(self.evict_gate[-1].bias, evict_bias_init)

        self.pred_head = PredHead(d_model, obs_dim)

    def init_state(self, batch_size: int, device):
        return {
            "h": torch.zeros(batch_size, self.d_model, device=device),
            "M": torch.zeros(batch_size, self.num_slots, self.d_model, device=device),
            "m": torch.zeros(batch_size, self.num_slots, 1, device=device),
            "recency": torch.zeros(batch_size, self.recency_len, self.d_model, device=device),
            "recency_m": torch.zeros(batch_size, self.recency_len, 1, device=device),
        }

    def _build_attn_mask(self, m, recency_m, B: int, device, dtype):
        K = self.num_slots
        R = self.recency_len
        nhead = self.transformer.layers[0].self_attn.num_heads
        total_len = 1 + K + R + 1

        h_bias = torch.zeros(B, 1, device=device, dtype=dtype)
        slot_bias = (1.0 - m.squeeze(-1).detach().to(dtype)) * (-1e4)
        cand_bias = torch.zeros(B, 1, device=device, dtype=dtype)

        if R > 0:
            rec_bias = (1.0 - recency_m.squeeze(-1).detach().to(dtype)) * (-1e4)
            key_bias = torch.cat([h_bias, slot_bias, rec_bias, cand_bias], dim=1)
        else:
            key_bias = torch.cat([h_bias, slot_bias, cand_bias], dim=1)

        return key_bias.unsqueeze(1).unsqueeze(1).expand(B, nhead, total_len, total_len).reshape(
            B * nhead, total_len, total_len
        )

    def _assemble_tokens(self, h_token, M, recency, tail_token):
        B = h_token.shape[0]
        K = self.num_slots
        R = self.recency_len
        device = h_token.device

        if R > 0:
            rec_toks = recency + self.rec_pos_emb.to(recency.dtype)
            S = torch.cat([h_token.unsqueeze(1), M, rec_toks, tail_token], dim=1)
            type_ids = torch.tensor([0] + [1] * K + [2] * R + [3], device=device, dtype=torch.long)
        else:
            S = torch.cat([h_token.unsqueeze(1), M, tail_token], dim=1)
            type_ids = torch.tensor([0] + [1] * K + [3], device=device, dtype=torch.long)

        return S + self.type_emb(type_ids).unsqueeze(0).to(S.dtype)

    def predict_step(self, x_prev, a_prev, state):
        B = x_prev.shape[0]
        h_old = state["h"].detach()
        M = state["M"]
        m = state["m"]
        recency = state["recency"]
        recency_m = state["recency_m"]

        z_prev = self.obs_encoder(x_prev)
        q_emb = self.seq_emb(z_prev, a_prev).unsqueeze(1)

        S = self._assemble_tokens(h_old, M, recency, q_emb)
        attn_mask = self._build_attn_mask(m, recency_m, B, x_prev.device, S.dtype)
        trans_out = self.transformer(S, mask=attn_mask)
        h_new = self.h_norm(self.h_out_proj(trans_out[:, -1, :]))

        obs_pred, rew_pred = self.pred_head(h_new)
        next_pos = torch.clamp(x_prev[..., :2] + obs_pred[..., :2], -1.0, 1.0)
        full_obs_pred = torch.cat([next_pos, obs_pred[..., 2:]], dim=-1)

        new_state = {
            "h": h_new,
            "M": M,
            "m": m,
            "recency": recency,
            "recency_m": recency_m,
        }
        return (full_obs_pred, rew_pred), new_state

    def step(self, x_prev, a_prev, state, x_curr=None):
        (full_obs_pred, rew_pred), pred_state = self.predict_step(x_prev, a_prev, state)

        if x_curr is None:
            B = x_prev.shape[0]
            K = self.num_slots
            aux = {
                "write_prob": torch.zeros(B, 1, device=x_prev.device, dtype=x_prev.dtype),
                "keep_probs": torch.ones(B, K, 1, device=x_prev.device, dtype=x_prev.dtype),
                "write_mask": torch.zeros(B, K, 1, device=x_prev.device, dtype=x_prev.dtype),
                "m_kept": pred_state["m"],
                "surprise": torch.zeros(B, 1, device=x_prev.device, dtype=x_prev.dtype),
            }
            return (full_obs_pred, rew_pred), pred_state, aux

        B = x_prev.shape[0]
        K = self.num_slots
        D = self.d_model
        h_new = pred_state["h"]
        M = pred_state["M"]
        m = pred_state["m"]
        recency = pred_state["recency"]
        recency_m = pred_state["recency_m"]

        delta_x = x_curr - x_prev
        if self.use_surprise_gate:
            surprise_vec = (x_curr - full_obs_pred).detach()
            surprise_mag = surprise_vec.pow(2).mean(dim=-1, keepdim=True)
        else:
            surprise_vec = torch.zeros_like(x_curr)
            surprise_mag = torch.zeros(B, 1, device=x_prev.device, dtype=x_prev.dtype)

        if self.use_transition_tokens:
            m_cand_raw = self.trans_encoder(x_prev, a_prev, x_curr, delta_x, surprise_vec)
        else:
            m_cand_raw = self.simple_cand_proj(torch.cat([x_curr, a_prev], dim=-1))

        S = self._assemble_tokens(h_new, M, recency, m_cand_raw.unsqueeze(1))
        attn_mask = self._build_attn_mask(m, recency_m, B, x_prev.device, S.dtype)
        trans_out = self.transformer(S, mask=attn_mask)

        m_cand_ctx = self.cand_ctx_proj(trans_out[:, -1, :])
        h_post = self.h_norm(self.h_out_proj(trans_out[:, 0, :]))
        M_store = self.store_norm(m_cand_raw + 0.5 * m_cand_ctx).unsqueeze(1)  # [B, 1, D]

        # 4-step Local Recency Buffer FIFO update
        if self.recency_len > 0:
            ones_rec = torch.ones(B, 1, 1, device=x_prev.device, dtype=recency_m.dtype)
            recency_next = torch.cat([recency[:, 1:, :], M_store], dim=1)
            recency_m_next = torch.cat([recency_m[:, 1:, :], ones_rec], dim=1)
        else:
            recency_next = recency
            recency_m_next = recency_m

        # Write gate (uses h_post seeing true event outcome)
        write_in = torch.cat([M_store.squeeze(1), h_post, surprise_mag], dim=-1)
        w = torch.sigmoid(self.write_gate(write_in)).unsqueeze(-1)  # [B, 1, 1]
        w_ste = (w > 0.5).to(w.dtype) - w.detach() + w

        # Duplicate redundancy check (NaN-safe at M=0)
        M_norm = _safe_normalize(M, dim=-1)
        store_norm = _safe_normalize(M_store, dim=-1)
        cos_sim = torch.sum(M_norm * store_norm, dim=-1, keepdim=True)
        gated_sim = cos_sim * w.detach()

        # Evict gate (uses h_post to confirm event completion, e.g. door unlock)
        if self.use_eviction:
            h_post_exp = h_post.unsqueeze(1).expand(B, K, D)
            evict_in = torch.cat([M, h_post_exp, gated_sim], dim=-1)
            k = torch.sigmoid(self.evict_gate(evict_in))
            k_ste = (k > 0.5).to(k.dtype) - k.detach() + k
        else:
            k = torch.ones(B, K, 1, device=x_prev.device, dtype=M.dtype)
            k_ste = k
        m_kept = m * k_ste

        # Slot allocation (least occupied slot)
        idx = torch.argmin(m_kept.squeeze(-1), dim=-1)
        o = F.one_hot(idx, num_classes=K).to(M.dtype).unsqueeze(-1)
        u = o * w_ste

        M_next = M * (1.0 - u) + M_store * u
        m_next = m_kept * (1.0 - u) + u

        next_state = {
            "h": h_post,
            "M": M_next,
            "m": m_next,
            "recency": recency_next,
            "recency_m": recency_m_next,
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
        B, T, _ = obs.shape
        state = self.init_state(B, obs.device)

        obs_preds, rew_preds, write_probs, keep_probs, m_history = [], [], [], [], []
        for t in range(T):
            x_t = obs[:, t]
            a_t = act[:, t]
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
    """Baseline A: 2-layer GRU with hidden=256 (~0.80M params)."""
    def __init__(self, obs_dim: int = 16, act_dim: int = 4, d_model: int = 128, hidden_size: int = 256):
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

    def init_state(self, batch_size: int, device):
        return {"h": torch.zeros(2, batch_size, self.hidden_size, device=device)}

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
        z = self.obs_encoder(obs)
        tokens = self.seq_emb(z, act)
        h0 = torch.zeros(2, obs.shape[0], self.hidden_size, device=obs.device, dtype=tokens.dtype)
        out, _ = self.gru(tokens, h0)
        h_feat = self.h_out_proj(out)
        obs_pred, rew_pred = self.pred_head(h_feat)
        next_pos = torch.clamp(obs[..., :2] + obs_pred[..., :2], -1.0, 1.0)
        full_obs_pred = torch.cat([next_pos, obs_pred[..., 2:]], dim=-1)
        return {"obs_pred": full_obs_pred, "rew_pred": rew_pred}


class SlidingTF_WM(nn.Module):
    """Baseline B: 3-layer Pre-LN Transformer with fixed 16-step sliding window (~0.69M params)."""
    def __init__(
        self,
        obs_dim: int = 16,
        act_dim: int = 4,
        d_model: int = 128,
        window_size: int = 16,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
    ):
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
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.h_out_proj = nn.Linear(d_model, d_model)
        self.pred_head = PredHead(d_model, obs_dim)

    def init_state(self, batch_size: int, device):
        return {"buf": torch.zeros(batch_size, 0, self.d_model, device=device)}

    def step(self, x_prev, a_prev, state, x_curr=None):
        z_prev = self.obs_encoder(x_prev)
        m_cand = self.seq_emb(z_prev, a_prev).unsqueeze(1)
        buf = state["buf"]
        buf = torch.cat([buf, m_cand], dim=1) if buf.shape[1] < self.window_size else torch.cat([buf[:, 1:], m_cand], dim=1)
        L = buf.shape[1]
        seq = buf + self.pos_emb[:, :L, :].to(buf.dtype)
        trans_out = self.transformer(seq)
        h_feat = self.h_out_proj(trans_out[:, -1, :])
        obs_pred, rew_pred = self.pred_head(h_feat)
        next_pos = torch.clamp(x_prev[..., :2] + obs_pred[..., :2], -1.0, 1.0)
        full_obs_pred = torch.cat([next_pos, obs_pred[..., 2:]], dim=-1)
        return (full_obs_pred, rew_pred), {"buf": buf}, {}

    def forward(self, obs, act):
        B, T, _ = obs.shape
        state = self.init_state(B, obs.device)
        obs_preds, rew_preds = [], []
        for t in range(T):
            (o_pred, r_pred), state, _ = self.step(obs[:, t], act[:, t], state)
            obs_preds.append(o_pred)
            rew_preds.append(r_pred)
        return {"obs_pred": torch.stack(obs_preds, dim=1), "rew_pred": torch.stack(rew_preds, dim=1)}


class FullTF_WM(nn.Module):
    """Baseline C: 3-layer Pre-LN Transformer with full causal episode context (~0.71M params)."""
    def __init__(
        self,
        obs_dim: int = 16,
        act_dim: int = 4,
        d_model: int = 128,
        max_seq_len: int = 256,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.obs_encoder = ObsEncoder(obs_dim, d_model)
        self.seq_emb = SeqEmb(d_model, act_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.h_out_proj = nn.Linear(d_model, d_model)
        self.pred_head = PredHead(d_model, obs_dim)

    def init_state(self, batch_size: int, device):
        return {"buf": torch.zeros(batch_size, 0, self.d_model, device=device)}

    def step(self, x_prev, a_prev, state, x_curr=None):
        z_prev = self.obs_encoder(x_prev)
        m_cand = self.seq_emb(z_prev, a_prev).unsqueeze(1)
        buf = torch.cat([state["buf"], m_cand], dim=1)
        if buf.shape[1] > self.max_seq_len:
            buf = buf[:, -self.max_seq_len:, :]
        L = buf.shape[1]
        seq = buf + self.pos_emb[:, :L, :].to(buf.dtype)
        causal_mask = torch.triu(torch.full((L, L), -1e4, device=x_prev.device, dtype=seq.dtype), diagonal=1)
        trans_out = self.transformer(seq, mask=causal_mask)
        h_feat = self.h_out_proj(trans_out[:, -1, :])
        obs_pred, rew_pred = self.pred_head(h_feat)
        next_pos = torch.clamp(x_prev[..., :2] + obs_pred[..., :2], -1.0, 1.0)
        full_obs_pred = torch.cat([next_pos, obs_pred[..., 2:]], dim=-1)
        return (full_obs_pred, rew_pred), {"buf": buf}, {}

    def forward(self, obs, act):
        B, T, _ = obs.shape
        assert T <= self.max_seq_len, f"T={T} exceeds max_seq_len={self.max_seq_len}"
        z = self.obs_encoder(obs)
        tokens = self.seq_emb(z, act)
        seq = tokens + self.pos_emb[:, :T, :].to(tokens.dtype)
        causal_mask = torch.triu(torch.full((T, T), -1e4, device=obs.device, dtype=seq.dtype), diagonal=1)
        trans_out = self.transformer(seq, mask=causal_mask)
        h_feat = self.h_out_proj(trans_out)
        obs_pred, rew_pred = self.pred_head(h_feat)
        next_pos = torch.clamp(obs[..., :2] + obs_pred[..., :2], -1.0, 1.0)
        full_obs_pred = torch.cat([next_pos, obs_pred[..., 2:]], dim=-1)
        return {"obs_pred": full_obs_pred, "rew_pred": rew_pred}
