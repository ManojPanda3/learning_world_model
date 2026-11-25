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
