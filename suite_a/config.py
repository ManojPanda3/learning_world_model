"""Configuration and hyperparameters for Suite A experiments (~0.8M parameter budget)."""
from dataclasses import dataclass


@dataclass
class SuiteAConfig:
    obs_dim: int = 16
    act_dim: int = 4
    d_model: int = 128
    num_slots: int = 16
    recency_len: int = 4
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 512
    gru_hidden: int = 256
    max_seq_len: int = 256
    dropout: float = 0.0

    # Sequence & optimization
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    epochs: int = 25

    # Curriculum penalties & target capacity
    target_slots: float = 5.0
    warmup_epochs: int = 10
    ramp_epochs: int = 20
    lambda_w_max: float = 0.001
    lambda_k_max: float = 0.015
    rollout_loss_weight: float = 0.5
    rollout_horizon: int = 4
    reward_loss_weight: float = 2.0

    # Gate bias inits: keep slots by default (+2.0), write novel transitions (+1.0)
    write_gate_bias_init: float = 1.0
    evict_gate_bias_init: float = 2.0

    # Dataset (200k transitions per task)
    target_train_transitions: int = 200_000
    val_episodes: int = 400
    eval_episodes: int = 100

    seed: int = 42
    device: str = "cpu"
