"""Configuration and hyperparameters for SM-WM experiments."""
from dataclasses import dataclass

@dataclass
class Config:
    obs_dim: int = 16
    act_dim: int = 4
    d_model: int = 128
    num_slots: int = 16
    nhead: int = 4
    num_layers: int = 2
    dropout: float = 0.0
    
    # Sequence & optimization
    seq_len: int = 64
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    epochs: int = 30
    
    # Curriculum penalties & target capacity
    target_slots: float = 5.0
    warmup_epochs: int = 10
    ramp_epochs: int = 22
    lambda_w_max: float = 0.005
    lambda_k_max: float = 0.015
    rollout_loss_weight: float = 0.5
    rollout_horizon: int = 4
    
    # Gate bias inits
    write_gate_bias_init: float = 1.0
    evict_gate_bias_init: float = 0.0
    
    # Dataset
    train_episodes: int = 10000
    val_episodes: int = 1000
    ood_episodes: int = 1000
    episode_len: int = 80
    
    seed: int = 42
    device: str = "cpu"
