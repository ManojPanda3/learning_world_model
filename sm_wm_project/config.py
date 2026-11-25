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
    
    # Anti-collapse curriculum penalties
    warmup_epochs: int = 5
    ramp_epochs: int = 15
    lambda_w_max: float = 0.02
    lambda_k_max: float = 0.012
    
    # Gate bias inits
    write_gate_bias_init: float = 0.5
    evict_gate_bias_init: float = 0.5
    
    # Dataset
    train_episodes: int = 10000
    val_episodes: int = 1000
    ood_episodes: int = 1000
    episode_len: int = 80
    
    seed: int = 42
    device: str = "cpu"
