"""Configuration for the lightweight DreamerV3."""

import argparse
from dataclasses import dataclass, fields


@dataclass
class Config:
    # Environment
    env_id: str = "CartPole-v1"
    pixels: bool = False
    img_size: int = 64
    seed: int = 0

    # Interaction / training schedule
    steps: int = 50_000
    warmup: int = 5_000
    train_every: int = 8
    updates_per_train: int = 1
    replay_size: int = 100_000
    batch_size: int = 16
    batch_length: int = 32
    eval_every: int = 10_000
    eval_episodes: int = 5
    log_every: int = 1_000
    ckpt_every: int = 10_000

    # World model
    deter: int = 128
    hidden: int = 128
    embed: int = 128
    classes: int = 16
    dims: int = 8
    unimix: float = 0.01

    # Loss scales
    recon_scale: float = 1.0
    reward_scale: float = 1.0
    cont_scale: float = 1.0
    kl_free: float = 1.0
    dyn_scale: float = 0.5
    rep_scale: float = 0.1

    # Imagination / RL
    horizon: int = 15
    gamma: float = 0.997
    lam: float = 0.95
    entropy_scale: float = 3e-4
    slow_tau: float = 0.02

    # Optimization
    model_lr: float = 3e-4
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    grad_clip: float = 100.0

    # Runtime
    device: str = "cpu"
    out_dir: str = "runs/dreamer"
    resume: str = ""

    @classmethod
    def parse(cls) -> "Config":
        parser = argparse.ArgumentParser(description="Lightweight DreamerV3 in PyTorch")
        for f in fields(cls):
            default = f.default
            if isinstance(default, bool):
                parser.add_argument(
                    f"--{f.name}",
                    type=lambda s: s.lower() in ("1", "true", "yes", "y"),
                    default=default,
                    help=f"(default: {default})",
                )
            else:
                parser.add_argument(
                    f"--{f.name}",
                    type=type(default),
                    default=default,
                    help=f"(default: {default})",
                )
        return cls(**vars(parser.parse_args()))
