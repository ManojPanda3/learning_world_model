# Lightweight DreamerV3 in PyTorch (CPU-ready)

Minimal PyTorch implementation of Danijar Hafner's [DreamerV3](https://github.com/danijar/dreamerv3) ("Mastering Diverse Domains through World Models"). Stripped down for CPU training, fast experimentation, and readable learning code (~700 LOC total, zero heavy dependencies).

## What's included

- **World Model (RSSM)**: GRU-based deterministic path + factorized categorical stochastic latents with straight-through gradient estimator and unimix exploration.
- **Categorical KL Balancing**: 0.5 dynamic / 0.1 representation balance with free-nats clamping.
- **Symlog transforms**: robust scaling for continuous observations and targets.
- **Two-Hot encoding**: scalar predictions for reward and critic over 255 discrete bins.
- **Latent imagination**: rollout trajectories purely in world-model state space.
- **Actor-Critic**: REINFORCE actor with running 5–95 percentile return normalization, slow critic target network, and $\lambda$-returns.
- **Vector & Pixel support**: MLP encoder/decoder for vectors (default CartPole), lightweight 4-layer CNN for images (`--pixels True`).

## Quick start

```bash
# 1. Install (uv uses PyTorch CPU wheels)
uv sync

# 2. Train on CartPole-v1 (reaches 100+ return in ~3-4k steps)
uv run python -m learning_world_model.train \
  --steps 10000 \
  --warmup 500 \
  --train_every 4 \
  --batch_size 16 \
  --batch_length 16 \
  --horizon 10

# 3. Resume from checkpoint
uv run python -m learning_world_model.train --resume runs/dreamer/ckpt_final.pt
```

## Pixel mode (e.g. 64x64)

```bash
uv run python -m learning_world_model.train \
  --pixels True \
  --img_size 64 \
  --batch_size 8 \
  --batch_length 16 \
  --steps 20000
```

## Code layout

- `src/learning_world_model/utils.py`: symlog, two-hot loss/decode, MLP, percentile `ReturnNorm`, KL.
- `src/learning_world_model/models.py`: CNN/MLP `Encoder`/`Decoder`, categorical `RSSM`, `Actor`, `Critic`.
- `src/learning_world_model/agent.py`: `Dreamer` agent (state tracking, world-model loss, latent imagination, actor-critic update).
- `src/learning_world_model/replay.py`: minimal zero-dependency episode buffer.
- `src/learning_world_model/envs.py`: Gymnasium wrapper handling continuous/discrete actions and optional pixel rendering.
- `src/learning_world_model/train.py`: CLI training loop with evaluation and checkpointing.
- `src/learning_world_model/config.py`: typed configuration dataclass.
