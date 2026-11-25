"""Training loop for the lightweight DreamerV3."""

import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from .agent import Dreamer
from .config import Config
from .envs import Env
from .replay import Replay


@torch.no_grad()
def evaluate(agent: Dreamer, env: Env, episodes: int) -> float:
    returns = []
    for _ in range(episodes):
        obs = env.reset()
        state = agent.init_state(obs)
        done, total = False, 0.0
        while not done:
            action = agent.act(state, deterministic=True)
            obs, reward, done, _ = env.step(action)
            state = agent.observe_state(obs, state, action)
            total += reward
        returns.append(total)
    return float(np.mean(returns))


def main() -> None:
    cfg = Config.parse()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    env = Env(cfg, seed=cfg.seed)
    eval_env = Env(cfg, seed=cfg.seed + 1) if cfg.eval_every > 0 else None
    replay = Replay(
        cfg.replay_size,
        env.obs_shape,
        discrete=env.discrete,
        pixels=cfg.pixels,
        seed=cfg.seed,
    )
    agent = Dreamer(cfg, env.obs_shape, env.action_dim, env.discrete)

    step = agent.load(cfg.resume) if cfg.resume else 0
    obs = env.reset()
    state = agent.init_state(obs)
    ep_return, ep_len = 0.0, 0
    recent_returns: deque[float] = deque(maxlen=20)
    metrics: dict[str, float] = {}
    last_step, last_time = step, time.time()

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    while step < cfg.steps:
        if step < cfg.warmup:
            action = env.random_action()
        else:
            action = agent.act(state)

        next_obs, reward, done, cont = env.step(action)
        replay.add(obs, action, reward, cont)
        state = agent.observe_state(next_obs, state, action)
        obs = next_obs
        ep_return += reward
        ep_len += 1

        if done:
            replay.end_episode()
            recent_returns.append(ep_return)
            obs = env.reset()
            state = agent.init_state(obs)
            ep_return, ep_len = 0.0, 0

        step += 1

        if step >= cfg.warmup and step % cfg.train_every == 0 and replay.episodes:
            for _ in range(cfg.updates_per_train):
                batch = replay.sample(
                    cfg.batch_size, cfg.batch_length, agent.device)
                metrics = agent.update(batch)

        if step % cfg.log_every == 0:
            now = time.time()
            sps = (step - last_step) / max(now - last_time, 1e-6)
            ret = float(np.mean(recent_returns)
                        ) if recent_returns else float("nan")
            msg = f"step {step:7d} | return {ret:7.1f} | fps {sps:5.0f}"
            for key in ("wm_loss", "rec", "kl", "reward", "cont", "actor_loss", "critic_loss", "entropy"):
                if key in metrics:
                    msg += f" | {key} {metrics[key]:+.3f}"
            print(msg, flush=True)
            last_step, last_time = step, now

        if eval_env is not None and step % cfg.eval_every == 0:
            score = evaluate(agent, eval_env, cfg.eval_episodes)
            print(f"step {step:7d} | eval return {score:7.1f}", flush=True)

        if cfg.ckpt_every > 0 and step % cfg.ckpt_every == 0:
            path = out_dir / f"ckpt_{step:07d}.pt"
            agent.save(str(path), step)
            print(f"saved {path}", flush=True)

    agent.save(str(out_dir / "ckpt_final.pt"), step)
    env.close()
    if eval_env is not None:
        eval_env.close()
    print(f"done at step {step}; final checkpoint in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
