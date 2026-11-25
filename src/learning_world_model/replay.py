"""Simple episode-based replay buffer (no external dependencies)."""

import random

import numpy as np
import torch


class Episode:
    __slots__ = ("obs", "action", "reward", "cont")

    def __init__(self):
        self.obs: list[np.ndarray] = []
        self.action: list = []
        self.reward: list[float] = []
        self.cont: list[float] = []

    def __len__(self):
        return len(self.obs)


class Replay:
    def __init__(self, capacity: int, obs_shape: tuple[int, ...], discrete: bool, pixels: bool, seed: int = 0):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.discrete = discrete
        self.pixels = pixels
        self.obs_dtype = np.uint8 if pixels else np.float32
        self.episodes: list[Episode] = []
        self.total = 0
        self.rng = random.Random(seed)
        self.current = Episode()

    def add(self, obs, action, reward: float, cont: float) -> None:
        self.current.obs.append(np.asarray(obs, dtype=self.obs_dtype))
        self.current.action.append(int(action) if self.discrete else np.float32(action))
        self.current.reward.append(float(reward))
        self.current.cont.append(float(cont))

    def end_episode(self) -> None:
        if len(self.current) == 0:
            return
        self.episodes.append(self.current)
        self.total += len(self.current)
        self.current = Episode()
        while self.total > self.capacity and len(self.episodes) > 1:
            old = self.episodes.pop(0)
            self.total -= len(old)

    def __len__(self) -> int:
        return self.total + len(self.current)

    def sample(self, batch_size: int, length: int, device: torch.device) -> dict[str, torch.Tensor]:
        obs_b, act_b, rew_b, cont_b, mask_b = [], [], [], [], []
        for _ in range(batch_size):
            ep = self.rng.choice(self.episodes)
            if len(ep) >= length:
                start = self.rng.randint(0, len(ep) - length)
            else:
                start = 0
            n = min(length, len(ep) - start)

            obs = np.zeros((length, *self.obs_shape), dtype=self.obs_dtype)
            act = np.zeros(length, dtype=np.int64 if self.discrete else np.float32)
            rew = np.zeros(length, dtype=np.float32)
            cont = np.zeros(length, dtype=np.float32)
            mask = np.zeros(length, dtype=np.float32)
            if n > 0:
                obs[:n] = np.stack(ep.obs[start : start + n])
                act[:n] = np.asarray(ep.action[start : start + n])
                rew[:n] = np.asarray(ep.reward[start : start + n])
                cont[:n] = np.asarray(ep.cont[start : start + n])
                mask[:n] = 1.0
            obs_b.append(obs)
            act_b.append(act)
            rew_b.append(rew)
            cont_b.append(cont)
            mask_b.append(mask)

        obs = torch.from_numpy(np.stack(obs_b))
        if self.pixels:
            obs = obs.float().div_(255.0)
        else:
            obs = obs.float()
        return {
            "obs": obs.to(device),
            "action": torch.from_numpy(np.stack(act_b)).to(device),
            "reward": torch.from_numpy(np.stack(rew_b)).to(device),
            "cont": torch.from_numpy(np.stack(cont_b)).to(device),
            "mask": torch.from_numpy(np.stack(mask_b)).to(device),
        }
