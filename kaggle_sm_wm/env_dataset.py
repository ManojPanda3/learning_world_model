"""Gymnasium KeymapMemoryEnv and fast dataset generation."""
import itertools
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

ALL_PERMUTATIONS = list(itertools.permutations([0, 1, 2, 3]))  # 24 permutations
# 0: Up (-1, 0), 1: Down (+1, 0), 2: Left (0, -1), 3: Right (0, +1)
MOVE_DELTAS = [
    np.array([-1, 0], dtype=np.int32),  # Up
    np.array([1, 0], dtype=np.int32),   # Down
    np.array([0, -1], dtype=np.int32),  # Left
    np.array([0, 1], dtype=np.int32),   # Right
]

class KeymapMemoryEnv(gym.Env):
    """
    KeymapMemoryEnv:
    - 5x5 Grid.
    - Actions 0..3 map to a random permutation of {Up, Down, Left, Right}.
    - Permutations 0..17: train / val.
    - Permutations 18..23: ood.
    - Key spawned at t in [5, 15] of color c in {0..3}.
    - Door spawned matching color c after distractor delay (20..40 steps).
    - Carrying indicator masked to 0 after pickup step.
    - Obs: 16 dims.
    """
    metadata = {"render_modes": []}

    def __init__(self, fixed_perm_id=None):
        super().__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(16,), dtype=np.float32)
        self.fixed_perm_id = fixed_perm_id
        
        self.t = 0
        self.max_t = 80
        self.pos = np.zeros(2, dtype=np.int32)
        self.perm_id = 0
        self.key_step = 10
        self.door_step = 40
        self.key_color = 0
        self.has_key = False
        self.door_opened = False
        self.probed_actions = set()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        split = "train"
        if options and "split" in options:
            split = options["split"]

        if self.fixed_perm_id is not None:
            self.perm_id = self.fixed_perm_id
        elif split == "ood":
            self.perm_id = int(self.np_random.integers(18, 24))
        else:
            self.perm_id = int(self.np_random.integers(0, 18))

        self.t = 0
        self.pos = np.array([self.np_random.integers(0, 5), self.np_random.integers(0, 5)], dtype=np.int32)
        self.key_step = int(self.np_random.integers(5, 16))
        distractor_delay = int(self.np_random.integers(20, 41))
        self.door_step = min(self.key_step + distractor_delay, self.max_t - 2)
        self.key_color = int(self.np_random.integers(0, 4))
        self.has_key = False
        self.door_opened = False
        self.probed_actions = set()

        info = {
            "perm_id": self.perm_id,
            "is_novel_probe": False,
            "is_key_pickup": False,
            "is_door_open": False,
            "is_distractor": True,
        }
        obs = self._get_obs(is_pickup_step=False, is_door_step=False)
        return obs, info

    def step(self, action):
        action = int(action)
        is_novel_probe = action not in self.probed_actions
        self.probed_actions.add(action)

        actual_dir = ALL_PERMUTATIONS[self.perm_id][action]
        delta = MOVE_DELTAS[actual_dir]
        self.pos = np.clip(self.pos + delta, 0, 4)

        self.t += 1
        is_key_pickup = (self.t == self.key_step)
        if is_key_pickup:
            self.has_key = True

        is_door_open = (self.t == self.door_step and self.has_key)
        reward = 0.0
        if is_door_open:
            reward = 1.0
            self.has_key = False
            self.door_opened = True

        is_distractor = not (is_novel_probe or is_key_pickup or is_door_open)

        info = {
            "perm_id": self.perm_id,
            "is_novel_probe": is_novel_probe,
            "is_key_pickup": is_key_pickup,
            "is_door_open": is_door_open,
            "is_distractor": is_distractor,
        }

        obs = self._get_obs(is_pickup_step=is_key_pickup, is_door_step=is_door_open)
        terminated = False
        truncated = (self.t >= self.max_t)

        return obs, reward, terminated, truncated, info

    def _get_obs(self, is_pickup_step: bool, is_door_step: bool):
        # 16 dims:
        # [0:2] agent (x,y) normalized to [-1, 1]
        # [2:7] tile type one-hot: {empty, wall, key, door, distractor}
        # [7:11] tile color one-hot (4 dims)
        # [11] carrying-key indicator (1 dim, only 1 on exact pickup step)
        # [12] door-open event flag (1 dim)
        # [13:16] step noise / distractor features (3 dims)
        obs = np.zeros(16, dtype=np.float32)
        obs[0:2] = (self.pos.astype(np.float32) / 2.0) - 1.0

        if is_pickup_step:
            obs[2 + 2] = 1.0  # key tile
            obs[7 + self.key_color] = 1.0
            obs[11] = 1.0     # carrying key indicator on pickup step
        elif is_door_step:
            obs[2 + 3] = 1.0  # door tile
            obs[7 + self.key_color] = 1.0
            obs[12] = 1.0     # door open flag
        else:
            obs[2 + 0] = 1.0  # empty tile

        # 3 dims distractor noise
        obs[13:16] = self.np_random.uniform(-0.1, 0.1, size=3).astype(np.float32)
        return obs


def collect_dataset(num_episodes=1000, split="train", seed=42):
    """Collect episodes using an exploratory policy."""
    rng = np.random.default_rng(seed)
    all_obs = []
    all_act = []
    all_rew = []
    all_info_novel = []
    all_info_pickup = []
    all_info_door = []
    all_info_distractor = []
    all_perm_ids = []

    for ep in range(num_episodes):
        env = KeymapMemoryEnv()
        obs, info = env.reset(seed=int(rng.integers(0, 1000000)), options={"split": split})
        ep_obs = [obs]
        ep_act = []
        ep_rew = []
        ep_novel = []
        ep_pickup = []
        ep_door = []
        ep_dist = []

        # Early probe: try actions 0, 1, 2, 3 early, then random walk
        for step_i in range(80):
            if step_i < 4:
                act = step_i
            else:
                act = int(rng.integers(0, 4))
            
            next_obs, rew, term, trunc, info = env.step(act)
            ep_act.append(act)
            ep_rew.append(rew)
            ep_novel.append(info["is_novel_probe"])
            ep_pickup.append(info["is_key_pickup"])
            ep_door.append(info["is_door_open"])
            ep_dist.append(info["is_distractor"])
            if step_i < 79:
                ep_obs.append(next_obs)

        all_obs.append(np.array(ep_obs, dtype=np.float32))       # [80, 16]
        all_act.append(np.array(ep_act, dtype=np.int64))         # [80]
        all_rew.append(np.array(ep_rew, dtype=np.float32))       # [80]
        all_info_novel.append(np.array(ep_novel, dtype=bool))
        all_info_pickup.append(np.array(ep_pickup, dtype=bool))
        all_info_door.append(np.array(ep_door, dtype=bool))
        all_info_distractor.append(np.array(ep_dist, dtype=bool))
        all_perm_ids.append(info["perm_id"])

    return {
        "obs": torch.from_numpy(np.stack(all_obs)),           # [N, 80, 16]
        "act": torch.from_numpy(np.stack(all_act)),           # [N, 80]
        "rew": torch.from_numpy(np.stack(all_rew)),           # [N, 80]
        "is_novel_probe": torch.from_numpy(np.stack(all_info_novel)),
        "is_key_pickup": torch.from_numpy(np.stack(all_info_pickup)),
        "is_door_open": torch.from_numpy(np.stack(all_info_door)),
        "is_distractor": torch.from_numpy(np.stack(all_info_distractor)),
        "perm_id": torch.tensor(all_perm_ids, dtype=torch.long),
    }
