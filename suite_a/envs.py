"""Suite A Mechanistic Probe Environments & Dataset Collector.

Tasks:
1. KeyDoorScalingEnv (L in {16, 32, 64, 128}):
   - Key of color c appears at key_step.
   - After L distractor steps, a door cue of color d appears at door_step = key_step + L.
   - 50% matching color (d == c -> reward=1.0, door_open=1.0), 50% decoy (d != c -> reward=0.0, door_open=0.0).
   - Because cue & decoy look identical locally, predicting reward/unlock at t=door_step requires
     retaining key_color across L steps.

2. ScrambledControllerEnv:
   - Starts with action permutation perm_1 (0..17 train/val, 18..23 ood).
   - Silently switches at t_flip in [25, 40] to a distinct permutation perm_2 != perm_1.
   - Tests fast in-context adaptation (via 4-step recency buffer + surprise-gated memory).

3. MultiStageKeyDoorEnv:
   - 3 sequential (key_i, door_i) stages in one episode (T=140).
   - Proves event-triggered eviction: after door_i unlocks, the slot is cleared before key_{i+1}.
"""
import itertools
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

ALL_PERMUTATIONS = list(itertools.permutations([0, 1, 2, 3]))  # 24 permutations
MOVE_DELTAS = [
    np.array([-1, 0], dtype=np.int32),  # 0: Up
    np.array([1, 0], dtype=np.int32),   # 1: Down
    np.array([0, -1], dtype=np.int32),  # 2: Left
    np.array([0, 1], dtype=np.int32),   # 3: Right
]

TASK_SPECS = {
    "kd_L16": {"kind": "key_door", "delay_L": 16, "episode_len": 64},
    "kd_L32": {"kind": "key_door", "delay_L": 32, "episode_len": 64},
    "kd_L64": {"kind": "key_door", "delay_L": 64, "episode_len": 96},
    "kd_L128": {"kind": "key_door", "delay_L": 128, "episode_len": 160},
    "scrambled": {"kind": "scrambled", "delay_L": 0, "episode_len": 80},
    "multistage": {"kind": "multistage", "delay_L": 24, "episode_len": 140},
}


def _sample_perm(rng, split: str, exclude_perm: int | None = None) -> int:
    lo, hi = (18, 24) if split == "ood" else (0, 18)
    choices = [p for p in range(lo, hi) if p != exclude_perm]
    return int(choices[int(rng.integers(0, len(choices)))])


def _make_obs(
    rng,
    pos: np.ndarray,
    tile_type: int,
    color: int | None = None,
    carrying_flag: float = 0.0,
    door_open_flag: float = 0.0,
) -> np.ndarray:
    """
    16-dim observation vector:
      [0:2]   normalized (x, y) position in [-1, 1]
      [2:7]   tile type one-hot: {0: empty, 1: wall/unused, 2: key, 3: door_cue, 4: door_result}
      [7:11]  color one-hot (4 dims)
      [11]    carrying-key pickup pulse (1.0 on key pickup step only)
      [12]    door-open event flag (1.0 on door unlock outcome step only)
      [13:16] 3-dim distractor noise in [-0.1, 0.1]
    """
    obs = np.zeros(16, dtype=np.float32)
    obs[0:2] = (pos.astype(np.float32) / 2.0) - 1.0
    obs[2 + tile_type] = 1.0
    if color is not None:
        obs[7 + color] = 1.0
    obs[11] = carrying_flag
    obs[12] = door_open_flag
    obs[13:16] = rng.uniform(-0.1, 0.1, size=3).astype(np.float32)
    return obs


class KeyDoorScalingEnv(gym.Env):
    """Key-Door environment with exact configurable delay L in {16, 32, 64, 128}."""
    metadata = {"render_modes": []}

    def __init__(self, delay_L: int = 16, max_t: int | None = None, fixed_perm_id: int | None = None):
        super().__init__()
        self.delay_L = int(delay_L)
        self.max_t = int(max_t) if max_t is not None else max(64, self.delay_L + 32)
        self.fixed_perm_id = fixed_perm_id

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(16,), dtype=np.float32)

        self.t = 0
        self.pos = np.zeros(2, dtype=np.int32)
        self.perm_id = 0
        self.key_step = 8
        self.door_cue_step = 8 + self.delay_L
        self.key_color = 0
        self.door_color = 0
        self.door_matches = True
        self.has_key = False
        self.probed_actions = set()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        split = options.get("split", "train") if options else "train"
        self.perm_id = self.fixed_perm_id if self.fixed_perm_id is not None else _sample_perm(self.np_random, split)

        self.t = 0
        self.pos = self.np_random.integers(0, 5, size=2, dtype=np.int32)
        latest_key = max(5, self.max_t - self.delay_L - 6)
        self.key_step = int(self.np_random.integers(4, min(14, latest_key + 1)))
        self.door_cue_step = self.key_step + self.delay_L - 1
        self.key_color = int(self.np_random.integers(0, 4))
        self.door_matches = bool(self.np_random.integers(0, 2) == 1)
        if self.door_matches:
            self.door_color = self.key_color
        else:
            other_colors = [c for c in range(4) if c != self.key_color]
            self.door_color = int(other_colors[int(self.np_random.integers(0, 3))])
        self.has_key = False
        self.probed_actions = set()

        obs = _make_obs(self.np_random, self.pos, tile_type=0)
        info = {
            "perm_id": self.perm_id,
            "is_novel_probe": False,
            "is_key_pickup": False,
            "is_door_cue": False,
            "is_door_open": False,
            "is_door_decoy": False,
            "is_distractor": True,
            "door_matches": self.door_matches,
        }
        return obs, info

    def step(self, action):
        action = int(action)
        is_novel_probe = action not in self.probed_actions
        self.probed_actions.add(action)

        actual_dir = ALL_PERMUTATIONS[self.perm_id][action]
        self.pos = np.clip(self.pos + MOVE_DELTAS[actual_dir], 0, 4)

        prev_was_door_cue = (self.t == self.door_cue_step)
        self.t += 1

        is_key_pickup = (self.t == self.key_step)
        if is_key_pickup:
            self.has_key = True

        is_door_cue = (self.t == self.door_cue_step)
        is_door_open = prev_was_door_cue and self.has_key and self.door_matches
        is_door_decoy = prev_was_door_cue and (not self.door_matches)

        reward = 0.0
        if is_door_open:
            reward = 1.0
            self.has_key = False
        elif is_door_decoy:
            self.has_key = False  # trial concludes

        if is_key_pickup:
            obs = _make_obs(self.np_random, self.pos, tile_type=2, color=self.key_color, carrying_flag=1.0)
        elif is_door_cue:
            obs = _make_obs(self.np_random, self.pos, tile_type=3, color=self.door_color)
        elif is_door_open:
            obs = _make_obs(self.np_random, self.pos, tile_type=4, color=self.door_color, door_open_flag=1.0)
        elif is_door_decoy:
            obs = _make_obs(self.np_random, self.pos, tile_type=4, color=self.door_color, door_open_flag=0.0)
        else:
            obs = _make_obs(self.np_random, self.pos, tile_type=0)

        is_distractor = not (is_novel_probe or is_key_pickup or is_door_cue or is_door_open or is_door_decoy)
        info = {
            "perm_id": self.perm_id,
            "is_novel_probe": is_novel_probe,
            "is_key_pickup": is_key_pickup,
            "is_door_cue": is_door_cue,
            "is_door_open": is_door_open,
            "is_door_decoy": is_door_decoy,
            "is_distractor": is_distractor,
            "door_matches": self.door_matches,
        }
        return obs, reward, False, (self.t >= self.max_t), info


class ScrambledControllerEnv(gym.Env):
    """Mid-episode action permutation flip environment."""
    metadata = {"render_modes": []}

    def __init__(self, max_t: int = 80, fixed_perm_id: int | None = None):
        super().__init__()
        self.max_t = int(max_t)
        self.fixed_perm_id = fixed_perm_id
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(16,), dtype=np.float32)

        self.t = 0
        self.pos = np.zeros(2, dtype=np.int32)
        self.perm_1 = 0
        self.perm_2 = 1
        self.perm_id = 0
        self.t_flip = 32
        self.probed_actions = set()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        split = options.get("split", "train") if options else "train"
        self.perm_1 = self.fixed_perm_id if self.fixed_perm_id is not None else _sample_perm(self.np_random, split)
        self.perm_2 = _sample_perm(self.np_random, split, exclude_perm=self.perm_1)
        self.perm_id = self.perm_1
        self.t_flip = int(self.np_random.integers(25, 41))
        self.t = 0
        self.pos = self.np_random.integers(0, 5, size=2, dtype=np.int32)
        self.probed_actions = set()

        obs = _make_obs(self.np_random, self.pos, tile_type=0)
        info = {
            "perm_id": self.perm_1,
            "perm_2": self.perm_2,
            "t_flip": self.t_flip,
            "is_flip_step": False,
            "is_novel_probe": False,
            "is_key_pickup": False,
            "is_door_cue": False,
            "is_door_open": False,
            "is_door_decoy": False,
            "is_distractor": True,
        }
        return obs, info

    def step(self, action):
        action = int(action)
        is_flip_step = (self.t == self.t_flip)
        if is_flip_step:
            self.probed_actions = set()
        active_perm = self.perm_1 if self.t < self.t_flip else self.perm_2
        self.perm_id = active_perm

        is_novel_probe = action not in self.probed_actions
        self.probed_actions.add(action)

        actual_dir = ALL_PERMUTATIONS[active_perm][action]
        self.pos = np.clip(self.pos + MOVE_DELTAS[actual_dir], 0, 4)
        self.t += 1

        obs = _make_obs(self.np_random, self.pos, tile_type=0)
        info = {
            "perm_id": active_perm,
            "perm_2": self.perm_2,
            "t_flip": self.t_flip,
            "is_flip_step": is_flip_step,
            "is_novel_probe": is_novel_probe,
            "is_key_pickup": False,
            "is_door_cue": False,
            "is_door_open": False,
            "is_door_decoy": False,
            "is_distractor": not is_novel_probe,
        }
        return obs, 0.0, False, (self.t >= self.max_t), info


class MultiStageKeyDoorEnv(gym.Env):
    """Three sequential (key, door) stages in a single 140-step episode to test eviction."""
    metadata = {"render_modes": []}

    def __init__(self, num_stages: int = 3, delay_L: int = 24, max_t: int = 140, fixed_perm_id: int | None = None):
        super().__init__()
        self.num_stages = num_stages
        self.delay_L = delay_L
        self.max_t = max_t
        self.fixed_perm_id = fixed_perm_id

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(16,), dtype=np.float32)

        self.t = 0
        self.pos = np.zeros(2, dtype=np.int32)
        self.perm_id = 0
        self.stages = []
        self.current_key_color = None
        self.probed_actions = set()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        split = options.get("split", "train") if options else "train"
        self.perm_id = self.fixed_perm_id if self.fixed_perm_id is not None else _sample_perm(self.np_random, split)

        self.t = 0
        self.pos = self.np_random.integers(0, 5, size=2, dtype=np.int32)
        self.current_key_color = None
        self.probed_actions = set()

        # Schedule 3 non-overlapping stages across 140 steps
        self.stages = []
        base_starts = [6, 50, 94]
        for s_idx in range(self.num_stages):
            k_step = int(base_starts[s_idx] + self.np_random.integers(0, 6))
            d_step = k_step + int(self.delay_L + self.np_random.integers(-4, 5))
            k_col = int(self.np_random.integers(0, 4))
            matches = bool(self.np_random.integers(0, 2) == 1)
            if matches:
                d_col = k_col
            else:
                others = [c for c in range(4) if c != k_col]
                d_col = int(others[int(self.np_random.integers(0, 3))])
            self.stages.append({
                "key_step": k_step,
                "door_cue_step": d_step,
                "key_color": k_col,
                "door_color": d_col,
                "matches": matches,
            })

        obs = _make_obs(self.np_random, self.pos, tile_type=0)
        info = {
            "perm_id": self.perm_id,
            "is_novel_probe": False,
            "is_key_pickup": False,
            "is_door_cue": False,
            "is_door_open": False,
            "is_door_decoy": False,
            "is_distractor": True,
        }
        return obs, info

    def step(self, action):
        action = int(action)
        is_novel_probe = action not in self.probed_actions
        self.probed_actions.add(action)

        actual_dir = ALL_PERMUTATIONS[self.perm_id][action]
        self.pos = np.clip(self.pos + MOVE_DELTAS[actual_dir], 0, 4)

        prev_t = self.t
        self.t += 1

        is_key_pickup = False
        is_door_cue = False
        is_door_open = False
        is_door_decoy = False
        active_color = None
        reward = 0.0

        for st in self.stages:
            if self.t == st["key_step"]:
                is_key_pickup = True
                self.current_key_color = st["key_color"]
                active_color = st["key_color"]
            elif self.t == st["door_cue_step"]:
                is_door_cue = True
                active_color = st["door_color"]
            elif prev_t == st["door_cue_step"]:
                active_color = st["door_color"]
                if st["matches"] and self.current_key_color == st["key_color"]:
                    is_door_open = True
                    reward = 1.0
                else:
                    is_door_decoy = True
                self.current_key_color = None

        if is_key_pickup:
            obs = _make_obs(self.np_random, self.pos, tile_type=2, color=active_color, carrying_flag=1.0)
        elif is_door_cue:
            obs = _make_obs(self.np_random, self.pos, tile_type=3, color=active_color)
        elif is_door_open:
            obs = _make_obs(self.np_random, self.pos, tile_type=4, color=active_color, door_open_flag=1.0)
        elif is_door_decoy:
            obs = _make_obs(self.np_random, self.pos, tile_type=4, color=active_color, door_open_flag=0.0)
        else:
            obs = _make_obs(self.np_random, self.pos, tile_type=0)

        is_distractor = not (is_novel_probe or is_key_pickup or is_door_cue or is_door_open or is_door_decoy)
        info = {
            "perm_id": self.perm_id,
            "is_novel_probe": is_novel_probe,
            "is_key_pickup": is_key_pickup,
            "is_door_cue": is_door_cue,
            "is_door_open": is_door_open,
            "is_door_decoy": is_door_decoy,
            "is_distractor": is_distractor,
        }
        return obs, reward, False, (self.t >= self.max_t), info


def make_env(task_name: str, fixed_perm_id: int | None = None) -> gym.Env:
    spec = TASK_SPECS[task_name]
    if spec["kind"] == "key_door":
        return KeyDoorScalingEnv(delay_L=spec["delay_L"], max_t=spec["episode_len"], fixed_perm_id=fixed_perm_id)
    elif spec["kind"] == "scrambled":
        return ScrambledControllerEnv(max_t=spec["episode_len"], fixed_perm_id=fixed_perm_id)
    elif spec["kind"] == "multistage":
        return MultiStageKeyDoorEnv(delay_L=spec["delay_L"], max_t=spec["episode_len"], fixed_perm_id=fixed_perm_id)
    raise ValueError(f"Unknown task: {task_name}")


def collect_task_dataset(task_name: str, num_episodes: int = 1000, split: str = "train", seed: int = 42):
    """Collect episodes using a mixed exploratory + heuristic probe policy."""
    spec = TASK_SPECS[task_name]
    T = spec["episode_len"]
    rng = np.random.default_rng(seed)

    all_obs = np.zeros((num_episodes, T, 16), dtype=np.float32)
    all_act = np.zeros((num_episodes, T), dtype=np.int64)
    all_rew = np.zeros((num_episodes, T), dtype=np.float32)
    all_novel = np.zeros((num_episodes, T), dtype=bool)
    all_pickup = np.zeros((num_episodes, T), dtype=bool)
    all_cue = np.zeros((num_episodes, T), dtype=bool)
    all_door = np.zeros((num_episodes, T), dtype=bool)
    all_decoy = np.zeros((num_episodes, T), dtype=bool)
    all_dist = np.zeros((num_episodes, T), dtype=bool)
    all_t_flip = np.full((num_episodes,), -1, dtype=np.int64)
    all_perm = np.zeros((num_episodes,), dtype=np.int64)

    env = make_env(task_name)
    for ep in range(num_episodes):
        obs, info = env.reset(seed=int(rng.integers(0, 2_000_000)), options={"split": split})
        all_perm[ep] = info["perm_id"]
        if "t_flip" in info:
            all_t_flip[ep] = info["t_flip"]

        # Mixed random + heuristic directional sweep
        sweep_dir = int(rng.integers(0, 4))
        for step_i in range(T):
            all_obs[ep, step_i] = obs
            if step_i < 4:
                act = step_i
            elif rng.random() < 0.30:
                if step_i % 4 == 0:
                    sweep_dir = (sweep_dir + 1) % 4
                act = sweep_dir
            else:
                act = int(rng.integers(0, 4))

            next_obs, rew, _, _, step_info = env.step(act)
            all_act[ep, step_i] = act
            all_rew[ep, step_i] = rew
            all_novel[ep, step_i] = step_info["is_novel_probe"]
            all_pickup[ep, step_i] = step_info["is_key_pickup"]
            all_cue[ep, step_i] = step_info["is_door_cue"]
            all_door[ep, step_i] = step_info["is_door_open"]
            all_decoy[ep, step_i] = step_info["is_door_decoy"]
            all_dist[ep, step_i] = step_info["is_distractor"]
            obs = next_obs

    return {
        "task": task_name,
        "episode_len": T,
        "obs": torch.from_numpy(all_obs),
        "act": torch.from_numpy(all_act),
        "rew": torch.from_numpy(all_rew),
        "is_novel_probe": torch.from_numpy(all_novel),
        "is_key_pickup": torch.from_numpy(all_pickup),
        "is_door_cue": torch.from_numpy(all_cue),
        "is_door_open": torch.from_numpy(all_door),
        "is_door_decoy": torch.from_numpy(all_decoy),
        "is_distractor": torch.from_numpy(all_dist),
        "t_flip": torch.from_numpy(all_t_flip),
        "perm_id": torch.from_numpy(all_perm),
    }
