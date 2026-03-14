"""
Cooperative multi-agent navigation environment for ThessLink RL.

Centralized controller design: a single "god-camera" model sees the full
global state and outputs a joint action that controls both agents
independently while selecting a shared target.

Observation (35 floats — relative, position-invariant):
    delta_a1_to_pois (6) + delta_a2_to_pois (6) + wall_bits_a1 (4)
    + wall_bits_a2 (4) + cost_components × 3 POIs (15)
    Relative vectors (in [-1,1]) let the policy generalize across positions.
    Cost components (in [0,1]) use BFS distances for target selection.

Action: MultiDiscrete([3, 5, 5])
    [0] target_idx ∈ {0,1,2}  — which POI both agents navigate to
    [1] move1      ∈ {0..4}   — agent1 movement (NONE,N,S,W,E)
    [2] move2      ∈ {0..4}   — agent2 movement (NONE,N,S,W,E)

Reward: shared
    +progress            — reward for each step that reduces total BFS distance
    -STEP_PENALTY        — per-step penalty = TERMINAL_BONUS / (2 * max_steps)
    -SWITCH_PENALTY      — target-switch penalty = TERMINAL_BONUS * 0.05 per switch
    -oscillation_penalty — 2× STEP_PENALTY per agent that returns to its previous cell
    -cost_penalty        — weighted cost of current target × COST_PENALTY_SCALE
    +near_bonus          — small bonus when one agent is at POI and other is 1-2 steps away
    +TERMINAL_BONUS      — bonus (5.0) when both agents reach the target
Episode ends when both agents reach chosen target or max_steps exceeded.
"""
from __future__ import annotations

from collections import deque
from typing import FrozenSet

import gymnasium as gym
import numpy as np

from cost_function import DEFAULT_WEIGHTS

# Movement deltas: NONE, NORTH, SOUTH, WEST, EAST
_MOVES = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS = _MOVES[1:]  # cardinal directions only

_OBS_DIM = 35


def _wall_bits(
    pos: tuple[int, int],
    obstacles: FrozenSet[tuple[int, int]],
    rows: int,
    cols: int,
) -> list[float]:
    """Return 4 floats (N, S, W, E): 1.0 if direction is blocked, 0.0 otherwise."""
    return [
        1.0 if not (0 <= pos[0] + dr < rows and 0 <= pos[1] + dc < cols)
               or (pos[0] + dr, pos[1] + dc) in obstacles
        else 0.0
        for dr, dc in _DIRS
    ]


def _bfs_dist_map(
    source: tuple[int, int],
    obstacles: FrozenSet[tuple[int, int]],
    rows: int,
    cols: int,
) -> dict[tuple[int, int], float]:
    """BFS from source. Returns {cell: distance} for all reachable free cells."""
    dist: dict[tuple[int, int], float] = {source: 0.0}
    queue: deque[tuple[int, int]] = deque([source])
    while queue:
        r, c = queue.popleft()
        d = dist[(r, c)]
        for dr, dc in _DIRS:
            nb = (r + dr, c + dc)
            if 0 <= nb[0] < rows and 0 <= nb[1] < cols and nb not in obstacles and nb not in dist:
                dist[nb] = d + 1.0
                queue.append(nb)
    return dist


def _build_global_obs(
    agent1_pos: tuple[int, int],
    agent2_pos: tuple[int, int],
    pois: list[tuple[int, int]],
    poi_dist_maps: list[dict[tuple[int, int], float]],
    grid_size: tuple[int, int],
    obstacles: FrozenSet[tuple[int, int]],
) -> np.ndarray:
    """
    Build 35-float relative observation:
      delta_a1_to_pois(6) + delta_a2_to_pois(6) + wall_bits_a1(4)
      + wall_bits_a2(4) + cost_components×3_POIs(15)

    Relative vectors make the policy position-invariant: it sees
    "POI is 3 cells north" instead of "agent at (2,4), POI at (5,4)".
    """
    rows, cols = grid_size
    max_dist = float(rows + cols)
    scale_r, scale_c = max(rows - 1, 1), max(cols - 1, 1)

    # Relative vectors: agent → each POI, normalized to [-1, 1]
    deltas_a1 = np.array(
        [v for poi in pois for v in ((poi[0] - agent1_pos[0]) / scale_r,
                                     (poi[1] - agent1_pos[1]) / scale_c)],
        dtype=np.float32,
    )
    deltas_a2 = np.array(
        [v for poi in pois for v in ((poi[0] - agent2_pos[0]) / scale_r,
                                     (poi[1] - agent2_pos[1]) / scale_c)],
        dtype=np.float32,
    )

    walls_a1 = np.array(_wall_bits(agent1_pos, obstacles, rows, cols), dtype=np.float32)
    walls_a2 = np.array(_wall_bits(agent2_pos, obstacles, rows, cols), dtype=np.float32)

    parts = [deltas_a1, deltas_a2, walls_a1, walls_a2]
    for dist_map in poi_dist_maps:
        dist_a = min(dist_map.get(agent1_pos, float("inf")), max_dist)
        dist_h = min(dist_map.get(agent2_pos, float("inf")), max_dist)
        te_a = dist_a / max_dist
        te_h = dist_h / max_dist
        energy = 0.6 * te_a + 0.4 * te_h
        privacy = 1.0 - te_h
        ttm = max(te_a, te_h)
        parts.append(np.array([te_a, te_h, energy, privacy, ttm], dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)


class PoINavigationEnv(gym.Env):
    """
    Centralized cooperative navigation environment.

    A single "god-camera" controller observes the full global state and
    outputs a joint action: target POI + independent moves for both agents.
    Both agents navigate to the same chosen POI.

    Action space: MultiDiscrete([3, 5, 5])
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        grid_size: tuple[int, int] = (64, 64),
        obstacle_density: float = 0.10,
        max_steps: int = 200,
        weights: tuple[float, ...] | None = None,
        seed: int | None = None,
        terminal_bonus: float = 5.0,
        switch_penalty_frac: float = 0.05,
        progress_scale: float = 2.0,
        cost_penalty_scale: float = 0.3,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.rows, self.cols = grid_size
        self.obstacle_density = obstacle_density
        self.max_steps = max_steps
        self.weights = DEFAULT_WEIGHTS if weights is None else weights

        self.TERMINAL_BONUS = terminal_bonus
        self.STEP_PENALTY = terminal_bonus / (2.0 * max_steps)
        self.SWITCH_PENALTY = terminal_bonus * switch_penalty_frac
        self.PROGRESS_SCALE = progress_scale
        self.COST_PENALTY_SCALE = cost_penalty_scale

        # Observation: 35 floats — deltas in [-1,1], cost components in [0,1]
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(_OBS_DIM,), dtype=np.float32
        )
        # MultiDiscrete: [target_idx, move1, move2]
        self.action_space = gym.spaces.MultiDiscrete([3, 5, 5])

        self._rng = np.random.default_rng(seed)
        self._agent1_pos: tuple[int, int] = (0, 0)
        self._agent2_pos: tuple[int, int] = (0, 0)
        self._pois: list[tuple[int, int]] = []
        self._target_poi: tuple[int, int] = (0, 0)
        self._obstacles: FrozenSet[tuple[int, int]] = frozenset()
        self._step_count: int = 0

        self._poi_dist_maps: list[dict[tuple[int, int], float]] = []
        self._target_idx: int = 0
        self._prev_target_idx: int = -1
        self._init_agent1_pos: tuple[int, int] = (0, 0)
        self._init_agent2_pos: tuple[int, int] = (0, 0)
        self._prev_agent1_pos: tuple[int, int] = (0, 0)
        self._prev_agent2_pos: tuple[int, int] = (0, 0)

    # ------------------------------------------------------------------
    # Obstacle generation with connectivity guarantee
    # ------------------------------------------------------------------

    def _generate_obstacles(
        self, occupied: set[tuple[int, int]]
    ) -> FrozenSet[tuple[int, int]]:
        total = self.rows * self.cols
        n_obs = int(total * self.obstacle_density)
        candidates = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if (r, c) not in occupied
        ]
        chosen = self._rng.choice(len(candidates), size=min(n_obs, len(candidates)), replace=False)
        obstacles: set[tuple[int, int]] = {candidates[i] for i in chosen}

        free_start = self._agent1_pos
        visited: set[tuple[int, int]] = {free_start}
        queue: deque[tuple[int, int]] = deque([free_start])
        while queue:
            r, c = queue.popleft()
            for dr, dc in _DIRS:
                nb = (r + dr, c + dc)
                if (
                    0 <= nb[0] < self.rows
                    and 0 <= nb[1] < self.cols
                    and nb not in obstacles
                    and nb not in visited
                ):
                    visited.add(nb)
                    queue.append(nb)

        all_free = {
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if (r, c) not in obstacles
        }
        obstacles |= (all_free - visited) - occupied
        return frozenset(obstacles)

    # ------------------------------------------------------------------
    # Reset / Step
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        all_cells = [(r, c) for r in range(self.rows) for c in range(self.cols)]
        idxs = self._rng.choice(len(all_cells), size=5, replace=False)
        positions = [all_cells[i] for i in idxs]
        self._agent1_pos = positions[0]
        self._agent2_pos = positions[1]
        self._pois = positions[2:5]

        occupied = {self._agent1_pos, self._agent2_pos, *self._pois}
        self._obstacles = self._generate_obstacles(occupied)

        self._init_agent1_pos = self._agent1_pos
        self._init_agent2_pos = self._agent2_pos

        self._poi_dist_maps = [
            _bfs_dist_map(poi, self._obstacles, self.rows, self.cols) for poi in self._pois
        ]
        self._step_count = 0
        self._target_poi = self._pois[0]
        self._target_idx = 0
        self._prev_target_idx = -1
        self._prev_agent1_pos = self._agent1_pos
        self._prev_agent2_pos = self._agent2_pos

        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        return _build_global_obs(
            self._agent1_pos, self._agent2_pos,
            self._pois, self._poi_dist_maps, self.grid_size, self._obstacles,
        )

    def _try_move(self, pos: tuple[int, int], action: int) -> tuple[int, int]:
        dr, dc = _MOVES[action]
        nr, nc = pos[0] + dr, pos[1] + dc
        if 0 <= nr < self.rows and 0 <= nc < self.cols and (nr, nc) not in self._obstacles:
            return (nr, nc)
        return pos

    def step(
        self,
        action: np.ndarray | list | int,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        action: MultiDiscrete [target_idx, move1, move2]
                OR flat int (target_idx * 25 + move1 * 5 + move2) for backwards compat.
        """
        if isinstance(action, (int, np.integer)):
            a = int(action)
            target_idx = min(a // 25, 2)
            move1 = (a % 25) // 5
            move2 = a % 5
        else:
            arr = np.asarray(action, dtype=int).ravel()
            target_idx = int(arr[0])
            move1 = int(arr[1])
            move2 = int(arr[2])

        switch_penalty = self.SWITCH_PENALTY if (self._prev_target_idx != -1 and target_idx != self._prev_target_idx) else 0.0

        self._prev_target_idx = target_idx
        self._target_idx = target_idx
        self._target_poi = self._pois[target_idx]

        max_dist = float(self.rows + self.cols)
        target_map = self._poi_dist_maps[target_idx]

        prev_dist1 = min(target_map.get(self._agent1_pos, float("inf")), max_dist)
        prev_dist2 = min(target_map.get(self._agent2_pos, float("inf")), max_dist)

        old_a1, old_a2 = self._agent1_pos, self._agent2_pos
        if self._agent1_pos != self._target_poi:
            self._agent1_pos = self._try_move(self._agent1_pos, move1)
        if self._agent2_pos != self._target_poi:
            self._agent2_pos = self._try_move(self._agent2_pos, move2)
        self._step_count += 1

        dist1 = min(target_map.get(self._agent1_pos, float("inf")), max_dist)
        dist2 = min(target_map.get(self._agent2_pos, float("inf")), max_dist)

        te_a = dist1 / max_dist
        te_h = dist2 / max_dist
        energy = 0.6 * te_a + 0.4 * te_h
        privacy = 1.0 - te_h
        ttm = max(te_a, te_h)
        cost = sum(w * v for w, v in zip(self.weights, (te_a, te_h, energy, privacy, ttm)))

        progress = (prev_dist1 + prev_dist2 - dist1 - dist2) / (2.0 * max_dist)
        progress_reward = progress * self.PROGRESS_SCALE

        # Oscillation penalty: agent moved back to where it was 1 step ago
        osc = 0
        if self._agent1_pos == self._prev_agent1_pos and self._agent1_pos != self._target_poi:
            osc += 1
        if self._agent2_pos == self._prev_agent2_pos and self._agent2_pos != self._target_poi:
            osc += 1
        oscillation_penalty = osc * self.STEP_PENALTY * 2.0
        self._prev_agent1_pos = old_a1
        self._prev_agent2_pos = old_a2

        # Near-meeting bonus: one agent at POI, other is 1-2 steps away
        near_bonus = 0.0
        a1_at = self._agent1_pos == self._target_poi
        a2_at = self._agent2_pos == self._target_poi
        if a1_at != a2_at:
            remaining = dist2 if a1_at else dist1
            if remaining <= 2.0:
                near_bonus = self.TERMINAL_BONUS * 0.05 * (3.0 - remaining) / 2.0

        both_arrived = a1_at and a2_at
        terminal_bonus = self.TERMINAL_BONUS if both_arrived else 0.0

        cost_penalty = cost * self.COST_PENALTY_SCALE
        reward = (progress_reward - self.STEP_PENALTY - switch_penalty
                  - cost_penalty - oscillation_penalty + near_bonus + terminal_bonus)
        reward = float(np.clip(reward, -10.0, 10.0))

        terminated = both_arrived
        truncated = self._step_count >= self.max_steps

        info = {
            "agent1_pos": self._agent1_pos,
            "agent2_pos": self._agent2_pos,
            "target_poi": self._target_poi,
            "target_idx": self._target_idx,
            "pois": self._pois.copy(),
            "obstacles": self._obstacles,
            "step": self._step_count,
            "both_arrived": both_arrived,
            "cost": cost,
        }
        return self._get_obs(), reward, terminated, truncated, info


class FlatActionWrapper(gym.ActionWrapper):
    """Wraps MultiDiscrete([3,5,5]) env with Discrete(75) for DQN / Q-Learning."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(75)

    def action(self, act: int) -> np.ndarray:
        a = int(act)
        return np.array([a // 25, (a % 25) // 5, a % 5], dtype=int)
