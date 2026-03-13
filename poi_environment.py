"""
Cooperative multi-agent navigation environment for ThessLink RL.

Two symmetric agents (agent1, agent2) share a policy. The policy chooses which
POI to navigate to (cost-optimal) and both agents navigate there. Target can
be re-selected each step.

Observation (per agent, 19 floats):
    self_pos (2) + other_pos (2) + cost_components × 3 POIs (15)
    Cost components use BFS distances; policy infers best POI from these.
Action: Discrete(15) — composite (target_idx, move): target_idx ∈ {0,1,2}, move ∈ {NONE,N,S,W,E}
Reward: shared — -cost - step_penalty for chosen target POI
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


def _bfs_dist_map(
    source: tuple[int, int],
    obstacles: FrozenSet[tuple[int, int]],
    rows: int,
    cols: int,
) -> dict[tuple[int, int], float]:
    """
    BFS from source. Returns {cell: distance} for all reachable free cells.
    Unreachable cells are absent (caller should use .get(pos, inf)).
    """
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


def _cost_components_from_maps(
    poi: tuple[int, int],
    self_map: dict[tuple[int, int], float],
    other_map: dict[tuple[int, int], float],
    max_dist: float,
) -> tuple[float, float, float, float, float]:
    """Compute cost components using pre-built BFS distance maps."""
    dist_s = min(self_map.get(poi, float("inf")), max_dist)
    dist_o = min(other_map.get(poi, float("inf")), max_dist)
    te_s = dist_s / max_dist
    te_o = dist_o / max_dist
    energy = 0.2 + 0.6 * te_o
    privacy = 1.0 - te_o
    ttm = max(te_s, te_o)
    return te_s, te_o, energy, privacy, ttm


def _build_nav_obs_bfs(
    self_pos: tuple[int, int],
    other_pos: tuple[int, int],
    poi_dist_maps: list[dict[tuple[int, int], float]],
    grid_size: tuple[int, int],
) -> np.ndarray:
    """
    Build 19-float observation: self_pos(2) + other_pos(2) + cost_components×3(15).
    Policy chooses target from cost components.
    """
    rows, cols = grid_size
    max_dist = float(rows + cols)
    self_norm = np.array([self_pos[0] / max(rows - 1, 1), self_pos[1] / max(cols - 1, 1)], dtype=np.float32)
    other_norm = np.array([other_pos[0] / max(rows - 1, 1), other_pos[1] / max(cols - 1, 1)], dtype=np.float32)
    parts = [self_norm, other_norm]
    for dist_map in poi_dist_maps:
        dist_s = min(dist_map.get(self_pos, float("inf")), max_dist)
        dist_o = min(dist_map.get(other_pos, float("inf")), max_dist)
        te_s = dist_s / max_dist
        te_o = dist_o / max_dist
        energy = 0.2 + 0.6 * te_o
        privacy = 1.0 - te_o
        ttm = max(te_s, te_o)
        parts.append(np.array([te_s, te_o, energy, privacy, ttm], dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)


class PoINavigationEnv(gym.Env):
    """
    Cooperative multi-agent navigation environment.

    Two symmetric agents (agent1, agent2) share a policy and must navigate
    to the same target POI on a grid with static obstacles.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        grid_size: tuple[int, int] = (64, 64),
        obstacle_density: float = 0.10,
        max_steps: int = 200,
        weights: tuple[float, ...] | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.rows, self.cols = grid_size
        self.obstacle_density = obstacle_density
        self.max_steps = max_steps
        self.weights = DEFAULT_WEIGHTS if weights is None else weights

        # Observation: self_pos(2) + other_pos(2) + cost_components*3(15) = 19
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(19,), dtype=np.float32
        )
        # Composite: target_idx*5 + move (target_idx 0-2, move NONE/N/S/W/E 0-4)
        self.action_space = gym.spaces.Discrete(15)

        self._rng = np.random.default_rng(seed)
        self._agent1_pos: tuple[int, int] = (0, 0)
        self._agent2_pos: tuple[int, int] = (0, 0)
        self._pois: list[tuple[int, int]] = []
        self._target_poi: tuple[int, int] = (0, 0)
        self._obstacles: FrozenSet[tuple[int, int]] = frozenset()
        self._step_count: int = 0

        # BFS distance maps from each POI — rebuilt once per episode
        self._poi_dist_maps: list[dict[tuple[int, int], float]] = []
        self._target_idx: int = 0
        self._init_agent1_pos: tuple[int, int] = (0, 0)
        self._init_agent2_pos: tuple[int, int] = (0, 0)

    # ------------------------------------------------------------------
    # Obstacle generation with connectivity guarantee
    # ------------------------------------------------------------------

    def _generate_obstacles(
        self, occupied: set[tuple[int, int]]
    ) -> FrozenSet[tuple[int, int]]:
        """
        Generate random obstacles with a single connectivity check at the end.
        Place obstacles randomly, then flood-fill from a free cell; any free cell
        not reached is converted back to free (obstacle removed) to guarantee
        full connectivity. Occupied cells (agents, POIs) are never blocked.
        """
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

        # Single BFS flood fill from any free cell to find disconnected free cells
        free_start = next(
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if (r, c) not in obstacles
        )
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

        # Remove obstacles that isolated free cells (restore connectivity)
        all_free = {
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if (r, c) not in obstacles
        }
        isolated = all_free - visited
        obstacles -= isolated

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

        # Build BFS distance maps from each POI (policy chooses target each step)
        self._poi_dist_maps = [
            _bfs_dist_map(poi, self._obstacles, self.rows, self.cols) for poi in self._pois
        ]
        self._step_count = 0
        self._target_poi = self._pois[0]  # placeholder until first step
        self._target_idx = 0

        return self._get_obs(), {}

    def _get_obs(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (obs_agent1, obs_agent2) using BFS distances (no target_onehot)."""
        obs1 = _build_nav_obs_bfs(
            self._agent1_pos, self._agent2_pos,
            self._poi_dist_maps, self.grid_size
        )
        obs2 = _build_nav_obs_bfs(
            self._agent2_pos, self._agent1_pos,
            self._poi_dist_maps, self.grid_size
        )
        return obs1, obs2

    def _try_move(self, pos: tuple[int, int], action: int) -> tuple[int, int]:
        """Apply action, return new position (stays if wall/obstacle)."""
        dr, dc = _MOVES[action]
        nr, nc = pos[0] + dr, pos[1] + dc
        if 0 <= nr < self.rows and 0 <= nc < self.cols and (nr, nc) not in self._obstacles:
            return (nr, nc)
        return pos

    def step(
        self,
        actions: tuple[int, int],
    ) -> tuple[tuple[np.ndarray, np.ndarray], float, bool, bool, dict]:
        """
        actions: (action_agent1, action_agent2) — composite (target_idx, move)
        Returns: (obs_pair, shared_reward, terminated, truncated, info)
        """
        a1, a2 = int(actions[0]), int(actions[1])
        target_idx = min(a1 // 5, 2)  # policy chooses target (use agent1's)
        move1 = a1 % 5
        move2 = a2 % 5

        self._target_idx = target_idx
        self._target_poi = self._pois[target_idx]

        self._agent1_pos = self._try_move(self._agent1_pos, move1)
        self._agent2_pos = self._try_move(self._agent2_pos, move2)
        self._step_count += 1

        # Distance to chosen target from _poi_dist_maps
        max_dist = float(self.rows + self.cols)
        target_map = self._poi_dist_maps[target_idx]
        dist1 = min(target_map.get(self._agent1_pos, float("inf")), max_dist)
        dist2 = min(target_map.get(self._agent2_pos, float("inf")), max_dist)

        # Cost-based reward for chosen target
        te_a = dist1 / max_dist
        te_h = dist2 / max_dist
        energy = 0.2 + 0.6 * te_h
        privacy = 1.0 - te_h
        ttm = max(te_a, te_h)
        cost = sum(w * v for w, v in zip(self.weights, (te_a, te_h, energy, privacy, ttm)))
        reward = -cost - 0.01  # step penalty
        reward = float(np.clip(reward, -10.0, 10.0))

        both_arrived = (self._agent1_pos == self._target_poi and self._agent2_pos == self._target_poi)
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
