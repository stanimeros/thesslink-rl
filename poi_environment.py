"""
Cooperative multi-agent navigation environment for ThessLink RL.

Two symmetric agents (agent1, agent2) share a policy and navigate to the same
target POI on a 64×64 grid with static obstacles.

Observation (per agent, 19 floats):
    self_pos (2) + other_pos (2) + cost_components × 3 POIs (15)
Action: Discrete(5) — NONE, NORTH, SOUTH, WEST, EAST
Reward: shared — step penalty + progress bonus + terminal reward
Episode ends when both agents reach target POI or max_steps exceeded.

Performance: BFS distance maps are computed once per episode (one per POI + one
for the target), so all distance lookups during step/obs are O(1).
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
    """
    Compute cost components using pre-built BFS distance maps.
    self_map  = BFS from self_pos  (distances to all cells, including poi)
    other_map = BFS from other_pos
    Returns (travel_effort_self, travel_effort_other, energy, privacy, time_to_meet).
    """
    dist_s = min(self_map.get(poi, float("inf")), max_dist)
    dist_o = min(other_map.get(poi, float("inf")), max_dist)
    te_s = dist_s / max_dist
    te_o = dist_o / max_dist
    energy = 0.2 + 0.6 * te_o
    privacy = 1.0 - te_o
    ttm = max(te_s, te_o)
    return te_s, te_o, energy, privacy, ttm


def _build_nav_obs(
    self_pos: tuple[int, int],
    other_pos: tuple[int, int],
    pois: list[tuple[int, int]],
    poi_dist_maps: list[dict[tuple[int, int], float]],
    grid_size: tuple[int, int],
) -> np.ndarray:
    """
    Build 19-float observation using pre-computed BFS distance maps.
    poi_dist_maps[i] = BFS distances from pois[i] to all cells.
    dist(self → poi) = poi_dist_maps[i][self_pos]  (same as dist(poi → self) on undirected grid)
    """
    rows, cols = grid_size
    max_dist = float(rows + cols)
    self_norm = np.array([self_pos[0] / (rows - 1), self_pos[1] / (cols - 1)], dtype=np.float32)
    other_norm = np.array([other_pos[0] / (rows - 1), other_pos[1] / (cols - 1)], dtype=np.float32)
    parts = [self_norm, other_norm]
    for poi_map in poi_dist_maps:
        # dist(self→poi) and dist(other→poi) from the poi's BFS map
        dist_s = min(poi_map.get(self_pos, float("inf")), max_dist)
        dist_o = min(poi_map.get(other_pos, float("inf")), max_dist)
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
        # NONE, NORTH, SOUTH, WEST, EAST
        self.action_space = gym.spaces.Discrete(5)

        self._rng = np.random.default_rng(seed)
        self._agent1_pos: tuple[int, int] = (0, 0)
        self._agent2_pos: tuple[int, int] = (0, 0)
        self._pois: list[tuple[int, int]] = []
        self._target_poi: tuple[int, int] = (0, 0)
        self._obstacles: FrozenSet[tuple[int, int]] = frozenset()
        self._step_count: int = 0
        self._prev_dist1: float = 0.0
        self._prev_dist2: float = 0.0

        # BFS distance maps — rebuilt once per episode
        # _target_dist_map: distances FROM target (for progress reward)
        # _poi_dist_maps: distances FROM each POI source (for observation cost_components)
        self._target_dist_map: dict[tuple[int, int], float] = {}
        self._poi_dist_maps: list[dict[tuple[int, int], float]] = []

    # ------------------------------------------------------------------
    # Obstacle generation with connectivity guarantee
    # ------------------------------------------------------------------

    def _generate_obstacles(
        self, occupied: set[tuple[int, int]]
    ) -> FrozenSet[tuple[int, int]]:
        """Generate random obstacles ensuring all non-obstacle cells are connected."""
        total = self.rows * self.cols
        n_obs = int(total * self.obstacle_density)
        all_cells = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if (r, c) not in occupied
        ]
        self._rng.shuffle(all_cells)
        obstacles: set[tuple[int, int]] = set()
        for cell in all_cells:
            if len(obstacles) >= n_obs:
                break
            obstacles.add(cell)
            if not self._is_connected(obstacles, occupied):
                obstacles.discard(cell)
        return frozenset(obstacles)

    def _is_connected(
        self,
        obstacles: set[tuple[int, int]],
        occupied: set[tuple[int, int]],
    ) -> bool:
        """BFS flood fill to verify all free cells are reachable."""
        free = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if (r, c) not in obstacles
        ]
        if not free:
            return False
        start = free[0]
        visited = {start}
        queue = [start]
        while queue:
            r, c = queue.pop()
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
        return len(visited) == len(free)

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

        # Build BFS maps from each POI (used for cost_components in observation)
        self._poi_dist_maps = [
            _bfs_dist_map(poi, self._obstacles, self.rows, self.cols)
            for poi in self._pois
        ]

        # Select target POI: argmin weighted cost using BFS distances
        # Build agent maps once (reused for all 3 POIs)
        max_dist = float(self.rows + self.cols)
        a1_map = _bfs_dist_map(self._agent1_pos, self._obstacles, self.rows, self.cols)
        a2_map = _bfs_dist_map(self._agent2_pos, self._obstacles, self.rows, self.cols)
        best_idx = 0
        best_cost = float("inf")
        for i, poi in enumerate(self._pois):
            te_a, te_h, energy, privacy, ttm = _cost_components_from_maps(poi, a1_map, a2_map, max_dist)
            c = sum(w * v for w, v in zip(self.weights, (te_a, te_h, energy, privacy, ttm)))
            if c < best_cost:
                best_cost = c
                best_idx = i
        self._target_poi = self._pois[best_idx]

        # Build BFS map from target (used for progress reward — O(1) lookup per step)
        self._target_dist_map = _bfs_dist_map(self._target_poi, self._obstacles, self.rows, self.cols)

        self._step_count = 0
        self._prev_dist1 = self._target_dist_map.get(self._agent1_pos, float("inf"))
        self._prev_dist2 = self._target_dist_map.get(self._agent2_pos, float("inf"))

        return self._get_obs(), {}

    def _get_obs(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (obs_agent1, obs_agent2) using cached BFS maps — O(1) per POI."""
        # For observation, cost_components(poi, self, other) needs:
        #   dist(self → poi) = poi_dist_map[self_pos]  (BFS from poi)
        #   dist(other → poi) = poi_dist_map[other_pos]
        obs1 = _build_nav_obs(
            self._agent1_pos, self._agent2_pos,
            self._pois, self._poi_dist_maps,
            self.grid_size,
        )
        obs2 = _build_nav_obs(
            self._agent2_pos, self._agent1_pos,
            self._pois, self._poi_dist_maps,
            self.grid_size,
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
        actions: (action_agent1, action_agent2)
        Returns: (obs_pair, shared_reward, terminated, truncated, info)
        """
        a1, a2 = int(actions[0]), int(actions[1])
        self._agent1_pos = self._try_move(self._agent1_pos, a1)
        self._agent2_pos = self._try_move(self._agent2_pos, a2)
        self._step_count += 1

        # O(1) distance lookup from cached BFS map
        dist1 = self._target_dist_map.get(self._agent1_pos, float("inf"))
        dist2 = self._target_dist_map.get(self._agent2_pos, float("inf"))
        max_dist = float(self.rows + self.cols)

        progress = (self._prev_dist1 - dist1 + self._prev_dist2 - dist2) / max_dist
        self._prev_dist1 = dist1
        self._prev_dist2 = dist2

        reward = -0.01 + progress

        both_arrived = (self._agent1_pos == self._target_poi and self._agent2_pos == self._target_poi)
        terminated = both_arrived
        truncated = self._step_count >= self.max_steps

        if both_arrived:
            # Terminal reward: negative cost at meeting point (O(1) — distances are 0)
            te_a = 0.0
            te_h = 0.0
            energy = 0.2
            privacy = 1.0
            ttm = 0.0
            terminal_cost = sum(w * v for w, v in zip(self.weights, (te_a, te_h, energy, privacy, ttm)))
            reward += -terminal_cost

        info = {
            "agent1_pos": self._agent1_pos,
            "agent2_pos": self._agent2_pos,
            "target_poi": self._target_poi,
            "pois": self._pois.copy(),
            "obstacles": self._obstacles,
            "step": self._step_count,
            "both_arrived": both_arrived,
        }
        return self._get_obs(), reward, terminated, truncated, info
