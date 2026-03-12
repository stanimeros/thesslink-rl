"""
Gymnasium environments for POI suggestion and navigation.

PoISuggestionEnv  — single-step: agent picks best POI (legacy, kept for comparison)
PoINavigationEnv  — multi-step: two cooperative agents navigate to a shared POI
"""
from __future__ import annotations

from typing import FrozenSet

import gymnasium as gym
import numpy as np

from cost_function import (
    astar_distance,
    cost_components,
    cost_function,
    DEFAULT_WEIGHTS,
    nearest_human_baseline,
)


def build_observation(
    human_pos: tuple[int, int],
    agent_pos: tuple[int, int],
    pois: list[tuple[int, int]],
    grid_size: tuple[int, int] = (64, 64),
) -> np.ndarray:
    """Build observation vector for RL policy. Same format as PoISuggestionEnv._get_obs."""
    parts = []
    for poi in pois:
        comps = cost_components(poi, agent_pos, human_pos, grid_size)
        parts.append(np.array(comps, dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)


class PoISuggestionEnv(gym.Env):
    """
    RL environment for learning which POI to suggest.

    Each episode: sample random (human_pos, agent_pos, 3 POIs).
    Agent chooses one POI. Reward = -cost (Travel Effort, energy expenditure, privacy, Time-to-Meet).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        grid_size: tuple[int, int] = (64, 64),
        weights: tuple[float, ...] | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.rows, self.cols = grid_size
        self.weights = DEFAULT_WEIGHTS if weights is None else weights

        # State: cost_components for each POI (travel_effort_agent, travel_effort_human, energy, privacy, time_to_meet) * 3 POIs = 15 floats
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(15,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(3)

        self._rng = np.random.default_rng(seed)
        self._human_pos: tuple[int, int] | None = None
        self._agent_pos: tuple[int, int] | None = None
        self._pois: list[tuple[int, int]] | None = None

    def _sample_scenario(self) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]]:
        """Sample distinct positions for human, agent, and 3 POIs."""
        cells = [(r, c) for r in range(self.rows) for c in range(self.cols)]
        chosen = self._rng.choice(len(cells), size=5, replace=False)
        positions = [cells[j] for j in chosen]
        return positions[0], positions[1], positions[2:5]

    def _get_obs(self) -> np.ndarray:
        """Build observation vector from current state."""
        assert self._human_pos is not None and self._agent_pos is not None and self._pois is not None
        parts = []
        for poi in self._pois:
            comps = cost_components(poi, self._agent_pos, self._human_pos, self.grid_size)
            parts.append(np.array(comps, dtype=np.float32))
        return np.concatenate(parts).astype(np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._human_pos, self._agent_pos, self._pois = self._sample_scenario()
        return self._get_obs(), {}

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._pois is None:
            raise RuntimeError("Call reset() before step()")
        action = int(np.clip(action, 0, 2))
        chosen_poi = self._pois[action]

        cost = cost_function(
            chosen_poi,
            self._agent_pos,
            self._human_pos,
            self.grid_size,
            *self.weights,
        )
        reward = -cost

        terminated = True
        truncated = False
        info = {
            "chosen_poi": chosen_poi,
            "reward": reward,
            "human_pos": self._human_pos,
            "agent_pos": self._agent_pos,
            "pois": self._pois.copy(),
        }
        return self._get_obs(), reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Cooperative multi-agent navigation environment
# ---------------------------------------------------------------------------

# Movement deltas: NONE, NORTH, SOUTH, WEST, EAST
_MOVES = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]


def _build_nav_obs(
    self_pos: tuple[int, int],
    other_pos: tuple[int, int],
    pois: list[tuple[int, int]],
    obstacles: FrozenSet[tuple[int, int]],
    grid_size: tuple[int, int],
) -> np.ndarray:
    """Build 19-float observation for one agent in PoINavigationEnv."""
    rows, cols = grid_size
    self_norm = np.array([self_pos[0] / (rows - 1), self_pos[1] / (cols - 1)], dtype=np.float32)
    other_norm = np.array([other_pos[0] / (rows - 1), other_pos[1] / (cols - 1)], dtype=np.float32)
    parts = [self_norm, other_norm]
    for poi in pois:
        comps = cost_components(poi, self_pos, other_pos, grid_size, obstacles)
        parts.append(np.array(comps, dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)


class PoINavigationEnv(gym.Env):
    """
    Cooperative multi-agent navigation environment.

    Two symmetric agents (agent1, agent2) share a policy and must navigate
    to the same target POI on a grid with static obstacles.

    Observation (per agent, 19 floats):
        self_pos (2) + other_pos (2) + A* cost_components × 3 POIs (15)
    Action: Discrete(5) — NONE, NORTH, SOUTH, WEST, EAST
    Reward: shared — step penalty + progress bonus + terminal reward
    Episode ends when both agents reach target POI or max_steps exceeded.
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
            # Quick connectivity check via flood fill from first non-obstacle cell
            if not self._is_connected(obstacles, occupied):
                obstacles.discard(cell)
        return frozenset(obstacles)

    def _is_connected(
        self,
        obstacles: set[tuple[int, int]],
        occupied: set[tuple[int, int]],
    ) -> bool:
        """BFS flood fill to verify all free cells (not obstacle, not occupied) are reachable."""
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
            for dr, dc in _MOVES[1:]:
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

    def _sample_free(
        self, n: int, exclude: set[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """Sample n distinct free (non-obstacle) positions."""
        free = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if (r, c) not in self._obstacles and (r, c) not in exclude
        ]
        idxs = self._rng.choice(len(free), size=n, replace=False)
        return [free[i] for i in idxs]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Sample positions first (no obstacles yet), then generate obstacles
        all_cells = [(r, c) for r in range(self.rows) for c in range(self.cols)]
        idxs = self._rng.choice(len(all_cells), size=5, replace=False)
        positions = [all_cells[i] for i in idxs]
        self._agent1_pos = positions[0]
        self._agent2_pos = positions[1]
        self._pois = positions[2:5]

        occupied = {self._agent1_pos, self._agent2_pos, *self._pois}
        self._obstacles = self._generate_obstacles(occupied)

        # Target POI = argmin A* cost (sum of both agents' distances)
        best_idx = 0
        best_cost = float("inf")
        for i, poi in enumerate(self._pois):
            c = cost_function(
                poi, self._agent1_pos, self._agent2_pos,
                self.grid_size, *self.weights, self._obstacles,
            )
            if c < best_cost:
                best_cost = c
                best_idx = i
        self._target_poi = self._pois[best_idx]

        self._step_count = 0
        self._prev_dist1 = astar_distance(self._agent1_pos, self._target_poi, self._obstacles, self.grid_size)
        self._prev_dist2 = astar_distance(self._agent2_pos, self._target_poi, self._obstacles, self.grid_size)

        return self._get_obs(), {}

    def _get_obs(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (obs_agent1, obs_agent2)."""
        obs1 = _build_nav_obs(self._agent1_pos, self._agent2_pos, self._pois, self._obstacles, self.grid_size)
        obs2 = _build_nav_obs(self._agent2_pos, self._agent1_pos, self._pois, self._obstacles, self.grid_size)
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

        dist1 = astar_distance(self._agent1_pos, self._target_poi, self._obstacles, self.grid_size)
        dist2 = astar_distance(self._agent2_pos, self._target_poi, self._obstacles, self.grid_size)
        max_dist = self.rows + self.cols

        # Progress bonus (normalized)
        progress = (self._prev_dist1 - dist1 + self._prev_dist2 - dist2) / max_dist
        self._prev_dist1 = dist1
        self._prev_dist2 = dist2

        # Step penalty
        reward = -0.01 + progress

        both_arrived = (self._agent1_pos == self._target_poi and self._agent2_pos == self._target_poi)
        terminated = both_arrived
        truncated = self._step_count >= self.max_steps

        if both_arrived:
            terminal_cost = cost_function(
                self._target_poi, self._agent1_pos, self._agent2_pos,
                self.grid_size, *self.weights, self._obstacles,
            )
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
