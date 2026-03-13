"""
Cooperative multi-agent navigation environment for ThessLink RL.

Centralized controller design: a single "god-camera" model sees the full
global state (both agent positions, all POIs, obstacles) and outputs a joint
action that controls both agents independently while selecting a shared target.

Observation (33 floats — global state):
    agent1_pos (2) + agent2_pos (2) + wall_bits_a1 (4) + wall_bits_a2 (4)
    + poi_positions (6) + cost_components × 3 POIs (15)
    Wall bits encode blocked N/S/W/E directions for each agent.
    POI positions give explicit direction vectors so the model can navigate.
    Cost components use BFS distances so the model can select the best POI.

Action: Discrete(75) — joint (target_idx, move1, move2)
    target_idx ∈ {0,1,2}  — which POI both agents navigate to
    move1      ∈ {0..4}   — agent1 movement (NONE,N,S,W,E)
    move2      ∈ {0..4}   — agent2 movement (NONE,N,S,W,E)
    Encoding:  action = target_idx * 25 + move1 * 5 + move2

Reward: shared
    -cost            — distance-based cost for the chosen target POI
    -0.01            — step penalty (discourages stalling)
    -0.10            — target-switch penalty (allows rerouting but filters noise;
                        only applies when target_idx changes from previous step)
    +5.00            — terminal bonus when both agents reach the target
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


def _build_global_obs(
    agent1_pos: tuple[int, int],
    agent2_pos: tuple[int, int],
    pois: list[tuple[int, int]],
    poi_dist_maps: list[dict[tuple[int, int], float]],
    grid_size: tuple[int, int],
    obstacles: FrozenSet[tuple[int, int]],
) -> np.ndarray:
    """
    Build 33-float global observation for the centralized controller:
      agent1_pos(2) + agent2_pos(2) + wall_bits_a1(4) + wall_bits_a2(4)
      + poi_positions(6) + cost_components×3_POIs(15)

    POI positions give the model explicit direction vectors so it can
    navigate toward a target (without them the model only knows scalar
    distance and cannot determine which way to move).

    Cost components per POI (5 floats each):
      te_a (agent1 BFS dist), te_h (agent2 BFS dist),
      energy, privacy, ttm
    """
    rows, cols = grid_size
    max_dist = float(rows + cols)
    a1_norm = np.array([agent1_pos[0] / max(rows - 1, 1), agent1_pos[1] / max(cols - 1, 1)], dtype=np.float32)
    a2_norm = np.array([agent2_pos[0] / max(rows - 1, 1), agent2_pos[1] / max(cols - 1, 1)], dtype=np.float32)
    walls_a1 = np.array(_wall_bits(agent1_pos, obstacles, rows, cols), dtype=np.float32)
    walls_a2 = np.array(_wall_bits(agent2_pos, obstacles, rows, cols), dtype=np.float32)
    # Normalized POI positions — lets the model compute direction to target
    poi_pos = np.array(
        [coord / max(dim - 1, 1) for poi in pois for coord, dim in zip(poi, (rows, cols))],
        dtype=np.float32,
    )
    parts = [a1_norm, a2_norm, walls_a1, walls_a2, poi_pos]
    for dist_map in poi_dist_maps:
        dist_a = min(dist_map.get(agent1_pos, float("inf")), max_dist)
        dist_h = min(dist_map.get(agent2_pos, float("inf")), max_dist)
        te_a = dist_a / max_dist
        te_h = dist_h / max_dist
        energy = 0.2 + 0.6 * te_h
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

        # Global observation: agent1_pos(2) + agent2_pos(2) + wall_bits×2(8)
        #                     + poi_positions(6) + cost_components×3_POIs(15) = 33
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(33,), dtype=np.float32
        )
        # Joint action: target_idx * 25 + move1 * 5 + move2
        # target_idx ∈ {0,1,2}, move1/move2 ∈ {NONE,N,S,W,E}
        self.action_space = gym.spaces.Discrete(75)

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
        self._prev_target_idx: int = -1  # -1 = no previous target (first step)
        self._init_agent1_pos: tuple[int, int] = (0, 0)
        self._init_agent2_pos: tuple[int, int] = (0, 0)

    # ------------------------------------------------------------------
    # Obstacle generation with connectivity guarantee
    # ------------------------------------------------------------------

    def _generate_obstacles(
        self, occupied: set[tuple[int, int]]
    ) -> FrozenSet[tuple[int, int]]:
        """
        Generate random obstacles with a connectivity guarantee.
        Place obstacles randomly, BFS from agent1's position (the connectivity
        anchor), then convert any unreachable free cells to obstacles so all
        remaining free cells form one connected component.
        Occupied cells (agents, POIs) are never made into obstacles.
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

        # BFS from an occupied cell (agent1) so the main component is anchored
        # to where agents and POIs live, not an arbitrary free cell.
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

        # Convert unreachable free cells to obstacles so every remaining free
        # cell is reachable from the agents/POIs (one connected component).
        all_free = {
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if (r, c) not in obstacles
        }
        isolated = all_free - visited
        obstacles |= isolated

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

        # Build BFS distance maps from each POI
        self._poi_dist_maps = [
            _bfs_dist_map(poi, self._obstacles, self.rows, self.cols) for poi in self._pois
        ]
        self._step_count = 0
        self._target_poi = self._pois[0]  # placeholder until first step
        self._target_idx = 0
        self._prev_target_idx = -1  # reset: no penalty on the first step

        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        """Returns the 33-float global observation."""
        return _build_global_obs(
            self._agent1_pos, self._agent2_pos,
            self._pois, self._poi_dist_maps, self.grid_size, self._obstacles,
        )

    def _try_move(self, pos: tuple[int, int], action: int) -> tuple[int, int]:
        """Apply action, return new position (stays if wall/obstacle)."""
        dr, dc = _MOVES[action]
        nr, nc = pos[0] + dr, pos[1] + dc
        if 0 <= nr < self.rows and 0 <= nc < self.cols and (nr, nc) not in self._obstacles:
            return (nr, nc)
        return pos

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        action: joint int encoding (target_idx * 25 + move1 * 5 + move2)
          target_idx — which POI both agents navigate to (0-2)
          move1      — agent1 movement direction (0-4: NONE,N,S,W,E)
          move2      — agent2 movement direction (0-4: NONE,N,S,W,E)

        Returns: (obs, shared_reward, terminated, truncated, info)
        """
        a = int(action)
        target_idx = min(a // 25, 2)
        move1 = (a % 25) // 5
        move2 = a % 5

        # Apply switch penalty before updating target
        switch_penalty = 0.1 if (self._prev_target_idx != -1 and target_idx != self._prev_target_idx) else 0.0

        self._prev_target_idx = target_idx
        self._target_idx = target_idx
        self._target_poi = self._pois[target_idx]

        self._agent1_pos = self._try_move(self._agent1_pos, move1)
        self._agent2_pos = self._try_move(self._agent2_pos, move2)
        self._step_count += 1

        # BFS distance to chosen target for each agent
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

        both_arrived = (self._agent1_pos == self._target_poi and self._agent2_pos == self._target_poi)
        terminal_bonus = 5.0 if both_arrived else 0.0

        reward = -cost - 0.01 - switch_penalty + terminal_bonus
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
