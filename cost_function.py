"""
Cost components and POI suggestion helpers.

- bfs_distance: BFS pathfinding distance (respects obstacles)
- cost_components: cost components for one POI
- cost_optimal_baseline: POI that minimizes cost

Uses BFS everywhere for consistency with the RL environment.

Cost components (all normalized to [0, 1]):
  te_a    — travel effort agent (robot):  dist_agent / max_dist
  te_h    — travel effort human:          dist_human / max_dist
  energy  — combined locomotion energy:   0.6 * te_a + 0.4 * te_h
            (robot weighs more: batteries vs human effort)
  privacy — location-disclosure risk:     1 - te_h
            POI near the human → reveals their location → high cost.
            POI far from the human → protects privacy → low cost.
  ttm     — time to meet:                max(te_a, te_h)
"""
from __future__ import annotations

from collections import deque
from typing import List, Tuple

import numpy as np

# Weights: (w_travel_effort_agent, w_travel_effort_human, w_energy, w_privacy, w_time_to_meet)
DEFAULT_WEIGHTS = (0.15, 0.25, 0.15, 0.15, 0.30)

_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def bfs_distance(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    obstacles: frozenset[Tuple[int, int]],
    grid_size: Tuple[int, int] = (64, 64),
) -> float:
    """BFS shortest path distance from start to goal avoiding obstacles."""
    if start == goal:
        return 0.0
    rows, cols = grid_size
    visited = {start}
    queue: deque[Tuple[Tuple[int, int], float]] = deque([(start, 0.0)])
    while queue:
        (r, c), d = queue.popleft()
        for dr, dc in _DIRS:
            nb = (r + dr, c + dc)
            if nb == goal:
                return d + 1.0
            if 0 <= nb[0] < rows and 0 <= nb[1] < cols and nb not in obstacles and nb not in visited:
                visited.add(nb)
                queue.append((nb, d + 1.0))
    return float("inf")


def cost_components(
    poi: Tuple[int, int],
    agent_pos: Tuple[int, int],
    human_pos: Tuple[int, int],
    obstacles: frozenset[Tuple[int, int]],
    grid_size: Tuple[int, int] = (64, 64),
) -> Tuple[float, float, float, float, float]:
    """Cost components for one POI: (te_agent, te_human, energy, privacy, ttm)."""
    max_dist = float(grid_size[0] + grid_size[1])
    dist_a = min(bfs_distance(agent_pos, poi, obstacles, grid_size), max_dist)
    dist_h = min(bfs_distance(human_pos, poi, obstacles, grid_size), max_dist)
    te_a = dist_a / max_dist
    te_h = dist_h / max_dist
    energy = 0.6 * te_a + 0.4 * te_h
    privacy = 1.0 - te_h
    ttm = max(te_a, te_h)
    return te_a, te_h, energy, privacy, ttm


def cost_optimal_baseline(
    pois: List[Tuple[int, int]],
    agent_pos: Tuple[int, int],
    human_pos: Tuple[int, int],
    obstacles: frozenset[Tuple[int, int]],
    weights: Tuple[float, ...] = DEFAULT_WEIGHTS,
    grid_size: Tuple[int, int] = (64, 64),
) -> int:
    """Return index of POI that minimizes cost."""
    costs = []
    for poi in pois:
        comps = cost_components(poi, agent_pos, human_pos, obstacles, grid_size)
        costs.append(sum(w * c for w, c in zip(weights, comps)))
    return int(np.argmin(costs))
