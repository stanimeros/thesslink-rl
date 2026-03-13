"""
Cost components and POI suggestion helpers.

- astar_distance: A* pathfinding distance (respects obstacles)
- cost_components: cost components for one POI (A* distances)
- cost_optimal_baseline: POI that minimizes cost
"""
from typing import List, Tuple

import heapq
import numpy as np

# Weights: (w_travel_effort_agent, w_travel_effort_human, w_energy, w_privacy, w_time_to_meet)
# Prioritize human comfort: travel_effort_human gets highest weight.
DEFAULT_WEIGHTS = (0.20, 0.35, 0.10, 0.10, 0.25)


def _manhattan(pos_a: Tuple[int, int], pos_b: Tuple[int, int]) -> float:
    return abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])


def astar_distance(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    obstacles: frozenset[Tuple[int, int]],
    grid_size: Tuple[int, int] = (64, 64),
) -> float:
    """
    A* shortest path distance from start to goal avoiding obstacles.
    Returns actual step count, or float('inf') if no path exists.
    """
    if start == goal:
        return 0.0
    rows, cols = grid_size
    open_heap: list[Tuple[float, Tuple[int, int]]] = []
    heapq.heappush(open_heap, (_manhattan(start, goal), start))
    g_score: dict[Tuple[int, int], float] = {start: 0.0}

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal:
            return g_score[goal]
        r, c = current
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nb = (r + dr, c + dc)
            if not (0 <= nb[0] < rows and 0 <= nb[1] < cols):
                continue
            if nb in obstacles:
                continue
            tentative = g_score[current] + 1.0
            if tentative < g_score.get(nb, float("inf")):
                g_score[nb] = tentative
                f = tentative + _manhattan(nb, goal)
                heapq.heappush(open_heap, (f, nb))
    return float("inf")


def cost_components(
    poi: Tuple[int, int],
    agent_pos: Tuple[int, int],
    human_pos: Tuple[int, int],
    obstacles: frozenset[Tuple[int, int]],
    grid_size: Tuple[int, int] = (64, 64),
) -> Tuple[float, float, float, float, float]:
    """
    Cost components for one POI: (te_agent, te_human, energy, privacy, ttm).
    Uses A* distances.
    """
    max_dist = float(grid_size[0] + grid_size[1])
    dist_a = min(astar_distance(agent_pos, poi, obstacles, grid_size), max_dist)
    dist_h = min(astar_distance(human_pos, poi, obstacles, grid_size), max_dist)
    te_a = dist_a / max_dist
    te_h = dist_h / max_dist
    energy = 0.2 + 0.6 * te_h
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
