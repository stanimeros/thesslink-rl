"""
Cost components and POI suggestion helpers.

- astar_distance: A* pathfinding distance (respects obstacles)
- nearest_human_baseline: POI closest to human by A* distance
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


def nearest_human_baseline(
    pois: List[Tuple[int, int]],
    human_pos: Tuple[int, int],
    obstacles: frozenset[Tuple[int, int]],
    grid_size: Tuple[int, int] = (64, 64),
) -> int:
    """Return index of POI closest to human by A* distance."""
    distances = [astar_distance(human_pos, poi, obstacles, grid_size) for poi in pois]
    return int(np.argmin(distances))
