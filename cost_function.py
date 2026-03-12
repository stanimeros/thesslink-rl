"""
Cost components and POI suggestion helpers.

- cost_components, cost_function: used for observation features and reward computation
- astar_distance: A* pathfinding distance (respects obstacles)
- nearest_human_baseline: simple heuristic baseline (pick POI closest to human)
"""
from typing import Tuple, List, FrozenSet
import heapq

import numpy as np

# Weights: (w_travel_effort_agent, w_travel_effort_human, w_energy, w_privacy, w_time_to_meet)
# Prioritize human comfort: travel_effort_human gets highest weight.
# energy and privacy are derivatives of travel_effort_human (see cost_components),
# so their weights are kept low to avoid triple-counting the same signal.
# time_to_meet captures the joint wait — moderate weight.
DEFAULT_WEIGHTS = (0.20, 0.35, 0.10, 0.10, 0.25)


def manhattan_distance(pos_a: Tuple[int, int], pos_b: Tuple[int, int]) -> float:
    """Manhattan distance on grid."""
    return abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])


def astar_distance(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    obstacles: FrozenSet[Tuple[int, int]],
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
    heapq.heappush(open_heap, (manhattan_distance(start, goal), start))
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
                f = tentative + manhattan_distance(nb, goal)
                heapq.heappush(open_heap, (f, nb))
    return float("inf")


def cost_components(
    poi: Tuple[int, int],
    agent_pos: Tuple[int, int],
    human_pos: Tuple[int, int],
    grid_size: Tuple[int, int] = (64, 64),
    obstacles: FrozenSet[Tuple[int, int]] | None = None,
) -> Tuple[float, float, float, float, float]:
    """
    Raw cost components for one POI.
    Returns (travel_effort_agent, travel_effort_human, energy_human, privacy, time_to_meet).
    Uses A* distance when obstacles are provided, Manhattan otherwise.
    cost = w_te_a*travel_effort_agent + w_te_h*travel_effort_human + w_e*energy + w_p*privacy + w_ttm*time_to_meet
    """
    max_dist = grid_size[0] + grid_size[1]

    if obstacles:
        dist_a = astar_distance(agent_pos, poi, obstacles, grid_size)
        dist_h = astar_distance(human_pos, poi, obstacles, grid_size)
    else:
        dist_a = manhattan_distance(agent_pos, poi)
        dist_h = manhattan_distance(human_pos, poi)

    # Normalize — cap at 1.0 to handle inf gracefully (unreachable POI gets max cost)
    travel_effort_agent = min(dist_a / max_dist, 1.0)
    travel_effort_human = min(dist_h / max_dist, 1.0)

    # Human energy expenditure: [0.2, 0.8]
    energy_human = 0.2 + 0.6 * travel_effort_human

    # Privacy: higher when POI is near human
    privacy = 1.0 - travel_effort_human

    # Time-to-Meet: max of both travel efforts
    time_to_meet = max(travel_effort_agent, travel_effort_human)

    return travel_effort_agent, travel_effort_human, energy_human, privacy, time_to_meet


def cost_function(
    poi: Tuple[int, int],
    agent_pos: Tuple[int, int],
    human_pos: Tuple[int, int],
    grid_size: Tuple[int, int] = (64, 64),
    w_travel_effort_agent: float = 0.20,
    w_travel_effort_human: float = 0.20,
    w_energy: float = 0.20,
    w_privacy: float = 0.20,
    w_time_to_meet: float = 0.20,
    obstacles: FrozenSet[Tuple[int, int]] | None = None,
) -> float:
    """Compute cost for one POI. Lower = better. Uses A* when obstacles provided."""
    comps = cost_components(poi, agent_pos, human_pos, grid_size, obstacles)
    return sum(w * c for w, c in zip((w_travel_effort_agent, w_travel_effort_human, w_energy, w_privacy, w_time_to_meet), comps))


def nearest_human_baseline(
    pois: List[Tuple[int, int]],
    human_pos: Tuple[int, int],
    obstacles: FrozenSet[Tuple[int, int]] | None = None,
    grid_size: Tuple[int, int] = (64, 64),
) -> int:
    """Return index of POI closest to human. Uses A* when obstacles provided, Manhattan otherwise."""
    if obstacles:
        distances = [astar_distance(human_pos, poi, obstacles, grid_size) for poi in pois]
    else:
        distances = [manhattan_distance(human_pos, poi) for poi in pois]
    return int(np.argmin(distances))
