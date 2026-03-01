"""
Cost components and POI suggestion helpers.

- cost_components, cost_function: used for observation features and cost-based ranking
- suggest_poi: cost-based baseline (used for RL evaluation)
"""
from typing import Tuple, List

import numpy as np

DEFAULT_WEIGHTS = (0.20, 0.20, 0.20, 0.20, 0.20)


def manhattan_distance(pos_a: Tuple[int, int], pos_b: Tuple[int, int]) -> float:
    """Manhattan distance on grid."""
    return abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])


def cost_components(
    poi: Tuple[int, int],
    agent_pos: Tuple[int, int],
    human_pos: Tuple[int, int],
    grid_size: Tuple[int, int] = (64, 64),
) -> Tuple[float, float, float, float, float]:
    """
    Raw cost components for one POI.
    Returns (travel_effort_agent, travel_effort_human, energy_human, privacy, time_to_meet).
    cost = w_te_a*travel_effort_agent + w_te_h*travel_effort_human + w_e*energy + w_p*privacy + w_ttm*time_to_meet
    """
    max_dist = grid_size[0] + grid_size[1]

    # 1. Travel Effort: Manhattan distances (agent→POI, human→POI), normalized
    travel_effort_agent = manhattan_distance(agent_pos, poi) / max_dist
    travel_effort_human = manhattan_distance(human_pos, poi) / max_dist

    # 2. Human energy expenditure: [0.2, 0.8]. Penalizes human travel distance;
    #    closer POIs = lower energy cost (less walking/physical effort).
    energy_human = 0.2 + 0.6 * travel_effort_human

    # 3. Privacy: 1 − travel_effort_human. Higher when POI is near human (meet close
    #    to home = more private); lower when meeting far from human's location.
    privacy = 1.0 - travel_effort_human

    # 4. Time-to-Meet: min steps for both to arrive = max(travel_effort_agent, travel_effort_human)
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
) -> float:
    """Compute cost for one POI. Lower = better. Includes Travel Effort, energy (human effort), privacy (near human), Time-to-Meet."""
    comps = cost_components(poi, agent_pos, human_pos, grid_size)
    return sum(w * c for w, c in zip((w_travel_effort_agent, w_travel_effort_human, w_energy, w_privacy, w_time_to_meet), comps))


def suggest_poi(
    pois: List[Tuple[int, int]],
    agent_pos: Tuple[int, int],
    human_pos: Tuple[int, int],
    weights: Tuple[float, float, float, float, float] | None = None,
    grid_size: Tuple[int, int] = (64, 64),
) -> int:
    """Return index of best POI by cost. Uses DEFAULT_WEIGHTS when weights is None."""
    w = DEFAULT_WEIGHTS if weights is None else weights
    costs = [
        sum(wi * c for wi, c in zip(w, cost_components(p, agent_pos, human_pos, grid_size)))
        for p in pois
    ]
    return int(np.argmin(costs))
