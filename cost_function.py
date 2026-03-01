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
    Returns (d_agent, d_human, energy_human, privacy, steps).
    cost = w_da*d_agent + w_dh*d_human + w_e*energy + w_p*privacy + w_s*steps
    """
    max_dist = grid_size[0] + grid_size[1]

    # 1. Both distances (Manhattan)
    d_agent = manhattan_distance(agent_pos, poi) / max_dist
    d_human = manhattan_distance(human_pos, poi) / max_dist

    # 2. Human energy: always in [0.2, 0.8] - humans want min energy but not 0
    energy_human = 0.2 + 0.6 * d_human

    # 3. Privacy (basic): higher when human far from POI
    privacy = 1.0 - d_human

    # 4. Steps: min steps for both to arrive = max(d_agent, d_human) in Manhattan
    steps = max(d_agent, d_human)

    return d_agent, d_human, energy_human, privacy, steps


def cost_function(
    poi: Tuple[int, int],
    agent_pos: Tuple[int, int],
    human_pos: Tuple[int, int],
    grid_size: Tuple[int, int] = (64, 64),
    w_d_agent: float = 0.20,
    w_d_human: float = 0.20,
    w_energy: float = 0.20,
    w_privacy: float = 0.20,
    w_steps: float = 0.20,
) -> float:
    """Compute cost for one POI. Lower = better. Includes d_agent, d_human, energy, privacy, steps."""
    comps = cost_components(poi, agent_pos, human_pos, grid_size)
    return sum(w * c for w, c in zip((w_d_agent, w_d_human, w_energy, w_privacy, w_steps), comps))


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
