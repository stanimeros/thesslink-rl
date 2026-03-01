"""
Cost/Reward function for POI suggestion ranking.

Factors: Distance, Privacy, Energy
Used to sort 3 POI suggestions and optimize via RL.
"""
import numpy as np
from typing import Tuple, List


def manhattan_distance(pos_a: Tuple[int, int], pos_b: Tuple[int, int]) -> float:
    """Manhattan distance on grid."""
    return abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])


def euclidean_distance(pos_a: Tuple[int, int], pos_b: Tuple[int, int]) -> float:
    """Euclidean distance."""
    return np.sqrt((pos_a[0] - pos_b[0]) ** 2 + (pos_a[1] - pos_b[1]) ** 2)


def cost_function(
    poi: Tuple[int, int],
    agent_pos: Tuple[int, int],
    human_pos: Tuple[int, int],
    grid_size: Tuple[int, int] = (64, 64),
    w_distance: float = 0.4,
    w_privacy: float = 0.3,
    w_energy: float = 0.3,
) -> float:
    """
    Compute cost for a single POI. Lower is better (minimize cost = maximize reward).

    - Distance: path length from agent to POI (shorter = lower cost)
    - Privacy: distance from human to POI (further from human = more privacy = lower cost)
    - Energy: proxy for effort (e.g., distance from agent; can be refined)
    """
    # Distance: agent -> POI (agent travel cost)
    dist_agent_poi = manhattan_distance(agent_pos, poi)
    dist_norm = dist_agent_poi / (grid_size[0] + grid_size[1])  # normalize to [0,1]

    # Privacy: human -> POI (further from human = more private)
    dist_human_poi = manhattan_distance(human_pos, poi)
    privacy_norm = dist_human_poi / (grid_size[0] + grid_size[1])
    # Invert: higher distance from human = lower cost (more private)
    privacy_cost = 1.0 - privacy_norm

    # Energy: can be distance-based or custom (e.g., terrain, elevation)
    energy_norm = dist_agent_poi / (grid_size[0] + grid_size[1])

    cost = w_distance * dist_norm + w_privacy * privacy_cost + w_energy * energy_norm
    return cost


def rank_pois(
    pois: List[Tuple[int, int]],
    agent_pos: Tuple[int, int],
    human_pos: Tuple[int, int],
    grid_size: Tuple[int, int] = (64, 64),
    **kwargs,
) -> List[Tuple[int, Tuple[int, int], float]]:
    """
    Sort 3 POI suggestions by cost (ascending). Returns list of (rank, poi, cost).
    """
    costs = [(cost_function(poi, agent_pos, human_pos, grid_size, **kwargs), poi) for poi in pois]
    costs.sort(key=lambda x: x[0])
    return [(i + 1, poi, c) for i, (c, poi) in enumerate(costs)]
