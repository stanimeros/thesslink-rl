"""
Cost/Reward function for POI suggestion ranking.

Factors: Distance, Privacy, Energy
Weights are learned dynamically through iterations (see train_weights.py).
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


def cost_components(
    poi: Tuple[int, int],
    agent_pos: Tuple[int, int],
    human_pos: Tuple[int, int],
    grid_size: Tuple[int, int] = (64, 64),
) -> Tuple[float, float, float]:
    """
    Return raw (distance, privacy, energy) components for one POI.
    Used for gradient-based weight updates: cost = w_d*d + w_p*p + w_e*e
    """
    dist_agent_poi = manhattan_distance(agent_pos, poi)
    dist_norm = dist_agent_poi / (grid_size[0] + grid_size[1])

    dist_human_poi = manhattan_distance(human_pos, poi)
    privacy_norm = dist_human_poi / (grid_size[0] + grid_size[1])
    privacy_cost = 1.0 - privacy_norm

    energy_norm = dist_agent_poi / (grid_size[0] + grid_size[1])
    return dist_norm, privacy_cost, energy_norm


def suggest_poi(
    pois: List[Tuple[int, int]],
    agent_pos: Tuple[int, int],
    human_pos: Tuple[int, int],
    weights: Tuple[float, float, float] | None = None,
    grid_size: Tuple[int, int] = (64, 64),
) -> int:
    """
    Agent suggests best POI (index 0, 1, or 2). Returns index of lowest-cost POI.
    """
    w_d, w_p, w_e = weights or (0.33, 0.33, 0.34)
    costs = [
        sum(w * c for w, c in zip((w_d, w_p, w_e), cost_components(p, agent_pos, human_pos, grid_size)))
        for p in pois
    ]
    return int(np.argmin(costs))


def compute_gradient(
    pois: List[Tuple[int, int]],
    agent_pos: Tuple[int, int],
    human_pos: Tuple[int, int],
    suggested_idx: int,
    grid_size: Tuple[int, int] = (64, 64),
) -> np.ndarray:
    """
    Gradient of cost w.r.t. weights for the suggested POI.
    d(cost)/d(w_d, w_p, w_e) = (dist_norm, privacy_cost, energy_norm)
    """
    d, p, e = cost_components(pois[suggested_idx], agent_pos, human_pos, grid_size)
    return np.array([d, p, e], dtype=np.float64)
