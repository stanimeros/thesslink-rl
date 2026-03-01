"""
Cost components and POI suggestion helpers.

- suggest_poi_by_steps: picks POI that minimizes steps for both to arrive (no gradient descent)
- cost_components, cost_function: used for observation features and optional cost-based ranking
"""
import json
import os
from pathlib import Path
from typing import Tuple, List

import numpy as np

WEIGHTS_FILE = Path(__file__).parent / "thesslink_weights.json"


def manhattan_distance(pos_a: Tuple[int, int], pos_b: Tuple[int, int]) -> float:
    """Manhattan distance on grid."""
    return abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])


def cost_components(
    poi: Tuple[int, int],
    agent_pos: Tuple[int, int],
    human_pos: Tuple[int, int],
    grid_size: Tuple[int, int] = (64, 64),
) -> Tuple[float, float, float, float]:
    """
    Raw cost components for one POI.
    Returns (d_agent, d_human, energy_human, privacy).
    cost = w_da*d_agent + w_dh*d_human + w_e*energy_human + w_p*privacy
    """
    max_dist = grid_size[0] + grid_size[1]

    # 1. Both distances (Manhattan)
    d_agent = manhattan_distance(agent_pos, poi) / max_dist
    d_human = manhattan_distance(human_pos, poi) / max_dist

    # 2. Human energy: always in [0.2, 0.8] - humans want min energy but not 0
    energy_human = 0.2 + 0.6 * d_human

    # 3. Privacy (basic): higher when human far from POI (suggesting a place far from human)
    privacy = 1.0 - d_human

    return d_agent, d_human, energy_human, privacy


def cost_function(
    poi: Tuple[int, int],
    agent_pos: Tuple[int, int],
    human_pos: Tuple[int, int],
    grid_size: Tuple[int, int] = (64, 64),
    w_d_agent: float = 0.25,
    w_d_human: float = 0.25,
    w_energy: float = 0.25,
    w_privacy: float = 0.25,
) -> float:
    """Compute cost for one POI. Lower = better."""
    d_a, d_h, e, p = cost_components(poi, agent_pos, human_pos, grid_size)
    return w_d_agent * d_a + w_d_human * d_h + w_energy * e + w_privacy * p


def rank_pois(
    pois: List[Tuple[int, int]],
    agent_pos: Tuple[int, int],
    human_pos: Tuple[int, int],
    grid_size: Tuple[int, int] = (64, 64),
    weights: Tuple[float, float, float, float] | None = None,
) -> List[Tuple[int, Tuple[int, int], float]]:
    """Sort POIs by cost (ascending). Returns [(rank, poi, cost), ...]"""
    w = load_weights() if weights is None else weights
    costs = [
        (cost_function(p, agent_pos, human_pos, grid_size, *w), p)
        for p in pois
    ]
    costs.sort(key=lambda x: x[0])
    return [(i + 1, poi, c) for i, (c, poi) in enumerate(costs)]


def steps_to_both_arrive(
    poi: Tuple[int, int],
    agent_pos: Tuple[int, int],
    human_pos: Tuple[int, int],
) -> int:
    """Steps for both to reach POI (Manhattan). Returns max(d_agent, d_human) - no circles."""
    return max(
        manhattan_distance(agent_pos, poi),
        manhattan_distance(human_pos, poi),
    )


def suggest_poi_by_steps(
    pois: List[Tuple[int, int]],
    agent_pos: Tuple[int, int],
    human_pos: Tuple[int, int],
) -> int:
    """Return index of POI that minimizes steps for both to arrive. Direct paths, no circles."""
    steps = [steps_to_both_arrive(p, agent_pos, human_pos) for p in pois]
    return int(np.argmin(steps))


def suggest_poi(
    pois: List[Tuple[int, int]],
    agent_pos: Tuple[int, int],
    human_pos: Tuple[int, int],
    weights: Tuple[float, float, float, float] | None = None,
    grid_size: Tuple[int, int] = (64, 64),
) -> int:
    """Return index of best POI by cost (legacy, uses saved weights). Prefer suggest_poi_by_steps."""
    w = load_weights() if weights is None else weights
    costs = [
        sum(wi * c for wi, c in zip(w, cost_components(p, agent_pos, human_pos, grid_size)))
        for p in pois
    ]
    return int(np.argmin(costs))


def save_weights(weights: Tuple[float, float, float, float], path: Path | str = WEIGHTS_FILE) -> None:
    """Save weights to JSON file."""
    data = {
        "d_agent": weights[0],
        "d_human": weights[1],
        "energy": weights[2],
        "privacy": weights[3],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_weights(path: Path | str = WEIGHTS_FILE) -> Tuple[float, float, float, float]:
    """Load weights from JSON file. Returns defaults if file missing."""
    if not os.path.exists(path):
        return (0.25, 0.25, 0.25, 0.25)
    with open(path) as f:
        data = json.load(f)
    # Support old 3-field format for backward compatibility
    if "d_agent" in data:
        return (data["d_agent"], data["d_human"], data["energy"], data["privacy"])
    # Legacy: distance, privacy, energy -> split distance into d_agent, d_human
    w_d = data.get("distance", 1.0 / 3)
    w_p = data.get("privacy", 1.0 / 3)
    w_e = data.get("energy", 1.0 / 3)
    return (w_d / 2, w_d / 2, w_e, w_p)


