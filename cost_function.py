"""
Cost/Reward function for POI suggestion ranking.

Components: (1) agent distance, (2) human distance, (3) human energy (20–80%), (4) privacy
Run as script to train and save weights. run_thesslink_demo.py loads saved weights.

  python cost_function.py              # Train and save weights to thesslink_weights.json
  python cost_function.py --no-train   # Just load existing weights (for testing)
"""
import argparse
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


def suggest_poi(
    pois: List[Tuple[int, int]],
    agent_pos: Tuple[int, int],
    human_pos: Tuple[int, int],
    weights: Tuple[float, float, float, float] | None = None,
    grid_size: Tuple[int, int] = (64, 64),
) -> int:
    """Return index of best POI. Uses saved weights if weights=None."""
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


def train_and_save(
    n_iterations: int = 500,
    lr: float = 0.02,
    grid_size: tuple[int, int] = (64, 64),
    seed: int | None = 42,
    log_every: int = 100,
    output_path: Path | str = WEIGHTS_FILE,
) -> Tuple[float, float, float, float]:
    """Train weights via gradient descent and save to file."""
    rng = np.random.default_rng(seed)
    rows, cols = grid_size
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    weights = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)

    for i in range(n_iterations):
        chosen = rng.choice(len(cells), size=5, replace=False)
        positions = [cells[j] for j in chosen]
        human_pos, agent_pos = positions[0], positions[1]
        pois = positions[2:5]

        suggested = suggest_poi(pois, agent_pos, human_pos, tuple(weights), grid_size)
        comps = cost_components(pois[suggested], agent_pos, human_pos, grid_size)
        grad = np.array(comps, dtype=np.float64)
        cost = cost_function(pois[suggested], agent_pos, human_pos, grid_size, *weights)

        weights = weights - lr * grad
        weights = np.maximum(weights, 0.01)
        weights = weights / weights.sum()

        if (i + 1) % log_every == 0:
            print(
                f"iter {i+1:5d} | cost={cost:.4f} | w=[d_a:{weights[0]:.3f} d_h:{weights[1]:.3f} e:{weights[2]:.3f} p:{weights[3]:.3f}]"
            )

    w_tuple = (float(weights[0]), float(weights[1]), float(weights[2]), float(weights[3]))
    save_weights(w_tuple, output_path)
    print(f"Saved weights to {output_path}")
    return w_tuple


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-train", action="store_true", help="Skip training, just show current weights")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.02)
    args = parser.parse_args()

    if args.no_train:
        w = load_weights()
        print(f"Loaded weights: d_agent={w[0]:.3f}, d_human={w[1]:.3f}, energy={w[2]:.3f}, privacy={w[3]:.3f}")
    else:
        print("Training cost function weights...")
        train_and_save(n_iterations=args.iterations, lr=args.lr)
