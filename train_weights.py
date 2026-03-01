#!/usr/bin/env python3
"""
Train cost function weights dynamically through iterations.

Inputs: human_pos, agent_pos, 3 POIs
Output: agent suggests best meeting point (lowest cost)
Weights (distance, privacy, energy) are updated each iteration.
"""
import numpy as np
from cost_function import suggest_poi, compute_gradient, cost_function


def sample_scenario(
    grid_size: tuple[int, int] = (64, 64),
    n_pois: int = 3,
    rng: np.random.Generator | None = None,
) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]]:
    """Sample random human, agent, and POI positions."""
    rng = rng or np.random.default_rng()
    rows, cols = grid_size
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    chosen = rng.choice(len(cells), size=2 + n_pois, replace=False)
    positions = [cells[i] for i in chosen]
    human_pos = positions[0]
    agent_pos = positions[1]
    pois = positions[2 : 2 + n_pois]
    return human_pos, agent_pos, pois


def train(
    n_iterations: int = 1000,
    lr: float = 0.01,
    grid_size: tuple[int, int] = (64, 64),
    seed: int | None = 42,
    log_every: int = 100,
) -> np.ndarray:
    """
    Iteratively update weights. Each iteration:
    1. Sample (human, agent, POIs)
    2. Agent suggests POI using current weights
    3. Compute reward (negative cost of suggested POI)
    4. Update weights via gradient descent to minimize cost
    """
    rng = np.random.default_rng(seed)
    # Start with equal weights, normalize to sum to 1
    weights = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3], dtype=np.float64)
    history: list[tuple[int, float, np.ndarray]] = []

    for i in range(n_iterations):
        human_pos, agent_pos, pois = sample_scenario(grid_size, n_pois=3, rng=rng)
        suggested = suggest_poi(pois, agent_pos, human_pos, tuple(weights), grid_size)
        grad = compute_gradient(pois, agent_pos, human_pos, suggested, grid_size)
        cost = cost_function(pois[suggested], agent_pos, human_pos, grid_size, *weights)

        # Gradient descent: minimize cost
        weights = weights - lr * grad

        # Keep weights non-negative and normalized
        weights = np.maximum(weights, 0.01)
        weights = weights / weights.sum()

        if (i + 1) % log_every == 0:
            history.append((i + 1, cost, weights.copy()))
            print(
                f"iter {i+1:5d} | cost={cost:.4f} | "
                f"w=[d:{weights[0]:.3f} p:{weights[1]:.3f} e:{weights[2]:.3f}]"
            )

    return weights


def train_with_reward(
    n_iterations: int = 1000,
    lr: float = 0.01,
    grid_size: tuple[int, int] = (64, 64),
    seed: int | None = 42,
    reward_fn=None,
) -> np.ndarray:
    """
    Train with custom reward function.

    reward_fn(human_pos, agent_pos, pois, suggested_idx, weights) -> float
    Higher reward = better. Weights are updated to maximize expected reward.
    """
    rng = np.random.default_rng(seed)
    weights = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3], dtype=np.float64)

    def _default_reward(hp, ap, pois, idx, w):
        c = cost_function(pois[idx], ap, hp, grid_size, *w)
        return -c

    reward_fn = reward_fn or _default_reward

    for i in range(n_iterations):
        human_pos, agent_pos, pois = sample_scenario(grid_size, n_pois=3, rng=rng)
        suggested = suggest_poi(pois, agent_pos, human_pos, tuple(weights), grid_size)
        grad = compute_gradient(pois, agent_pos, human_pos, suggested, grid_size)
        reward = reward_fn(human_pos, agent_pos, pois, suggested, weights)

        # Gradient ascent on reward (equiv to descent on -reward)
        # d(-cost)/dw = -grad, so we add lr * grad to maximize -cost
        weights = weights + lr * reward * grad  # REINFORCE-style scaling
        weights = np.maximum(weights, 0.01)
        weights = weights / weights.sum()

    return weights


if __name__ == "__main__":
    print("Training cost function weights (distance, privacy, energy)...")
    final_weights = train(n_iterations=500, lr=0.02, log_every=100)
    print(f"\nFinal weights: distance={final_weights[0]:.3f}, "
          f"privacy={final_weights[1]:.3f}, energy={final_weights[2]:.3f}")
