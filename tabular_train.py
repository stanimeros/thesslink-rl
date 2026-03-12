#!/usr/bin/env python3
"""
Train POI suggestion with Tabular RL (Q-Learning).

Tabular RL category: Q-Learning
  - State: discretized cost components per POI (3 bins each → 3^15 states)
  - Action: Discrete(3) – which POI to suggest
  - Reward: -cost

  python tabular_train.py              # Train 50k episodes, save model
  python tabular_train.py --episodes 200000
  python tabular_train.py --no-plot   # Skip plot
  python tabular_train.py --no-train  # Evaluate only
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from cost_function import nearest_human_baseline
from poi_environment import PoISuggestionEnv

MODEL_DIR = Path(__file__).parent / "models"
QLEARNING_DIR = MODEL_DIR / "qlearning"
MODEL_PATH = QLEARNING_DIR / "qtable.pkl"
PLOT_FILE = Path(__file__).parent / "training_plot_qlearning.png"

N_BINS = 3
N_COMPONENTS = 5
N_POIS = 3
STATE_SIZE = N_BINS ** (N_COMPONENTS * N_POIS)


def discretize_obs(obs: np.ndarray, n_bins: int = N_BINS) -> int:
    """
    Convert continuous observation (15 floats in [0,1]) to a single integer state index.
    Each component is binned into n_bins equal intervals.
    """
    bins = np.clip((obs * n_bins).astype(int), 0, n_bins - 1)
    state = 0
    for b in bins:
        state = state * n_bins + int(b)
    return state


def train(
    total_episodes: int = 50_000,
    seed: int = 42,
    save_path: Path | str = MODEL_PATH,
    plot_path: Path | str | None = PLOT_FILE,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.9999,
) -> np.ndarray:
    """Train Q-Learning agent. Returns Q-table."""
    env = PoISuggestionEnv(grid_size=(64, 64), seed=seed)
    rng = np.random.default_rng(seed)

    q_table = np.zeros((STATE_SIZE, N_POIS), dtype=np.float32)
    epsilon = epsilon_start

    reward_history: list[float] = []
    agreement_history: list[float] = []
    step_history: list[int] = []
    eval_freq = max(1000, total_episodes // 50)

    print(f"Q-table size: {STATE_SIZE} states × {N_POIS} actions")
    print(f"Training for {total_episodes} episodes...")

    for episode in range(total_episodes):
        obs, _ = env.reset()
        state = discretize_obs(obs)

        if rng.random() < epsilon:
            action = rng.integers(0, N_POIS)
        else:
            action = int(np.argmax(q_table[state]))

        obs_next, reward, terminated, truncated, _ = env.step(int(action))
        state_next = discretize_obs(obs_next)

        # Q-Learning update
        best_next = float(np.max(q_table[state_next]))
        q_table[state, action] += alpha * (reward + gamma * best_next - q_table[state, action])

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if (episode + 1) % eval_freq == 0:
            rewards, agreements = _evaluate(q_table, n_episodes=200, seed=seed)
            reward_history.append(rewards)
            agreement_history.append(agreements)
            step_history.append(episode + 1)
            print(f"  Episode {episode+1:>7}: reward={rewards:.4f}  agreement={agreements:.1%}  ε={epsilon:.3f}")

    QLEARNING_DIR.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(q_table, f)
    print(f"Saved Q-table to {save_path}")

    if plot_path and reward_history:
        _plot(step_history, reward_history, agreement_history, Path(plot_path))

    return q_table


def _evaluate(
    q_table: np.ndarray,
    n_episodes: int = 500,
    grid_size: tuple[int, int] = (64, 64),
    seed: int = 42,
) -> tuple[float, float]:
    """Returns (mean_reward, agreement_with_nearest_human_baseline)."""
    env = PoISuggestionEnv(grid_size=grid_size, seed=seed + 1)
    rewards = []
    agreements = 0

    for _ in range(n_episodes):
        obs, _ = env.reset()
        state = discretize_obs(obs)
        action = int(np.argmax(q_table[state]))
        _, reward, _, _, _ = env.step(action)
        rewards.append(reward)

        baseline_action = nearest_human_baseline(env._pois, env._human_pos)
        if action == baseline_action:
            agreements += 1

    return float(np.mean(rewards)), agreements / n_episodes


def _plot(
    steps: list[int],
    rewards: list[float],
    agreements: list[float],
    plot_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    axes[0].plot(steps, rewards, color="tab:orange", linewidth=0.8, alpha=0.9, marker="o", markersize=3)
    axes[0].set_ylabel("Mean Reward")
    axes[0].set_title("Tabular RL Training (Q-Learning)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, agreements, color="tab:orange", linewidth=0.8, alpha=0.9, marker="o", markersize=3)
    axes[1].set_ylabel("Agreement with nearest-human baseline")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("Episodes")
    axes[1].set_title("Agreement with nearest-human baseline")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training plot to {plot_path}")


def suggest_poi_qlearning(
    pois: list[tuple[int, int]],
    agent_pos: tuple[int, int],
    human_pos: tuple[int, int],
    grid_size: tuple[int, int] = (64, 64),
    model_path: Path | str | None = None,
) -> int:
    """Use trained Q-table to suggest POI. Returns index 0, 1, or 2."""
    import sys
    from poi_environment import build_observation

    path = Path(model_path) if model_path else MODEL_PATH
    if not path.exists():
        print("Q-table not found. Run tabular_train.py first.")
        sys.exit(1)

    with open(path, "rb") as f:
        q_table = pickle.load(f)

    obs = build_observation(human_pos, agent_pos, pois, grid_size)
    state = discretize_obs(obs)
    return int(np.argmax(q_table[state]))


def evaluate_vs_baseline(
    q_table: np.ndarray,
    n_episodes: int = 500,
    grid_size: tuple[int, int] = (64, 64),
) -> dict:
    """Compare Q-Learning vs nearest-human baseline."""
    mean_reward, agreement = _evaluate(q_table, n_episodes=n_episodes, grid_size=grid_size)
    agreements = int(agreement * n_episodes)
    return {"agreement": agreement, "agreements": agreements, "total": n_episodes, "mean_reward": mean_reward}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50_000, help="Training episodes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-train", action="store_true", help="Skip training, evaluate only")
    parser.add_argument("--no-plot", action="store_true", help="Skip generating training_plot_qlearning.png")
    parser.add_argument("--eval-episodes", type=int, default=500)
    args = parser.parse_args()

    if args.no_train:
        if not MODEL_PATH.exists():
            print(f"Q-table not found at {MODEL_PATH}. Run without --no-train first.")
            return
        with open(MODEL_PATH, "rb") as f:
            q_table = pickle.load(f)
        print("Evaluating Q-Learning vs nearest-human baseline...")
        stats = evaluate_vs_baseline(q_table, n_episodes=args.eval_episodes)
        print(f"Agreement: {stats['agreement']:.1%} ({stats['agreements']}/{stats['total']})")
        print(f"Mean reward: {stats['mean_reward']:.4f}")
        return

    print("Training POI suggestion with Q-Learning (tabular RL)...")
    print(f"  episodes={args.episodes}")
    q_table = train(
        total_episodes=args.episodes,
        seed=args.seed,
        plot_path=None if args.no_plot else PLOT_FILE,
    )
    print("\nEvaluating vs nearest-human baseline...")
    stats = evaluate_vs_baseline(q_table, n_episodes=args.eval_episodes)
    print(f"Agreement with nearest-human baseline: {stats['agreement']:.1%}")
    print(f"Mean reward: {stats['mean_reward']:.4f}")


if __name__ == "__main__":
    main()
