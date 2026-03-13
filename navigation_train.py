#!/usr/bin/env python3
"""
Train cooperative navigation with PoINavigationEnv.

Two agents share a policy and learn to navigate to the same POI on a grid
with static obstacles. Supports three algorithm categories:

  Tabular RL       : Q-Learning  (--algo qlearning)
  Deep Value-based : DQN         (--algo dqn)
  Policy Gradient  : PPO         (--algo ppo)   [default]

Usage:
  python navigation_train.py                              # PPO, 64x64, 200k steps
  python navigation_train.py --algo dqn --grid-size 32
  python navigation_train.py --algo qlearning --grid-size 8 --episodes 200000
  python navigation_train.py --no-train --grid-size 32   # evaluate only
  python navigation_train.py --no-plot
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from cost_function import cost_optimal_baseline, DEFAULT_WEIGHTS
from poi_environment import PoINavigationEnv

MODEL_DIR = Path(__file__).parent / "models"
PLOT_DIR = Path(__file__).parent

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _eval_navigation(
    predict_fn,
    n_episodes: int = 200,
    seed: int = 99,
    grid_size: tuple[int, int] = (64, 64),
) -> dict:
    """
    Evaluate a navigation policy.
    predict_fn(obs) -> action  (called once per agent per step)
    Returns dict with mean_reward, mean_cost, cost_success, mean_steps, agreement.
    cost_success = 1 - cost (clamped to [0,1]); higher = closer to cost-optimal.
    agreement = fraction of episodes where env target matches cost-optimal baseline.
    """
    env = PoINavigationEnv(seed=seed, grid_size=grid_size)
    rewards, costs, steps_list, agreements = [], [], [], []

    for _ in range(n_episodes):
        (obs1, obs2), _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            a1 = predict_fn(obs1)
            a2 = predict_fn(obs2)
            (obs1, obs2), reward, terminated, truncated, info = env.step((a1, a2))
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
        costs.append(info["cost"])
        steps_list.append(info["step"])

        # Agreement: does env target match cost-optimal baseline? (at init positions)
        baseline_idx = cost_optimal_baseline(
            env._pois, env._init_agent1_pos, env._init_agent2_pos,
            env._obstacles, DEFAULT_WEIGHTS, env.grid_size
        )
        agreements.append(float(env._target_poi == env._pois[baseline_idx]))

    cost_successes = [max(0.0, 1.0 - c) for c in costs]
    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_cost": float(np.mean(costs)),
        "cost_success": float(np.mean(cost_successes)),
        "mean_steps": float(np.mean(steps_list)),
        "agreement": float(np.mean(agreements)),
    }


def _plot(
    steps: list[int],
    rewards: list[float],
    cost_successes: list[float],
    agreements: list[float],
    title: str,
    color: str,
    plot_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    axes[0].plot(steps, rewards, color=color, linewidth=0.8, alpha=0.9, marker="o", markersize=3)
    axes[0].set_ylabel("Mean Episode Reward")
    axes[0].set_title(f"Navigation Training ({title})")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, cost_successes, color=color, linewidth=0.8, alpha=0.9, marker="o", markersize=3)
    axes[1].set_ylabel("Cost Success (1 - cost)")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Cost Success (higher = closer to cost-optimal)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, agreements, color=color, linewidth=0.8, alpha=0.9, marker="o", markersize=3)
    axes[2].set_ylabel("Agreement with cost-optimal baseline")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_xlabel("Steps / Episodes")
    axes[2].set_title("Agreement with cost-optimal baseline")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {plot_path}")


# ─────────────────────────────────────────────────────────────────────────────
# PPO (Policy Gradient — Actor-Critic)
# ─────────────────────────────────────────────────────────────────────────────

def train_ppo(
    total_timesteps: int = 200_000,
    seed: int = 42,
    eval_freq: int = 10_000,
    plot_path: Path | None = None,
    grid_size: tuple[int, int] = (64, 64),
) -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    size_tag = str(grid_size[0])
    save_dir = MODEL_DIR / "ppo"
    save_dir.mkdir(parents=True, exist_ok=True)

    import gymnasium as gym

    class _NavWrapper(gym.Env):
        """Wraps PoINavigationEnv as single-agent env (shared policy, agent1 view)."""
        def __init__(self):
            super().__init__()
            self._env = PoINavigationEnv(seed=seed, grid_size=grid_size)
            self.observation_space = self._env.observation_space
            self.action_space = self._env.action_space
            self._obs2 = None

        def reset(self, **kwargs):
            (obs1, obs2), info = self._env.reset(**kwargs)
            self._obs2 = obs2
            return obs1, info

        def step(self, action):
            a2 = action
            (obs1, obs2), reward, term, trunc, info = self._env.step((action, a2))
            self._obs2 = obs2
            return obs1, reward, term, trunc, info

    env = _NavWrapper()

    reward_history, cost_success_history, agreement_history, step_history = [], [], [], []

    class _Callback(BaseCallback):
        def _on_step(self):
            if self.num_timesteps % eval_freq == 0:
                def predict(obs):
                    a, _ = self.model.predict(obs, deterministic=True)
                    return int(a)
                stats = _eval_navigation(predict, n_episodes=100, grid_size=grid_size)
                reward_history.append(stats["mean_reward"])
                cost_success_history.append(stats["cost_success"])
                agreement_history.append(stats["agreement"])
                step_history.append(self.num_timesteps)
                print(
                    f"  [{self.num_timesteps:>8}] reward={stats['mean_reward']:.3f}"
                    f"  cost_success={stats['cost_success']:.1%}"
                    f"  steps={stats['mean_steps']:.1f}"
                )
            return True

        def _on_training_end(self):
            if plot_path and reward_history:
                _plot(step_history, reward_history, cost_success_history, agreement_history,
                      f"PPO {size_tag}x{size_tag}", "tab:blue", plot_path)

    model = PPO("MlpPolicy", env, learning_rate=1e-4, n_steps=128, batch_size=64,
                n_epochs=10, gamma=0.99, max_grad_norm=0.5, clip_range_vf=10.0,
                verbose=1, seed=seed)
    model.learn(total_timesteps=total_timesteps, callback=_Callback())
    model.save(str(save_dir / f"nav_ppo_{size_tag}"))
    print(f"Saved PPO model to {save_dir / f'nav_ppo_{size_tag}'}")

    def predict(obs):
        a, _ = model.predict(obs, deterministic=True)
        return int(a)
    stats = _eval_navigation(predict, n_episodes=500, grid_size=grid_size)
    print(f"Final eval — cost_success: {stats['cost_success']:.1%}  reward: {stats['mean_reward']:.3f}  agreement: {stats['agreement']:.1%}")


# ─────────────────────────────────────────────────────────────────────────────
# DQN (Deep Value-based RL)
# ─────────────────────────────────────────────────────────────────────────────

def train_dqn(
    total_timesteps: int = 200_000,
    seed: int = 42,
    eval_freq: int = 10_000,
    plot_path: Path | None = None,
    grid_size: tuple[int, int] = (64, 64),
) -> None:
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import BaseCallback
    import gymnasium as gym

    size_tag = str(grid_size[0])
    save_dir = MODEL_DIR / "dqn"
    save_dir.mkdir(parents=True, exist_ok=True)

    class _NavWrapper(gym.Env):
        def __init__(self):
            super().__init__()
            self._env = PoINavigationEnv(seed=seed, grid_size=grid_size)
            self.observation_space = self._env.observation_space
            self.action_space = self._env.action_space

        def reset(self, **kwargs):
            (obs1, _), info = self._env.reset(**kwargs)
            return obs1, info

        def step(self, action):
            (obs1, _), reward, term, trunc, info = self._env.step((action, action))
            return obs1, reward, term, trunc, info

    env = _NavWrapper()

    reward_history, cost_success_history, agreement_history, step_history = [], [], [], []

    class _Callback(BaseCallback):
        def _on_step(self):
            if self.num_timesteps % eval_freq == 0:
                def predict(obs):
                    a, _ = self.model.predict(obs, deterministic=True)
                    return int(a)
                stats = _eval_navigation(predict, n_episodes=100, grid_size=grid_size)
                reward_history.append(stats["mean_reward"])
                cost_success_history.append(stats["cost_success"])
                agreement_history.append(stats["agreement"])
                step_history.append(self.num_timesteps)
                print(
                    f"  [{self.num_timesteps:>8}] reward={stats['mean_reward']:.3f}"
                    f"  cost_success={stats['cost_success']:.1%}"
                    f"  steps={stats['mean_steps']:.1f}"
                )
            return True

        def _on_training_end(self):
            if plot_path and reward_history:
                _plot(step_history, reward_history, cost_success_history, agreement_history,
                      f"DQN {size_tag}x{size_tag}", "tab:green", plot_path)

    model = DQN("MlpPolicy", env, learning_rate=1e-4, buffer_size=50_000,
                learning_starts=1000, batch_size=64, gamma=0.99, verbose=1, seed=seed)
    model.learn(total_timesteps=total_timesteps, callback=_Callback())
    model.save(str(save_dir / f"nav_dqn_{size_tag}"))
    print(f"Saved DQN model to {save_dir / f'nav_dqn_{size_tag}'}")

    def predict(obs):
        a, _ = model.predict(obs, deterministic=True)
        return int(a)
    stats = _eval_navigation(predict, n_episodes=500, grid_size=grid_size)
    print(f"Final eval — cost_success: {stats['cost_success']:.1%}  reward: {stats['mean_reward']:.3f}  agreement: {stats['agreement']:.1%}")


# ─────────────────────────────────────────────────────────────────────────────
# Q-Learning (Tabular RL)
# ─────────────────────────────────────────────────────────────────────────────

_NAV_BINS = 3
_NAV_OBS_DIM = 22  # self(2)+other(2)+costs*3(15)+target_onehot(3)
_NAV_ACTIONS = 5
_NAV_STATE_SIZE = _NAV_BINS ** _NAV_OBS_DIM  # tabular Q-learning impractical at this size


def _discretize_nav(obs: np.ndarray, n_bins: int = _NAV_BINS) -> int:
    bins = np.clip((obs * n_bins).astype(int), 0, n_bins - 1)
    state = 0
    for b in bins:
        state = state * n_bins + int(b)
    return state


def train_qlearning(
    total_episodes: int = 500_000,
    seed: int = 42,
    eval_freq: int = 25_000,
    plot_path: Path | None = None,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.99999,
    grid_size: tuple[int, int] = (64, 64),
) -> None:
    size_tag = str(grid_size[0])
    save_dir = MODEL_DIR / "qlearning"
    save_dir.mkdir(parents=True, exist_ok=True)

    from collections import defaultdict
    q_table: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(_NAV_ACTIONS, dtype=np.float32))

    env = PoINavigationEnv(seed=seed, grid_size=grid_size)
    rng = np.random.default_rng(seed)
    epsilon = epsilon_start

    reward_history, cost_success_history, agreement_history, step_history = [], [], [], []

    print(f"Training Q-Learning (navigation, {size_tag}x{size_tag}) for {total_episodes} episodes...")

    for episode in range(total_episodes):
        (obs1, obs2), _ = env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            s1 = _discretize_nav(obs1)
            s2 = _discretize_nav(obs2)

            if rng.random() < epsilon:
                a1 = int(rng.integers(0, _NAV_ACTIONS))
                a2 = int(rng.integers(0, _NAV_ACTIONS))
            else:
                a1 = int(np.argmax(q_table[s1]))
                a2 = int(np.argmax(q_table[s2]))

            (next_obs1, next_obs2), reward, terminated, truncated, _ = env.step((a1, a2))
            done = terminated or truncated
            ep_reward += reward

            ns1 = _discretize_nav(next_obs1)
            ns2 = _discretize_nav(next_obs2)

            q_table[s1][a1] += alpha * (reward + gamma * np.max(q_table[ns1]) - q_table[s1][a1])
            q_table[s2][a2] += alpha * (reward + gamma * np.max(q_table[ns2]) - q_table[s2][a2])

            obs1, obs2 = next_obs1, next_obs2

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if (episode + 1) % eval_freq == 0:
            def predict(obs):
                return int(np.argmax(q_table[_discretize_nav(obs)]))
            stats = _eval_navigation(predict, n_episodes=100, grid_size=grid_size)
            reward_history.append(stats["mean_reward"])
            cost_success_history.append(stats["cost_success"])
            agreement_history.append(stats["agreement"])
            step_history.append(episode + 1)
            print(
                f"  Episode {episode+1:>8}: reward={stats['mean_reward']:.3f}"
                f"  cost_success={stats['cost_success']:.1%}"
                f"  ε={epsilon:.4f}"
            )

    model_path = save_dir / f"nav_qtable_{size_tag}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(dict(q_table), f)
    print(f"Saved Q-table to {model_path}")

    if plot_path and reward_history:
        _plot(step_history, reward_history, cost_success_history, agreement_history,
              f"Q-Learning {size_tag}x{size_tag}", "tab:orange", plot_path)

    def predict(obs):
        return int(np.argmax(q_table[_discretize_nav(obs)]))
    stats = _eval_navigation(predict, n_episodes=500, grid_size=grid_size)
    print(f"Final eval — cost_success: {stats['cost_success']:.1%}  reward: {stats['mean_reward']:.3f}  agreement: {stats['agreement']:.1%}")


# ─────────────────────────────────────────────────────────────────────────────
# Suggest helpers (for demo.py)
# ─────────────────────────────────────────────────────────────────────────────

def suggest_nav_ppo(obs: np.ndarray, grid_size: tuple[int, int] = (64, 64)) -> int:
    """Load PPO navigation model and predict action for one agent."""
    from stable_baselines3 import PPO
    size_tag = str(grid_size[0])
    model = PPO.load(str(MODEL_DIR / "ppo" / f"nav_ppo_{size_tag}"))
    a, _ = model.predict(obs, deterministic=True)
    return int(a)


def suggest_nav_dqn(obs: np.ndarray, grid_size: tuple[int, int] = (64, 64)) -> int:
    from stable_baselines3 import DQN
    size_tag = str(grid_size[0])
    model = DQN.load(str(MODEL_DIR / "dqn" / f"nav_dqn_{size_tag}"))
    a, _ = model.predict(obs, deterministic=True)
    return int(a)


def suggest_nav_qlearning(obs: np.ndarray, grid_size: tuple[int, int] = (64, 64)) -> int:
    size_tag = str(grid_size[0])
    model_path = MODEL_DIR / "qlearning" / f"nav_qtable_{size_tag}.pkl"
    with open(model_path, "rb") as f:
        q_table = pickle.load(f)
    state = _discretize_nav(obs)
    return int(np.argmax(q_table.get(state, np.zeros(_NAV_ACTIONS))))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo", "dqn", "qlearning"], default="ppo")
    parser.add_argument("--grid-size", type=int, choices=[8, 32, 64], default=64,
                        help="Grid size (8, 32, or 64)")
    parser.add_argument("--steps", type=int, default=200_000, help="Timesteps (ppo/dqn)")
    parser.add_argument("--episodes", type=int, default=500_000, help="Episodes (qlearning)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    grid_size = (args.grid_size, args.grid_size)
    size_tag = str(args.grid_size)
    plot_path = None if args.no_plot else PLOT_DIR / f"training_plot_nav_{args.algo}_{size_tag}.png"

    if args.no_train:
        print(f"Evaluating {args.algo} navigation model ({size_tag}x{size_tag})...")
        if args.algo == "ppo":
            from stable_baselines3 import PPO
            model = PPO.load(str(MODEL_DIR / "ppo" / f"nav_ppo_{size_tag}"))
            predict = lambda obs: int(model.predict(obs, deterministic=True)[0])
        elif args.algo == "dqn":
            from stable_baselines3 import DQN
            model = DQN.load(str(MODEL_DIR / "dqn" / f"nav_dqn_{size_tag}"))
            predict = lambda obs: int(model.predict(obs, deterministic=True)[0])
        else:
            model_path = MODEL_DIR / "qlearning" / f"nav_qtable_{size_tag}.pkl"
            with open(model_path, "rb") as f:
                q_table = pickle.load(f)
            predict = lambda obs: int(np.argmax(q_table.get(_discretize_nav(obs), np.zeros(_NAV_ACTIONS))))
        stats = _eval_navigation(predict, n_episodes=500, grid_size=grid_size)
        print(f"Cost success: {stats['cost_success']:.1%}  Reward: {stats['mean_reward']:.3f}  Steps: {stats['mean_steps']:.1f}  Agreement: {stats['agreement']:.1%}")
        return

    print(f"Training navigation with {args.algo.upper()} on {size_tag}x{size_tag} grid...")
    if args.algo == "ppo":
        train_ppo(total_timesteps=args.steps, seed=args.seed, plot_path=plot_path, grid_size=grid_size)
    elif args.algo == "dqn":
        train_dqn(total_timesteps=args.steps, seed=args.seed, plot_path=plot_path, grid_size=grid_size)
    else:
        train_qlearning(total_episodes=args.episodes, seed=args.seed, plot_path=plot_path, grid_size=grid_size)


if __name__ == "__main__":
    main()
