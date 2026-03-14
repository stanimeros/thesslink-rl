#!/usr/bin/env python3
"""
Train cooperative navigation with PoINavigationEnv (centralized controller).

A single model sees 35-float relative observations and outputs a joint action
(MultiDiscrete for PPO, Discrete(75) via FlatActionWrapper for DQN/Q-Learning).

  Tabular RL       : Q-Learning  (--algo qlearning)
  Deep Value-based : DQN         (--algo dqn)
  Policy Gradient  : PPO         (--algo ppo)   [default]

Usage:
  python navigation_train.py                              # PPO, 8x8, 500k steps
  python navigation_train.py --algo dqn --grid-size 32
  python navigation_train.py --algo qlearning --grid-size 8 --episodes 200000
  python navigation_train.py --no-train --grid-size 32   # evaluate only
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from cost_function import cost_optimal_baseline, DEFAULT_WEIGHTS
from poi_environment import PoINavigationEnv, FlatActionWrapper

MODEL_DIR = Path(__file__).parent / "models"


def get_history_path(algo: str, grid_size: int | str) -> Path:
    tag = str(grid_size)
    return MODEL_DIR / algo / f"training_history_{algo}_{tag}.pkl"


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_env(grid_size, max_steps, env_seed, flat=False):
    """Factory that returns a callable for SubprocVecEnv."""
    def _init():
        env = PoINavigationEnv(seed=env_seed, grid_size=grid_size, max_steps=max_steps)
        return FlatActionWrapper(env) if flat else env
    return _init


def _eval_navigation(
    predict_fn,
    n_episodes: int = 200,
    seed: int = 99,
    grid_size: tuple[int, int] = (64, 64),
    max_steps: int | None = None,
) -> dict:
    """
    Evaluate a navigation policy.
    predict_fn(obs) -> action (flat int for DQN/QL, or array for PPO)
    """
    if max_steps is None:
        max_steps = max(300, grid_size[0] * grid_size[1] // 2)
    env = PoINavigationEnv(seed=seed, grid_size=grid_size, max_steps=max_steps)
    rewards, costs, steps_list, agreements = [], [], [], []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action = predict_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
        costs.append(info["cost"])
        steps_list.append(info["step"])

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


def _load_history(algo: str, size_tag: str) -> dict:
    path = get_history_path(algo, size_tag)
    if path.exists():
        with open(path, "rb") as f:
            data = pickle.load(f)
        n = len(data.get("steps", []))
        print(f"  Resuming: loaded {n} existing checkpoints from {path.name}")
        return data
    return {"steps": [], "rewards": [], "cost_success": [], "agreement": []}


def _save_history(
    steps: list, rewards: list, cost_successes: list, agreements: list,
    algo: str, size_tag: str, **extra,
) -> None:
    save_dir = MODEL_DIR / algo
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"training_history_{algo}_{size_tag}.pkl"
    with open(path, "wb") as f:
        pickle.dump(
            {"steps": steps, "rewards": rewards, "cost_success": cost_successes,
             "agreement": agreements, **extra}, f,
        )
    print(f"Saved history to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# PPO — uses native MultiDiscrete([3, 5, 5])
# ─────────────────────────────────────────────────────────────────────────────

def train_ppo(
    total_timesteps: int = 1_000_000,
    seed: int = 42,
    eval_freq: int = 25_000,
    grid_size: tuple[int, int] = (64, 64),
) -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv

    size_tag = str(grid_size[0])
    save_dir = MODEL_DIR / "ppo"
    save_dir.mkdir(parents=True, exist_ok=True)

    max_steps = max(300, grid_size[0] * grid_size[1] // 2)

    n_envs = 6
    env = SubprocVecEnv([_make_env(grid_size, max_steps, seed + i) for i in range(n_envs)])

    hist = _load_history("ppo", size_tag)
    step_history = hist["steps"]
    reward_history = hist["rewards"]
    cost_success_history = hist["cost_success"]
    agreement_history = hist["agreement"]

    class _Callback(BaseCallback):
        def _on_step(self):
            if self.num_timesteps % eval_freq == 0:
                def predict(obs):
                    a, _ = self.model.predict(obs, deterministic=True)
                    return a
                stats = _eval_navigation(predict, n_episodes=100, grid_size=grid_size, max_steps=max_steps)
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
            if reward_history:
                _save_history(step_history, reward_history, cost_success_history, agreement_history,
                              "ppo", size_tag)

    save_path = save_dir / f"nav_ppo_{size_tag}"
    steps_per_env = max(max_steps * 2, 512)
    n_steps = (steps_per_env // n_envs // 64 + 1) * 64
    if save_path.with_suffix(".zip").exists():
        print(f"Resuming PPO from {save_path} (+{total_timesteps:,} steps)...")
        model = PPO.load(str(save_path), env=env)
        reset_num_timesteps = False
    else:
        model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=n_steps, batch_size=256,
                    n_epochs=10, gamma=0.99, max_grad_norm=0.5, clip_range_vf=10.0,
                    ent_coef=0.01,
                    policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
                    verbose=1, seed=seed)
        reset_num_timesteps = True

    model.learn(total_timesteps=total_timesteps, callback=_Callback(),
                reset_num_timesteps=reset_num_timesteps)
    model.save(str(save_path))
    print(f"Saved PPO model to {save_path}")

    def predict(obs):
        a, _ = model.predict(obs, deterministic=True)
        return a
    stats = _eval_navigation(predict, n_episodes=500, grid_size=grid_size, max_steps=max_steps)
    print(f"Final eval — cost_success: {stats['cost_success']:.1%}  reward: {stats['mean_reward']:.3f}  agreement: {stats['agreement']:.1%}")


# ─────────────────────────────────────────────────────────────────────────────
# DQN — uses FlatActionWrapper → Discrete(75)
# ─────────────────────────────────────────────────────────────────────────────

def train_dqn(
    total_timesteps: int = 1_000_000,
    seed: int = 42,
    eval_freq: int = 25_000,
    grid_size: tuple[int, int] = (64, 64),
) -> None:
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv

    size_tag = str(grid_size[0])
    save_dir = MODEL_DIR / "dqn"
    save_dir.mkdir(parents=True, exist_ok=True)

    max_steps = max(300, grid_size[0] * grid_size[1] // 2)

    n_envs = 6
    env = SubprocVecEnv([_make_env(grid_size, max_steps, seed + i, flat=True) for i in range(n_envs)])

    hist = _load_history("dqn", size_tag)
    step_history = hist["steps"]
    reward_history = hist["rewards"]
    cost_success_history = hist["cost_success"]
    agreement_history = hist["agreement"]

    class _Callback(BaseCallback):
        def _on_step(self):
            if self.num_timesteps % eval_freq == 0:
                def predict(obs):
                    a, _ = self.model.predict(obs, deterministic=True)
                    return int(a)
                stats = _eval_navigation(predict, n_episodes=100, grid_size=grid_size, max_steps=max_steps)
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
            if reward_history:
                _save_history(step_history, reward_history, cost_success_history, agreement_history,
                              "dqn", size_tag)

    save_path = save_dir / f"nav_dqn_{size_tag}"
    if save_path.with_suffix(".zip").exists():
        print(f"Resuming DQN from {save_path} (+{total_timesteps:,} steps)...")
        model = DQN.load(str(save_path), env=env)
        reset_num_timesteps = False
    else:
        model = DQN("MlpPolicy", env, learning_rate=1e-4, buffer_size=200_000,
                    learning_starts=5_000, batch_size=256, gamma=0.99,
                    exploration_fraction=0.4, exploration_final_eps=0.05,
                    target_update_interval=1_000,
                    policy_kwargs=dict(net_arch=[128, 128]),
                    verbose=1, seed=seed)
        reset_num_timesteps = True

    model.learn(total_timesteps=total_timesteps, callback=_Callback(),
                reset_num_timesteps=reset_num_timesteps)
    model.save(str(save_path))
    print(f"Saved DQN model to {save_path}")

    def predict(obs):
        a, _ = model.predict(obs, deterministic=True)
        return int(a)
    stats = _eval_navigation(predict, n_episodes=500, grid_size=grid_size, max_steps=max_steps)
    print(f"Final eval — cost_success: {stats['cost_success']:.1%}  reward: {stats['mean_reward']:.3f}  agreement: {stats['agreement']:.1%}")


# ─────────────────────────────────────────────────────────────────────────────
# Q-Learning (Tabular RL) — uses flat Discrete(75) actions
# ─────────────────────────────────────────────────────────────────────────────

_NAV_ACTIONS = 75

# Observation layout (35 floats — relative):
#   [0:6]   delta_a1_to_pois (dr,dc × 3 POIs)
#   [6:12]  delta_a2_to_pois (dr,dc × 3 POIs)
#   [12:16] wall_bits_a1 (N/S/W/E)
#   [16:20] wall_bits_a2 (N/S/W/E)
#   [20:35] cost_components × 3 POIs (5 each: te_a, te_h, energy, privacy, ttm)
_COST_START = 20
_COST_STRIDE = 5


def _discretize_nav(obs: np.ndarray) -> int:
    """
    Compact discrete state for tabular Q-Learning.
    State = (best_poi, wall_a1, dir_a1→target, dist_a1, wall_a2, dir_a2→target, dist_a2)
    Total states: 3 × 16 × 8 × 3 × 16 × 8 × 3 = 442,368
    """
    costs = [
        obs[_COST_START + i * _COST_STRIDE] * 0.20
        + obs[_COST_START + i * _COST_STRIDE + 1] * 0.35
        + obs[_COST_START + i * _COST_STRIDE + 2] * 0.10
        + obs[_COST_START + i * _COST_STRIDE + 3] * 0.10
        + obs[_COST_START + i * _COST_STRIDE + 4] * 0.25
        for i in range(3)
    ]
    best_poi = int(np.argmin(costs))

    wall_a1 = int(obs[12]) * 8 + int(obs[13]) * 4 + int(obs[14]) * 2 + int(obs[15])

    dr1, dc1 = float(obs[best_poi * 2]), float(obs[best_poi * 2 + 1])
    angle_a1 = int((np.arctan2(dc1, dr1) + np.pi) / (np.pi / 4)) % 8 if (abs(dr1) + abs(dc1)) > 1e-6 else 0

    te_a = float(obs[_COST_START + best_poi * _COST_STRIDE])
    dist_a1 = 0 if te_a < 0.15 else (1 if te_a < 0.45 else 2)

    wall_a2 = int(obs[16]) * 8 + int(obs[17]) * 4 + int(obs[18]) * 2 + int(obs[19])

    dr2, dc2 = float(obs[6 + best_poi * 2]), float(obs[7 + best_poi * 2])
    angle_a2 = int((np.arctan2(dc2, dr2) + np.pi) / (np.pi / 4)) % 8 if (abs(dr2) + abs(dc2)) > 1e-6 else 0

    te_h = float(obs[_COST_START + best_poi * _COST_STRIDE + 1])
    dist_a2 = 0 if te_h < 0.15 else (1 if te_h < 0.45 else 2)

    return (best_poi * (16 * 8 * 3 * 16 * 8 * 3)
            + wall_a1 * (8 * 3 * 16 * 8 * 3)
            + angle_a1 * (3 * 16 * 8 * 3)
            + dist_a1 * (16 * 8 * 3)
            + wall_a2 * (8 * 3)
            + angle_a2 * 3
            + dist_a2)


def _qlearning_worker(args: tuple) -> dict:
    (q_table_dict, chunk_episodes, worker_seed,
     alpha, gamma, epsilon, epsilon_end, epsilon_decay, grid_size) = args

    from collections import defaultdict

    max_steps = max(300, grid_size[0] * grid_size[1] // 2)
    env = PoINavigationEnv(seed=worker_seed, grid_size=grid_size, max_steps=max_steps)
    rng = np.random.default_rng(worker_seed)
    q_table = defaultdict(lambda: np.zeros(_NAV_ACTIONS, dtype=np.float32), q_table_dict)

    for _ in range(chunk_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            s = _discretize_nav(obs)
            if rng.random() < epsilon:
                action = int(rng.integers(0, _NAV_ACTIONS))
            else:
                action = int(np.argmax(q_table[s]))
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ns = _discretize_nav(next_obs)
            q_table[s][action] += alpha * (reward + gamma * np.max(q_table[ns]) - q_table[s][action])
            obs = next_obs
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    return dict(q_table)


def _merge_qtables(tables: list[dict]) -> dict:
    all_keys = set().union(*tables)
    merged = {}
    for k in all_keys:
        arrays = [t[k] for t in tables if k in t]
        merged[k] = np.mean(arrays, axis=0).astype(np.float32)
    return merged


def train_qlearning(
    total_episodes: int = 400_000,
    seed: int = 42,
    eval_freq: int = 10_000,
    n_workers: int = 6,
    alpha: float = 0.2,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.99997,
    grid_size: tuple[int, int] = (64, 64),
) -> None:
    """Parallel Q-Learning with periodic merge and evaluation."""
    import multiprocessing as mp
    from collections import defaultdict

    size_tag = str(grid_size[0])
    save_dir = MODEL_DIR / "qlearning"
    save_dir.mkdir(parents=True, exist_ok=True)

    max_steps = max(300, grid_size[0] * grid_size[1] // 2)
    model_path = save_dir / f"nav_qtable_{size_tag}.pkl"

    hist = _load_history("qlearning", size_tag)
    step_history: list = hist["steps"]
    reward_history: list = hist["rewards"]
    cost_success_history: list = hist["cost_success"]
    agreement_history: list = hist["agreement"]

    if model_path.exists():
        with open(model_path, "rb") as f:
            q_table: dict = dict(pickle.load(f))
        epsilon = float(hist.get("epsilon", epsilon_end))
        episodes_done = int(hist.get("episodes_done", 0))
        print(f"Resuming Q-Learning from {model_path} "
              f"(+{total_episodes:,} episodes, ε={epsilon:.4f})...")
    else:
        q_table = {}
        epsilon = epsilon_start
        episodes_done = 0

    n_chunks = max(1, total_episodes // eval_freq)
    chunk_size = total_episodes // n_chunks
    print(f"Training Q-Learning ({size_tag}×{size_tag}) — "
          f"{total_episodes:,} episodes in {n_chunks} chunks × {n_workers} workers...")

    for chunk in range(n_chunks):
        episodes_per_worker = max(1, chunk_size // n_workers)

        worker_args = [
            (q_table, episodes_per_worker, seed + chunk * n_workers + w,
             alpha, gamma, epsilon, epsilon_end, epsilon_decay, grid_size)
            for w in range(n_workers)
        ]

        with mp.Pool(processes=n_workers) as pool:
            results = pool.map(_qlearning_worker, worker_args)

        q_table = _merge_qtables(results)
        epsilon = max(epsilon_end, epsilon * (epsilon_decay ** chunk_size))
        episodes_done_total = episodes_done + (chunk + 1) * chunk_size

        q_table_dd = defaultdict(lambda: np.zeros(_NAV_ACTIONS, dtype=np.float32), q_table)
        def predict(obs, _qt=q_table_dd):
            return int(np.argmax(_qt[_discretize_nav(obs)]))
        stats = _eval_navigation(predict, n_episodes=100, grid_size=grid_size, max_steps=max_steps)
        reward_history.append(stats["mean_reward"])
        cost_success_history.append(stats["cost_success"])
        agreement_history.append(stats["agreement"])
        step_history.append(episodes_done_total)
        print(
            f"  Episode {episodes_done_total:>8}: reward={stats['mean_reward']:.3f}"
            f"  cost_success={stats['cost_success']:.1%}"
            f"  ε={epsilon:.4f}"
        )

    with open(model_path, "wb") as f:
        pickle.dump(q_table, f)
    print(f"Saved Q-table to {model_path}")

    _save_history(step_history, reward_history, cost_success_history, agreement_history,
                  "qlearning", size_tag,
                  epsilon=epsilon,
                  episodes_done=episodes_done + total_episodes)

    q_table_dd = defaultdict(lambda: np.zeros(_NAV_ACTIONS, dtype=np.float32), q_table)
    def predict(obs, _qt=q_table_dd):
        return int(np.argmax(_qt[_discretize_nav(obs)]))
    stats = _eval_navigation(predict, n_episodes=500, grid_size=grid_size, max_steps=max_steps)
    print(f"Final eval — cost_success: {stats['cost_success']:.1%}  reward: {stats['mean_reward']:.3f}  agreement: {stats['agreement']:.1%}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo", "dqn", "qlearning"], default="ppo")
    parser.add_argument("--grid-size", type=int, choices=[8, 32, 64], default=8,
                        help="Grid size (8, 32, or 64)")
    parser.add_argument("--steps", type=int, default=1_000_000, help="Timesteps (ppo/dqn)")
    parser.add_argument("--episodes", type=int, default=400_000, help="Episodes (qlearning)")
    parser.add_argument("--workers", type=int, default=6, help="Parallel workers (qlearning)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-train", action="store_true")
    args = parser.parse_args()

    grid_size = (args.grid_size, args.grid_size)
    size_tag = str(args.grid_size)

    if args.no_train:
        print(f"Evaluating {args.algo} navigation model ({size_tag}x{size_tag})...")
        if args.algo == "ppo":
            from stable_baselines3 import PPO
            model = PPO.load(str(MODEL_DIR / "ppo" / f"nav_ppo_{size_tag}"))
            predict = lambda obs: model.predict(obs, deterministic=True)[0]
        elif args.algo == "dqn":
            from stable_baselines3 import DQN
            model = DQN.load(str(MODEL_DIR / "dqn" / f"nav_dqn_{size_tag}"))
            predict = lambda obs: int(model.predict(obs, deterministic=True)[0])
        else:
            model_path = MODEL_DIR / "qlearning" / f"nav_qtable_{size_tag}.pkl"
            with open(model_path, "rb") as f:
                q_table = pickle.load(f)
            predict = lambda obs: int(np.argmax(q_table.get(_discretize_nav(obs), np.zeros(_NAV_ACTIONS))))
        eval_max_steps = max(300, grid_size[0] * grid_size[1] // 2)
        stats = _eval_navigation(predict, n_episodes=500, grid_size=grid_size, max_steps=eval_max_steps)
        print(f"Cost success: {stats['cost_success']:.1%}  Reward: {stats['mean_reward']:.3f}  Steps: {stats['mean_steps']:.1f}  Agreement: {stats['agreement']:.1%}")
        return

    print(f"Training navigation with {args.algo.upper()} on {size_tag}x{size_tag} grid...")
    if args.algo == "ppo":
        train_ppo(total_timesteps=args.steps, seed=args.seed, grid_size=grid_size)
    elif args.algo == "dqn":
        train_dqn(total_timesteps=args.steps, seed=args.seed, grid_size=grid_size)
    else:
        train_qlearning(total_episodes=args.episodes, seed=args.seed, grid_size=grid_size,
                        n_workers=args.workers)


if __name__ == "__main__":
    main()
