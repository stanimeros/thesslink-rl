#!/usr/bin/env python3
"""
Train cooperative navigation with PoINavigationEnv (centralized controller).

A single model sees the full global state (33 floats including POI positions
for direction awareness) and outputs a joint action that selects the target
POI and independently moves both agents. Supports three
algorithm categories:

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
from poi_environment import PoINavigationEnv

MODEL_DIR = Path(__file__).parent / "models"


def get_history_path(algo: str, grid_size: int | str) -> Path:
    """Path to training_history_*.pkl in the model folder."""
    tag = str(grid_size)
    return MODEL_DIR / algo / f"training_history_{algo}_{tag}.pkl"


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _eval_navigation(
    predict_fn,
    n_episodes: int = 200,
    seed: int = 99,
    grid_size: tuple[int, int] = (64, 64),
    max_steps: int | None = None,
) -> dict:
    """
    Evaluate a navigation policy.
    predict_fn(obs) -> action  (single global observation → joint action)
    Returns dict with mean_reward, mean_cost, cost_success, mean_steps, agreement.
    cost_success = 1 - cost (clamped to [0,1]); higher = closer to cost-optimal.
    agreement = fraction of episodes where env target matches cost-optimal baseline.
    max_steps must match the training env so reward scales are identical.
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

        # Agreement: does policy-chosen target match cost-optimal baseline?
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
    """
    Load existing training history, or return empty lists if none exists.
    Also carries extra keys (e.g. epsilon, episodes_done for Q-Learning).
    """
    path = get_history_path(algo, size_tag)
    if path.exists():
        with open(path, "rb") as f:
            data = pickle.load(f)
        n = len(data.get("steps", []))
        print(f"  Resuming: loaded {n} existing checkpoints from {path.name}")
        return data
    return {"steps": [], "rewards": [], "cost_success": [], "agreement": []}


def _save_history(
    steps: list,
    rewards: list,
    cost_successes: list,
    agreements: list,
    algo: str,
    size_tag: str,
    **extra,
) -> None:
    save_dir = MODEL_DIR / algo
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"training_history_{algo}_{size_tag}.pkl"
    with open(path, "wb") as f:
        pickle.dump(
            {"steps": steps, "rewards": rewards, "cost_success": cost_successes,
             "agreement": agreements, **extra},
            f,
        )
    print(f"Saved history to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# PPO (Policy Gradient — Actor-Critic)
# ─────────────────────────────────────────────────────────────────────────────

def train_ppo(
    total_timesteps: int = 200_000,
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

    def _make_env(env_seed: int):
        def _init():
            return PoINavigationEnv(seed=env_seed, grid_size=grid_size, max_steps=max_steps)
        return _init

    n_envs = 6  # match physical core count
    env = SubprocVecEnv([_make_env(seed + i) for i in range(n_envs)])

    # Load existing history (continues x-axis from where we left off)
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
                              "ppo", size_tag)

    save_path = save_dir / f"nav_ppo_{size_tag}"
    # n_steps: rollout length per env before each PPO update.
    # Need n_steps * n_envs >= batch_size (256) and enough steps to see full episodes.
    # min 2 full episodes worth of steps per env, rounded to nearest 64 for efficiency.
    steps_per_env = max(max_steps * 2, 512)
    n_steps = (steps_per_env // n_envs // 64 + 1) * 64
    if save_path.with_suffix(".zip").exists():
        print(f"Resuming PPO from {save_path} (+{total_timesteps:,} steps)...")
        model = PPO.load(str(save_path), env=env)
        reset_num_timesteps = False
    else:
        model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=n_steps, batch_size=256,
                    n_epochs=10, gamma=0.99, max_grad_norm=0.5, clip_range_vf=10.0,
                    verbose=1, seed=seed)
        reset_num_timesteps = True

    model.learn(total_timesteps=total_timesteps, callback=_Callback(),
                reset_num_timesteps=reset_num_timesteps)
    model.save(str(save_path))
    print(f"Saved PPO model to {save_path}")

    def predict(obs):
        a, _ = model.predict(obs, deterministic=True)
        return int(a)
    stats = _eval_navigation(predict, n_episodes=500, grid_size=grid_size, max_steps=max_steps)
    print(f"Final eval — cost_success: {stats['cost_success']:.1%}  reward: {stats['mean_reward']:.3f}  agreement: {stats['agreement']:.1%}")


# ─────────────────────────────────────────────────────────────────────────────
# DQN (Deep Value-based RL)
# ─────────────────────────────────────────────────────────────────────────────

def train_dqn(
    total_timesteps: int = 200_000,
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
    def _make_env(env_seed: int):
        def _init():
            return PoINavigationEnv(seed=env_seed, grid_size=grid_size, max_steps=max_steps)
        return _init

    env = SubprocVecEnv([_make_env(seed + i) for i in range(n_envs)])

    # Load existing history (continues x-axis from where we left off)
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
        model = DQN("MlpPolicy", env, learning_rate=1e-4, buffer_size=100_000,
                    learning_starts=n_envs * 200, batch_size=128, gamma=0.99,
                    exploration_fraction=0.3, exploration_final_eps=0.05,
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
# Q-Learning (Tabular RL)
# ─────────────────────────────────────────────────────────────────────────────

# Joint action space: target_idx(0-2) * 25 + move1(0-4) * 5 + move2(0-4)
_NAV_ACTIONS = 75

# Observation layout (33 floats):
#   [0:2]  agent1_pos, [2:4] agent2_pos
#   [4:8]  wall_bits_a1 (N/S/W/E), [8:12] wall_bits_a2
#   [12:18] poi_positions (poi0_r, poi0_c, poi1_r, poi1_c, poi2_r, poi2_c)
#   [18:33] cost_components × 3 POIs (5 each: te_a, te_h, energy, privacy, ttm)
_COST_START = 18
_COST_STRIDE = 5  # floats per POI


def _discretize_nav(obs: np.ndarray) -> int:
    """
    Compact discrete state for tabular Q-Learning.
    State = (best_poi_idx, wall_bits_a1, dir_to_target, dist_bucket)
    Total states: 3 × 16 × 8 × 3 = 1,152 — tractable for tabular RL.
    """
    # Best POI: index of POI with lowest weighted cost sum
    costs = [
        obs[_COST_START + i * _COST_STRIDE] * 0.20   # te_a
        + obs[_COST_START + i * _COST_STRIDE + 1] * 0.35  # te_h
        + obs[_COST_START + i * _COST_STRIDE + 2] * 0.10  # energy
        + obs[_COST_START + i * _COST_STRIDE + 3] * 0.10  # privacy
        + obs[_COST_START + i * _COST_STRIDE + 4] * 0.25  # ttm
        for i in range(3)
    ]
    best_poi = int(np.argmin(costs))  # 0-2

    # Wall bits a1: 4 binary bits → integer 0-15
    wall_bits = int(obs[4]) * 8 + int(obs[5]) * 4 + int(obs[6]) * 2 + int(obs[7])  # 0-15

    # Direction: encode agent1 position as 8 octants relative to grid center
    sr, sc = float(obs[0]), float(obs[1])
    dr = sr - 0.5
    dc = sc - 0.5
    angle = int((np.arctan2(dc, dr) + np.pi) / (np.pi / 4)) % 8  # 0-7

    # Distance bucket to best POI (te_a of best POI → 3 buckets)
    te_a_best = float(obs[_COST_START + best_poi * _COST_STRIDE])
    dist_bucket = 0 if te_a_best < 0.25 else (1 if te_a_best < 0.55 else 2)  # 0-2

    return best_poi * (16 * 8 * 3) + wall_bits * (8 * 3) + angle * 3 + dist_bucket


def _qlearning_worker(args: tuple) -> dict:
    """
    Top-level worker for multiprocessing — runs a chunk of Q-Learning episodes.
    Each worker starts from the same Q-table and explores independently.
    Returns the updated Q-table dict after running chunk_episodes episodes.
    """
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
    """
    Merge Q-tables from parallel workers by element-wise averaging.
    States visited by only some workers are averaged over those workers only.
    """
    all_keys = set().union(*tables)
    merged = {}
    for k in all_keys:
        arrays = [t[k] for t in tables if k in t]
        merged[k] = np.mean(arrays, axis=0).astype(np.float32)
    return merged


def train_qlearning(
    total_episodes: int = 200_000,
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
    """
    Parallel Q-Learning: each eval_freq chunk is split across n_workers processes
    that explore independently, then Q-tables are merged by element-wise averaging.
    Effective episodes = total_episodes; wall-clock time ≈ total_episodes / n_workers.
    """
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

        # Advance epsilon as if chunk_size episodes ran sequentially
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
    parser.add_argument("--steps", type=int, default=200_000, help="Timesteps (ppo/dqn)")
    parser.add_argument("--episodes", type=int, default=200_000, help="Episodes (qlearning)")
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
