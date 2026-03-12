#!/usr/bin/env python3
"""
Train POI suggestion with value-based RL (DQN).

  python value_based_train.py              # Train 50k timesteps, save model
  python value_based_train.py --steps 100000
  python value_based_train.py --no-train   # Just evaluate loaded model
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from cost_function import nearest_human_baseline
from poi_environment import PoISuggestionEnv

MODEL_DIR = Path(__file__).parent / "models"
DQN_DIR = MODEL_DIR / "dqn"
MODEL_PATH = DQN_DIR / "dqn_poi_suggestion"
PLOT_FILE = Path(__file__).parent / "training_plot_dqn.png"
_best_model_path = DQN_DIR / "best_model"


class PlottingCallback(BaseCallback):
    """Log eval reward and agreement with cost baseline for training plot."""

    def __init__(
        self,
        eval_freq: int,
        n_eval_episodes: int = 100,
        plot_path: Path | str | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.plot_path = Path(plot_path) if plot_path else None
        self.reward_history: list[float] = []
        self.agreement_history: list[float] = []
        self.step_history: list[int] = []

    def _on_step(self) -> bool:
        if self.num_timesteps > 0 and self.num_timesteps % self.eval_freq == 0:
            eval_env = PoISuggestionEnv(grid_size=(64, 64))
            # Eval mean reward
            rewards = []
            for _ in range(self.n_eval_episodes):
                obs, _ = eval_env.reset()
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, term, trunc, _ = eval_env.step(int(action))
                    done = term or trunc
                    rewards.append(reward)
            mean_reward = float(np.mean(rewards))
            self.reward_history.append(mean_reward)
            self.step_history.append(self.num_timesteps)

            # Agreement with nearest-human baseline
            agreements = 0
            for _ in range(self.n_eval_episodes):
                obs, _ = eval_env.reset()
                rl_action = int(self.model.predict(obs, deterministic=True)[0])
                human_pos = eval_env._human_pos
                pois = eval_env._pois
                baseline_action = nearest_human_baseline(pois, human_pos)
                agreements += 1 if rl_action == baseline_action else 0
            self.agreement_history.append(agreements / self.n_eval_episodes)
        return True

    def _on_training_end(self) -> None:
        if not self.plot_path or not self.reward_history:
            return
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        steps = self.step_history

        axes[0].plot(steps, self.reward_history, color="tab:green", linewidth=0.8, alpha=0.9, marker="o", markersize=3)
        axes[0].set_ylabel("Mean Reward")
        axes[0].set_title("RL Training (DQN)")
        axes[0].grid(True, alpha=0.3)

        if self.agreement_history:
            axes[1].plot(steps, self.agreement_history, color="tab:green", linewidth=0.8, alpha=0.9, marker="o", markersize=3)
            axes[1].set_ylabel("Agreement with nearest-human baseline")
            axes[1].set_ylim(0, 1.05)
        else:
            axes[1].plot(steps, self.reward_history, color="tab:green", linewidth=0.8, alpha=0.9)
            axes[1].set_ylabel("Mean Reward")
        axes[1].set_xlabel("Timesteps")
        axes[1].set_title("Agreement with nearest-human baseline" if self.agreement_history else "Mean Reward")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(str(self.plot_path), dpi=150, bbox_inches="tight")
        plt.close()
        if self.verbose:
            print(f"Saved training plot to {self.plot_path}")


def suggest_poi_dqn(
    pois: list[tuple[int, int]],
    agent_pos: tuple[int, int],
    human_pos: tuple[int, int],
    grid_size: tuple[int, int] = (64, 64),
    model_path: Path | str | None = None,
) -> int:
    """
    Use trained DQN to suggest POI. Returns index 0, 1, or 2.
    Exits with message if model not found.
    """
    import sys

    if model_path is not None:
        path = Path(model_path)
        if not path.suffix:
            path = Path(f"{path}.zip")
    else:
        for candidate in (_best_model_path, MODEL_PATH):
            p = Path(f"{candidate}.zip") if not str(candidate).endswith(".zip") else Path(candidate)
            if p.exists():
                path = p
                break
        else:
            print("Model not found. Run value_based_train.py first to train the DQN.")
            sys.exit(1)

    from poi_environment import build_observation

    model = DQN.load(str(path))
    obs = build_observation(human_pos, agent_pos, pois, grid_size)
    action, _ = model.predict(obs, deterministic=True)
    return int(action)


def register_env():
    if "PoISuggestion-v0" in gym.envs.registry:
        gym.envs.registry.pop("PoISuggestion-v0", None)
    gym.register(
        id="PoISuggestion-v0",
        entry_point="poi_environment:PoISuggestionEnv",
        kwargs={"grid_size": (64, 64)},
    )


def train(
    total_timesteps: int = 50_000,
    seed: int = 42,
    eval_freq: int = 5000,
    save_path: Path | str = MODEL_PATH,
    plot_path: Path | str | None = PLOT_FILE,
) -> DQN:
    register_env()
    env = gym.make("PoISuggestion-v0")
    eval_env = gym.make("PoISuggestion-v0")

    MODEL_DIR.mkdir(exist_ok=True)
    DQN_DIR.mkdir(exist_ok=True)

    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=str(DQN_DIR),
            log_path=str(DQN_DIR),
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
        ),
    ]
    if plot_path:
        plot_eval_freq = min(eval_freq, max(500, total_timesteps // 20))
        callbacks.append(
            PlottingCallback(
                eval_freq=plot_eval_freq,
                n_eval_episodes=50,
                plot_path=plot_path,
                verbose=1,
            )
        )

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=50_000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        verbose=1,
        seed=seed,
    )
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    model.save(str(save_path))
    print(f"Saved model to {save_path}")
    env.close()
    eval_env.close()
    gym.envs.registry.pop("PoISuggestion-v0", None)
    return model


def evaluate_vs_baseline(
    model: DQN,
    n_episodes: int = 500,
    grid_size: tuple[int, int] = (64, 64),
) -> dict:
    """Compare DQN vs nearest-human baseline. Returns agreement rate."""
    env = PoISuggestionEnv(grid_size=grid_size, seed=42)
    agreements = 0
    total = 0

    for _ in range(n_episodes):
        obs, _ = env.reset()
        rl_action, _ = model.predict(obs, deterministic=True)
        rl_action = int(rl_action)

        human_pos = env._human_pos
        pois = env._pois
        baseline_action = nearest_human_baseline(pois, human_pos)

        if rl_action == baseline_action:
            agreements += 1
        total += 1

    return {"agreement": agreements / total, "agreements": agreements, "total": total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50_000, help="Training timesteps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-train", action="store_true", help="Skip training, evaluate only")
    parser.add_argument("--no-plot", action="store_true", help="Skip generating training_plot_dqn.png")
    parser.add_argument("--eval-episodes", type=int, default=500)
    args = parser.parse_args()

    if args.no_train:
        best_zip = _best_model_path.with_suffix(".zip")
        model_zip = MODEL_PATH.with_suffix(".zip")
        if not model_zip.exists() and not best_zip.exists():
            print(f"Model not found. Run without --no-train first.")
            return
        path = best_zip if best_zip.exists() else model_zip
        model = DQN.load(str(path))
        print("Evaluating DQN vs nearest-human baseline...")
        stats = evaluate_vs_baseline(model, n_episodes=args.eval_episodes)
        print(f"Agreement: {stats['agreement']:.1%} ({stats['agreements']}/{stats['total']})")
        return

    print("Training POI suggestion with DQN (value-based, cost reward)...")
    print(f"  steps={args.steps}")
    model = train(
        total_timesteps=args.steps,
        seed=args.seed,
        plot_path=None if args.no_plot else PLOT_FILE,
    )
    print("\nEvaluating vs nearest-human baseline...")
    stats = evaluate_vs_baseline(model, n_episodes=args.eval_episodes)
    print(f"Agreement with nearest-human baseline: {stats['agreement']:.1%}")


if __name__ == "__main__":
    main()
