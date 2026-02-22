"""
ThessLink-RL Training Script for Near4all Research Project
Trains a PPO agent to suggest optimal Meeting Points (POIs) in Thessaloniki.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from env import ThessLinkEnv, fetch_thessaloniki_pois, THESSALONIKI_BBOX

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:
    raise ImportError(
        "stable-baselines3 is required. Install with: pip install stable-baselines3"
    )

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path(__file__).parent / "thesslink_policy.zip"
DEFAULT_POIS_PATH = Path(__file__).parent / "thesslink_pois.csv"


def make_env(pois: pd.DataFrame, seed: int | None = None):
    """Factory for creating a single env instance (for vectorization)."""

    def _init():
        env = ThessLinkEnv(pois=pois, seed=seed)
        return env

    return _init


def train(
    *,
    total_timesteps: int = 100_000,
    n_envs: int = 4,
    top_n_pois: int = 20,
    weight_distance: float = 0.5,
    weight_privacy: float = 0.5,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    pois_path: Path | str = DEFAULT_POIS_PATH,
    seed: int | None = 42,
    eval_freq: int = 5000,
    eval_episodes: int = 10,
) -> str:
    """
    Train a PPO agent on the ThessLink environment and save the policy.

    Returns:
        Path to the saved model file.
    """
    logger.info("Fetching POIs from OpenStreetMap (Thessaloniki)...")
    try:
        pois = fetch_thessaloniki_pois(top_n=top_n_pois)
    except ValueError as e:
        logger.error("Failed to fetch POIs: %s", e)
        raise

    logger.info("Found %d POIs. Types: %s", len(pois), pois["poi_type"].value_counts().to_dict())
    pois.to_csv(pois_path, index=False)
    logger.info("Saved POI list to %s", pois_path)

    env = ThessLinkEnv(
        pois=pois,
        weight_distance=weight_distance,
        weight_privacy=weight_privacy,
        seed=seed,
    )

    if n_envs > 1:
        vec_env = DummyVecEnv([make_env(pois, seed=seed + i) for i in range(n_envs)])
    else:
        vec_env = DummyVecEnv([make_env(pois, seed=seed)])

    eval_env = DummyVecEnv([make_env(pois, seed=seed + 1000)])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(Path(model_path).parent),
        log_path=str(Path(model_path).parent),
        eval_freq=eval_freq,
        n_eval_episodes=eval_episodes,
        deterministic=True,
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        seed=seed,
    )

    logger.info("Starting PPO training for %d timesteps...", total_timesteps)
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(str(model_path))
    logger.info("Saved trained policy to %s", model_path)

    vec_env.close()
    eval_env.close()
    return str(model_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train ThessLink-RL PPO agent for Meeting Point suggestion"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="Total training timesteps (default: 100000)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments (default: 4)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of POIs in action space (default: 20)",
    )
    parser.add_argument(
        "--weight-distance",
        type=float,
        default=0.5,
        help="Weight for distance minimization (default: 0.5)",
    )
    parser.add_argument(
        "--weight-privacy",
        type=float,
        default=0.5,
        help="Weight for privacy maximization (default: 0.5)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to save the trained model (default: thesslink_policy.zip)",
    )
    parser.add_argument(
        "--pois-path",
        type=Path,
        default=DEFAULT_POIS_PATH,
        help="Path to save the POI list (default: thesslink_pois.csv)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=5000,
        help="Evaluation frequency in timesteps (default: 5000)",
    )

    args = parser.parse_args()

    try:
        train(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            top_n_pois=args.top_n,
            weight_distance=args.weight_distance,
            weight_privacy=args.weight_privacy,
            model_path=args.model_path,
            pois_path=args.pois_path,
            seed=args.seed,
            eval_freq=args.eval_freq,
        )
        return 0
    except Exception as e:
        logger.exception("Training failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
