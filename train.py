"""
Train multiple RL algorithms on ThessLink and save policies per algorithm.
Each algorithm gets its own folder under policies/<algo>/ with best_model.zip and evaluations.npz.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from algorithms import ALGORITHMS, create_model, get_continuous_algos
from environment import ThessLinkEnv, fetch_thessaloniki_pois
from action_wrappers import ThessLinkContinuousWrapper


class TimeLimitCallback(BaseCallback):
    """Stop training after a fixed wall-clock time (seconds)."""

    def __init__(self, time_limit_sec: float, verbose: int = 0):
        super().__init__(verbose)
        self.time_limit_sec = time_limit_sec
        self.start_time: float | None = None

    def _on_training_start(self) -> None:
        self.start_time = time.monotonic()

    def _on_step(self) -> bool:
        if self.start_time is None:
            return True
        elapsed = time.monotonic() - self.start_time
        if elapsed >= self.time_limit_sec:
            if self.verbose:
                logger.info("Time limit reached (%.1f s), stopping training.", elapsed)
            return False
        return True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_POLICIES_DIR = Path(__file__).parent / "policies"
DEFAULT_POIS_PATH = Path(__file__).parent / "thesslink_pois.csv"


def make_env(pois: pd.DataFrame, seed: int | None, continuous: bool = False):
    def _init():
        env = ThessLinkEnv(pois=pois, seed=seed)
        if continuous:
            env = ThessLinkContinuousWrapper(env)
        return env
    return _init


def train_one(
    algo_name: str,
    pois: pd.DataFrame,
    policies_dir: Path,
    total_timesteps: int | None = None,
    time_limit_sec: float | None = None,
    n_envs: int = 4,
    eval_freq: int = 5_000,
    n_eval_episodes: int = 10,
    seed: int | None = 42,
) -> Path:
    """Train a single algorithm and save to policies/<algo_name>/."""
    continuous = algo_name in get_continuous_algos()
    out_dir = policies_dir / algo_name
    out_dir.mkdir(parents=True, exist_ok=True)

    vec_env = DummyVecEnv([lambda i=i: make_env(pois, seed=seed + i, continuous=continuous)() for i in range(n_envs)])
    eval_env = DummyVecEnv([lambda: make_env(pois, seed=seed + 1000, continuous=continuous)()])

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(out_dir),
        log_path=str(out_dir),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
    )

    if time_limit_sec is not None and time_limit_sec > 0:
        callbacks = CallbackList([eval_callback, TimeLimitCallback(time_limit_sec=time_limit_sec)])
        steps = 10_000_000
        logger.info("Training %s for %.1f minutes (wall-clock)...", algo_name, time_limit_sec / 60)
    else:
        callbacks = eval_callback
        steps = total_timesteps or 100_000
        logger.info("Training %s for %d timesteps...", algo_name, steps)

    model = create_model(algo_name, vec_env, seed=seed)
    model.learn(total_timesteps=steps, callback=callbacks)

    # Save final policy (in addition to best_model from EvalCallback)
    final_path = out_dir / f"{algo_name.lower()}_policy.zip"
    model.save(str(final_path))
    logger.info("Saved %s to %s", algo_name, final_path)

    vec_env.close()
    eval_env.close()
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train multiple RL algorithms on ThessLink"
    )
    parser.add_argument(
        "--algos",
        nargs="+",
        default=list(ALGORITHMS.keys()),
        help=f"Algorithms to train (default: all). Options: {list(ALGORITHMS.keys())}",
    )
    parser.add_argument(
        "--policies-dir",
        type=Path,
        default=DEFAULT_POLICIES_DIR,
        help=f"Output directory for policies (default: {DEFAULT_POLICIES_DIR})",
    )
    parser.add_argument(
        "--pois-path",
        type=Path,
        default=DEFAULT_POIS_PATH,
        help="Path to POI CSV",
    )
    parser.add_argument(
        "--minutes",
        type=float,
        default=1.0,
        help="Training time per algorithm in minutes (default: 1). Use for quick runs.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override: use fixed timesteps instead of time limit. Ignores --minutes.",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Parallel environments",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=5_000,
        help="Evaluation frequency",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    for name in args.algos:
        if name not in ALGORITHMS:
            logger.error("Unknown algorithm: %s. Options: %s", name, list(ALGORITHMS.keys()))
            return 1

    # Load POIs
    pois_path = Path(args.pois_path)
    if pois_path.exists():
        logger.info("Loading POIs from %s", pois_path)
        pois = pd.read_csv(pois_path)
    else:
        logger.info("Fetching POIs from OSM...")
        pois = fetch_thessaloniki_pois(top_n=50)
        pois.to_csv(pois_path, index=False)
    logger.info("Using %d POIs", len(pois))

    args.policies_dir.mkdir(parents=True, exist_ok=True)

    time_limit = None if args.timesteps is not None else args.minutes * 60

    for algo_name in args.algos:
        try:
            train_one(
                algo_name,
                pois,
                args.policies_dir,
                total_timesteps=args.timesteps,
                time_limit_sec=time_limit,
                n_envs=args.n_envs,
                eval_freq=args.eval_freq,
                seed=args.seed,
            )
        except Exception as e:
            logger.exception("Failed to train %s: %s", algo_name, e)
            return 1

    logger.info("All algorithms trained. Policies saved to %s", args.policies_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
