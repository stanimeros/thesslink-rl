"""
Plot training progress from evaluations.npz (saved by EvalCallback during train_all_algos.py).
Use for a single algorithm's curves; use plot_all_algos.py for multi-algo comparison.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

DEFAULT_EVAL_PATH = Path(__file__).parent / "policies" / "PPO" / "evaluations.npz"
DEFAULT_OUTPUT = Path(__file__).parent / "training_curves.png"


def plot_training(
    eval_path: Path | str = DEFAULT_EVAL_PATH,
    output_path: Path | str = DEFAULT_OUTPUT,
) -> Path:
    """Plot mean reward and episode length over training timesteps."""
    eval_path = Path(eval_path)
    if not eval_path.exists():
        raise FileNotFoundError(
            f"No evaluations found at {eval_path}. Run train_all_algos.py first."
        )

    data = np.load(eval_path)
    timesteps = data["timesteps"]
    results = data["results"]  # shape: (n_evals, n_episodes)
    ep_lengths = data["ep_lengths"]

    mean_reward = results.mean(axis=1)
    std_reward = results.std(axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # Reward over time
    ax1.plot(
        timesteps / 1000,
        mean_reward,
        "o-",
        color="#2563eb",
        linewidth=2,
        markersize=8,
        label="Mean reward",
    )
    ax1.fill_between(
        timesteps / 1000,
        mean_reward - std_reward,
        mean_reward + std_reward,
        alpha=0.3,
        color="#2563eb",
    )
    ax1.set_ylabel("Episode reward", fontsize=11)
    ax1.set_title("Policy improvement over training", fontsize=13, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # Episode length over time (often constant for this env, but included for completeness)
    mean_length = ep_lengths.mean(axis=1)
    std_length = ep_lengths.std(axis=1)
    ax2.plot(
        timesteps / 1000,
        mean_length,
        "o-",
        color="#059669",
        linewidth=2,
        markersize=8,
    )
    if std_length.any() > 0:
        ax2.fill_between(
            timesteps / 1000,
            mean_length - std_length,
            mean_length + std_length,
            alpha=0.3,
            color="#059669",
        )
    ax2.set_xlabel("Timesteps (thousands)", fontsize=11)
    ax2.set_ylabel("Episode length", fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    out = Path(output_path)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot training curves from evaluations.npz"
    )
    parser.add_argument(
        "--eval-path",
        type=Path,
        default=DEFAULT_EVAL_PATH,
        help=f"Path to evaluations.npz (default: {DEFAULT_EVAL_PATH})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output image path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    try:
        out = plot_training(eval_path=args.eval_path, output_path=args.output)
        print(f"Saved plot to {out}")
        return 0
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    sys.exit(main())
