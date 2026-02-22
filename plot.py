"""
Plot training curves for all trained algorithms in a single comparison figure.
Reads evaluations.npz from each policies/<algo>/ folder.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_POLICIES_DIR = Path(__file__).parent / "policies"
DEFAULT_OUTPUT = Path(__file__).parent / "training_comparison.png"


def load_evaluations(policies_dir: Path):
    """Load (algo_name, timesteps, mean_reward, std_reward) from each algo folder."""
    results = []
    for subdir in sorted(policies_dir.iterdir()):
        if not subdir.is_dir():
            continue
        eval_path = subdir / "evaluations.npz"
        if not eval_path.exists():
            continue
        data = np.load(eval_path)
        timesteps = np.asarray(data["timesteps"])
        results_arr = np.asarray(data["results"])  # (n_evals, n_episodes)
        mean_reward = results_arr.mean(axis=1)
        std_reward = results_arr.std(axis=1)
        results.append((subdir.name, timesteps, mean_reward, std_reward))
    return results


def plot_comparison(
    policies_dir: Path | str = DEFAULT_POLICIES_DIR,
    output_path: Path | str = DEFAULT_OUTPUT,
):
    """Plot mean reward over timesteps for all algorithms."""
    policies_dir = Path(policies_dir)
    output_path = Path(output_path)

    if not policies_dir.exists():
        raise FileNotFoundError(f"Policies directory not found: {policies_dir}")

    data = load_evaluations(policies_dir)
    if not data:
        raise FileNotFoundError(
            f"No evaluations.npz found in {policies_dir}. Run train_all_algos.py first."
        )

    colors = plt.cm.tab10(np.linspace(0, 1, len(data)))
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (name, timesteps, mean_reward, std_reward) in enumerate(data):
        ts_k = timesteps / 1000
        ax.plot(
            ts_k,
            mean_reward,
            "o-",
            color=colors[i],
            linewidth=2,
            markersize=6,
            label=name,
        )
        ax.fill_between(
            ts_k,
            mean_reward - std_reward,
            mean_reward + std_reward,
            alpha=0.25,
            color=colors[i],
        )

    ax.set_xlabel("Timesteps (thousands)", fontsize=11)
    ax.set_ylabel("Mean episode reward", fontsize=11)
    ax.set_title("RL algorithm comparison — ThessLink meeting point", fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot algorithm comparison from policies directory"
    )
    parser.add_argument(
        "--policies-dir",
        type=Path,
        default=DEFAULT_POLICIES_DIR,
        help=f"Policies directory (default: {DEFAULT_POLICIES_DIR})",
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
        out = plot_comparison(policies_dir=args.policies_dir, output_path=args.output)
        print(f"Saved comparison plot to {out}")
        return 0
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    sys.exit(main())
