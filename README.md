# ThessLink-RL

**Near4all Research Project** — Reinforcement Learning for optimal Meeting Point (POI) suggestion between two users in Thessaloniki, Greece.

## Overview

ThessLink-RL uses Proximal Policy Optimization (PPO) to learn the best meeting point from a set of real Points of Interest (cafes, restaurants, parks) in Thessaloniki. The reward function balances:

- **Energy (Distance)**: Minimize total travel distance for both users
- **Privacy**: Maximize a privacy score (Parks=0.9, Cafes=0.5, Restaurants=0.3)

## Requirements

- Python 3.10+
- Gymnasium, Stable-Baselines3, OSMnx, Pandas, Geopy

## Setup

```bash
pip install -r requirements.txt
```

## Project Structure

```
thesslink-rl/
├── environment.py      # Gymnasium environment logic
├── action_wrappers.py  # Continuous action wrapper (for SAC/TD3/DDPG)
├── algorithms/         # Modular RL algorithm registry
├── train.py            # Train multiple algorithms
├── plot.py             # Compare all trained algorithms
├── predict.py          # Suggest meeting point using any trained policy
├── policies/           # Per-algorithm policy folders (best_model.zip, etc.)
├── requirements.txt
└── README.md
```

## Training

Train multiple RL algorithms (DQN, PPO, A2C, TRPO, SAC, TD3, DDPG):

```bash
python train.py
```

Each algorithm is saved under `policies/<algo>/`:

- `best_model.zip` — Best checkpoint by eval reward
- `<algo>_policy.zip` — Final policy at end of training
- `evaluations.npz` — Evaluation history for plotting

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--algos` | all | Algorithms to train |
| `--minutes` | 1 | Training time per algo (wall-clock minutes) |
| `--timesteps` | — | Override: use fixed timesteps instead of time limit |
| `--n-envs` | 4 | Parallel environments |
| `--policies-dir` | policies/ | Output directory |
| `--seed` | 42 | Random seed |

Plot comparison:

```bash
python plot.py -o training_comparison.png
```

## Inference

Suggest a meeting point for two users:

```python
from predict import suggest_meeting_point

result = suggest_meeting_point(
    lat_a=40.6293, lon_a=22.9597,  # User A (e.g. near Aristotle University)
    lat_b=40.6261, lon_b=22.9484,  # User B (e.g. near White Tower)
)

print(f"Meet at: ({result.lat}, {result.lon}) — {result.poi_type}")
print(f"Privacy: {result.privacy_score}, Total distance: {result.total_distance_km:.2f} km")
```

Or via CLI:

```bash
python predict.py
# or with custom coordinates:
python predict.py 40.6293 22.9597 40.6261 22.9484
# use a specific algorithm's policy:
python predict.py 40.6293 22.9597 40.6261 22.9484 --model-path policies/DQN/best_model.zip
```

The predict script auto-detects the algorithm from the path (DQN, PPO, A2C, TRPO, SAC, TD3, DDPG).

## Environment Details

- **State**: Normalized (lat, lon) for User A and User B within Thessaloniki bounds
- **Action**: Discrete choice among top N POIs from OpenStreetMap
- **Reward**: `weight_distance * (-total_distance/scale) + weight_privacy * privacy_score`

## Edge Cases

- **No POIs found**: Training raises `ValueError` with a clear message. Ensure network connectivity and that OSMnx/Overpass can be reached.
- **Empty POI geometries**: Filtered out during preprocessing.
- **Invalid action index**: Clamped to valid range in predict.
