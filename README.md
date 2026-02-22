# ThessLink-RL

**Near4all Research Project** — Reinforcement Learning for optimal Meeting Point (POI) suggestion between two users in Thessaloniki, Greece.

## Overview

ThessLink-RL uses Proximal Policy Optimization (PPO) to learn the best meeting point from a set of real Points of Interest (cafes, restaurants, parks) in Thessaloniki. The reward function balances:

- **Energy (Distance)**: Minimize total travel distance for both users
- **Privacy**: Maximize a privacy score (Parks=0.9, Cafes=0.5, Restaurants=0.3)

## Requirements

- Python 3.10+
- Gymnasium, Stable-Baselines3, OSMnx, Pandas, Geopy, Folium

## Setup

```bash
pip install -r requirements.txt
```

## Project Structure

```
thesslink-rl/
├── env.py              # Gymnasium environment logic
├── env_wrappers.py     # Continuous action wrapper (for SAC/TD3/DDPG)
├── algorithms/         # Modular RL algorithm registry
├── train_all_algos.py  # Train multiple algorithms
├── plot_training.py    # Plot single-run training curves
├── plot_all_algos.py   # Compare all trained algorithms
├── inference.py        # Suggest meeting point using any trained policy
├── plot_result.py      # Plot result on interactive Folium map
├── policies/           # Per-algorithm policy folders (best_model.zip, etc.)
├── requirements.txt
└── README.md
```

## Training

Train multiple RL algorithms (DQN, PPO, A2C, TRPO, SAC, TD3, DDPG):

```bash
python train_all_algos.py
```

Each algorithm is saved under `policies/<algo>/`:

- `best_model.zip` — Best checkpoint by eval reward
- `<algo>_policy.zip` — Final policy at end of training
- `evaluations.npz` — Evaluation history for plotting

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--algos` | all | Algorithms to train |
| `--timesteps` | 100000 | Training timesteps per algo |
| `--n-envs` | 4 | Parallel environments |
| `--policies-dir` | policies/ | Output directory |
| `--seed` | 42 | Random seed |

Plot comparison:

```bash
python plot_all_algos.py -o algorithm_comparison.png
```

Single-algo curves:

```bash
python plot_training.py --eval-path policies/PPO/evaluations.npz
```

## Inference

Suggest a meeting point for two users:

```python
from inference import suggest_meeting_point

result = suggest_meeting_point(
    lat_a=40.6293, lon_a=22.9597,  # User A (e.g. near Aristotle University)
    lat_b=40.6261, lon_b=22.9484,  # User B (e.g. near White Tower)
)

print(f"Meet at: ({result.lat}, {result.lon}) — {result.poi_type}")
print(f"Privacy: {result.privacy_score}, Total distance: {result.total_distance_km:.2f} km")
```

Or via CLI:

```bash
python inference.py
# or with custom coordinates:
python inference.py 40.6293 22.9597 40.6261 22.9484
# use a specific algorithm's policy:
python inference.py 40.6293 22.9597 40.6261 22.9484 --model-path policies/DQN/best_model.zip
```

The inference script auto-detects the algorithm from the path (DQN, PPO, A2C, TRPO, SAC, TD3, DDPG).

## Map Visualization

Plot User A, User B, and the suggested meeting point on an interactive Folium map:

```bash
python plot_result.py
# or with custom coordinates:
python plot_result.py 40.6293 22.9597 40.6261 22.9484
# with custom output path:
python plot_result.py 40.6293 22.9597 40.6261 22.9484 my_map.html
```

Opens `meeting_point_map.html` in a browser to view the map with markers and polylines.

## Environment Details

- **State**: Normalized (lat, lon) for User A and User B within Thessaloniki bounds
- **Action**: Discrete choice among top N POIs from OpenStreetMap
- **Reward**: `weight_distance * (-total_distance/scale) + weight_privacy * privacy_score`

## Edge Cases

- **No POIs found**: Training raises `ValueError` with a clear message. Ensure network connectivity and that OSMnx/Overpass can be reached.
- **Empty POI geometries**: Filtered out during preprocessing.
- **Invalid action index**: Clamped to valid range in inference.

## License

Research project — Near4all.
