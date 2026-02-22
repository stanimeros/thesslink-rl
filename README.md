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
├── env.py         # Gymnasium environment logic (state, action, reward)
├── train.py       # PPO training script
├── inference.py   # Suggest meeting point using trained policy
├── plot_result.py # Plot result on interactive Folium map
├── requirements.txt
└── README.md
```

## Training

Train the PPO agent on the ThessLink environment:

```bash
python train.py
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--timesteps` | 100000 | Total training timesteps |
| `--n-envs` | 4 | Parallel environments |
| `--top-n` | 20 | Number of POIs in action space |
| `--weight-distance` | 0.5 | Weight for distance minimization |
| `--weight-privacy` | 0.5 | Weight for privacy maximization |
| `--model-path` | thesslink_policy.zip | Output model path |
| `--seed` | 42 | Random seed |

Output:

- `thesslink_policy.zip` — Trained PPO policy
- `thesslink_pois.csv` — POI list used during training (required for inference)

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
```

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
