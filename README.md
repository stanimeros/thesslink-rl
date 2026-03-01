# ThessLink RL

Cost/reward function for meeting point suggestion. The agent suggests the best POI (Point of Interest) from 3 options based on **Distance**, **Privacy**, and **Energy**. Weights are learned dynamically through iterations.

## Overview

- **Inputs:** Human position, agent position, 3 POI suggestions
- **Output:** Agent suggests the best meeting point (lowest cost)
- **Factors:** Distance (agent→POI), Privacy (human→POI), Energy (effort)
- **Weights:** Learned via gradient descent in `train_weights.py`
- **Visualization:** Custom grid with H, A, P1, P2, P3 (no lb-foraging icons)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e lb-foraging/
pip install numpy gymnasium pyglet
```

## Usage

### Train cost function weights

```bash
python train_weights.py
```

Weights (distance, privacy, energy) are updated each iteration to minimize cost of the suggested POI.

### Run demo

```bash
python run_thesslink_demo.py               # lb-foraging: H and A move toward suggested POI (P)
python run_thesslink_demo.py --no-visualize  # Skip window
```

Uses **lb-foraging** to visualize movement: **H** (human) and **A** (agent) move toward the suggested meeting point **P** (from cost function).

## Project structure

```
thesslink-rl/
├── cost_function.py    # Cost function, rank_pois, suggest_poi
├── train_weights.py    # Train weights through iterations
├── run_thesslink_demo.py   # Grid: H, A, P1, P2, P3 + cost-based suggestion
├── lb-foraging/        # Optional (from semitable/lb-foraging)
└── requirements.txt
```

## Cost function

```
cost = w_distance × (agent→POI) + w_privacy × (1 - human→POI) + w_energy × (agent→POI)
```

Lower cost = better suggestion. Weights sum to 1 and are non-negative.

## Custom reward

Use `train_with_reward()` to plug in your own reward:

```python
from train_weights import train_with_reward

def my_reward(human_pos, agent_pos, pois, suggested_idx, weights):
    # Your logic: user satisfaction, meeting success, etc.
    return score

weights = train_with_reward(n_iterations=1000, reward_fn=my_reward)
```

## License

Uses [lb-foraging](https://github.com/semitable/lb-foraging) (MIT) for visualization.
