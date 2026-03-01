# ThessLink RL

**Reinforcement Learning** for meeting point suggestion. The agent suggests the best POI (Point of Interest) from 3 options to **minimize steps** for both human and agent to arrive. Direct paths, no circles.

![lb-foraging environment](lb-foraging/docs/img/lbf.gif)

## Overview

- **Inputs:** Human position, agent position, 3 POI suggestions
- **Output:** Agent suggests the meeting point that minimizes steps for both to arrive
- **Reward:** `-steps/max_dist` where steps = max(d_agent, d_human) — Manhattan distance, direct path
- **No gradient descent:** RL (PPO) learns the policy; fallback is `suggest_poi_by_steps` (argmin steps)
- **Visualization:** lb-foraging grid with H, A, P1, P2, P3 labels showing steps per POI

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e lb-foraging/
pip install -r requirements.txt
```

## Usage

### 1. Train RL policy (`train_rl.py`)

```bash
python train_rl.py                    # Train PPO 50k steps (steps reward), save to models/
python train_rl.py --steps 100000     # More training
python train_rl.py --reward cost       # Use cost-based reward (legacy)
python train_rl.py --no-plot          # Skip generating rl_training_plot.png
python train_rl.py --no-train         # Evaluate loaded model vs suggest_poi_by_steps
```

Produces `models/best_model.zip`, `models/ppo_poi_suggestion.zip`, and `rl_training_plot.png`.

### 2. Run demo (`run_thesslink_demo.py`)

Uses RL policy (or `suggest_poi_by_steps` if no model). POIs show **steps** to arrive.

```bash
python run_thesslink_demo.py               # 5 scenarios, RL/steps (minimize steps)
python run_thesslink_demo.py --scenarios 10
python run_thesslink_demo.py --scenarios 0    # Infinite (until window closed)
python run_thesslink_demo.py --no-visualize
```

## Project structure

```
thesslink-rl/
├── cost_function.py         # suggest_poi_by_steps, steps_to_both_arrive, cost helpers
├── poi_env.py                # Gymnasium env for POI suggestion (RL)
├── train_rl.py               # PPO training, suggest_poi_rl()
├── models/                   # RL models (best_model.zip, ppo_poi_suggestion.zip)
├── rl_training_plot.png      # RL training plot
├── run_thesslink_demo.py     # Demo (default: RL/steps)
├── lb-foraging/              # lb-foraging env (visualization)
├── requirements.txt
└── README.md
```

## Steps-based selection

- **steps_to_both_arrive(poi)** = max(Manhattan(agent, poi), Manhattan(human, poi))
- Both move **directly** toward the chosen POI (no circles)
- **suggest_poi_by_steps** picks argmin steps (fallback when no RL model)
- RL policy learns the same objective via PPO

## Reinforcement Learning

- **State:** Normalized positions + cost components per POI
- **Action:** Discrete(3) – which POI to suggest
- **Reward:** `-steps/max_dist` (default) — minimizes time for both to arrive

## Flow

1. **train_rl.py** – Train RL policy (PPO, steps reward) → `models/best_model.zip`
2. **run_thesslink_demo.py** – Load RL model (or use suggest_poi_by_steps) → suggest POI → visualize

## License

Uses [lb-foraging](https://github.com/semitable/lb-foraging) (MIT) for visualization. The `lb-foraging/` folder is a full copy (not a submodule) with modifications for ThessLink: `allow_agent_on_food` and `allow_agent_on_agent` so agents can move onto POIs and share cells.
