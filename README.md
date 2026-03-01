# ThessLink RL

**Reinforcement Learning** for meeting point suggestion. The model takes **Travel Effort** (agent, human distances), **Time-to-Meet**, **energy** (20–80% for human), and **privacy** (basic), calculates a cost for each POI, and selects the **minimum cost**.

![lb-foraging environment](lb-foraging/docs/img/lbf.gif)

## Overview

- **Inputs:** Human position, agent position, 3 POI suggestions (64×64 grid)
- **Cost components per POI:** Travel Effort (agent, human), energy (20–80%), privacy, Time-to-Meet
- **Output:** POI with minimum cost
- **Reward:** `-cost` (RL learns to minimize cost)
- **Baseline:** `suggest_poi` (cost formula) used for RL evaluation
- **Demo:** Shows steps + cost per POI

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e lb-foraging/
pip install -r requirements.txt
```

## Usage

### 1. Train RL policy (`train.py`)

```bash
python train.py                    # Train PPO 50k steps (cost reward), save to models/
python train.py --steps 100000     # More training
python train.py --no-plot          # Skip generating training_plot.png
python train.py --no-train         # Evaluate loaded model vs cost baseline
```

Produces `models/best_model.zip`, `models/ppo_poi_suggestion.zip`, and `training_plot.png`.

![Training plot](training_plot.png)

### 2. Run demo (`demo.py`)

Uses RL policy. Shows **cost** per POI. Exits if model not found.

```bash
python demo.py               # 5 scenarios
python demo.py --scenarios 10
python demo.py --scenarios 0    # Infinite (until window closed)
python demo.py --no-visualize
```

## Project structure

```
thesslink-rl/
├── cost_function.py    # cost_components, cost_function, suggest_poi
├── poi_environment.py  # Gymnasium env for POI suggestion (RL)
├── train.py            # PPO training, suggest_poi_rl()
├── demo.py             # Demo with cost display
├── models/              # RL models (best_model.zip, ppo_poi_suggestion.zip)
├── training_plot.png   # RL training plot
├── lb-foraging/        # lb-foraging env (visualization)
├── requirements.txt
└── README.md
```

## Cost formula

```
cost = w_TE_a×travel_effort_agent + w_TE_h×travel_effort_human + w_energy×energy + w_privacy×privacy + w_TTM×time_to_meet
```

- **Travel Effort (agent, human):** Manhattan distances (agent→POI, human→POI), normalized
- **energy:** 0.2 + 0.6×travel_effort_human (range 20–80%)
- **privacy:** 1 − travel_effort_human (basic)
- **Time-to-Meet:** max(travel_effort_agent, travel_effort_human) — min steps for both to arrive

Lower cost = better suggestion. Default weights: 0.20 each.

## Reinforcement Learning

- **State:** Normalized positions + cost components (Travel Effort, energy, privacy, Time-to-Meet) per POI
- **Action:** Discrete(3) – which POI to suggest
- **Reward:** `-cost` (default) – minimize cost

## Flow

1. **train.py** – Train RL policy (PPO, cost reward) → `models/best_model.zip`
2. **demo.py** – Load RL model → suggest POI → visualize

## License

Uses [lb-foraging](https://github.com/semitable/lb-foraging) (MIT) for visualization. The `lb-foraging/` folder is a full copy (not a submodule) with modifications for ThessLink: `allow_agent_on_food` and `allow_agent_on_agent` so agents can move onto POIs and share cells.
