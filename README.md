# ThessLink RL

**Cooperative multi-agent navigation** for meeting point suggestion. Two agents learn to navigate together to the optimal POI on a 64×64 grid with static obstacles, using a shared RL policy.

![lb-foraging environment](example.gif)

## Overview

- **Agents:** Two symmetric agents (agent1, agent2) share a single policy
- **Environment:** 64×64 grid with ~10% static obstacles (connectivity guaranteed)
- **Goal:** Both agents navigate to the same target POI
- **Target POI:** Selected at episode start using the A\* cost function (minimum weighted cost)
- **Baseline:** Greedy A\* — always move toward the nearest-human POI without learning

## Cost function

The target POI is chosen by minimizing a weighted cost over A\* path distances:

$$\text{cost} = w_{TE_a} \cdot d_A + w_{TE_h} \cdot d_H + w_e \cdot e + w_p \cdot p + w_{TTM} \cdot ttm$$

| Symbol | Formula | Description |
|--------|---------|-------------|
| $d_A$ | $\frac{\text{A*}(\text{agent1}, \text{POI})}{D_{\max}}$ | Travel Effort (agent1 → POI), normalized |
| $d_H$ | $\frac{\text{A*}(\text{agent2}, \text{POI})}{D_{\max}}$ | Travel Effort (agent2 → POI), normalized |
| $D_{\max}$ | $\text{rows} + \text{cols}$ | Max distance on grid |
| $e$ | $0.2 + 0.6 \cdot d_H$ | Energy expenditure, range $[0.2, 0.8]$ |
| $p$ | $1 - d_H$ | Privacy (higher when POI near agent2) |
| $ttm$ | $\max(d_A, d_H)$ | Time-to-Meet |

Default weights: $w_{TE_a} = 0.20,\ w_{TE_h} = 0.35,\ w_e = 0.10,\ w_p = 0.10,\ w_{TTM} = 0.25$

## Reward

Each step the agents receive a shared reward:

$$r = -0.01 + \frac{(\Delta d_1 + \Delta d_2)}{D_{\max}}$$

where $\Delta d_i$ is the reduction in A\* distance to the target POI for agent $i$. On arrival both agents receive an additional terminal reward of $-\text{cost}$.

## Algorithms

Three RL categories are compared on the same environment:

| Category | Algorithm | File |
|----------|-----------|------|
| **Tabular RL** | Q-Learning | `navigation_train.py --algo qlearning` |
| **Deep Value-based RL** | DQN | `navigation_train.py --algo dqn` |
| **Policy Gradient (Actor-Critic)** | PPO | `navigation_train.py --algo ppo` |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e lb-foraging/
pip install -r requirements.txt
```

## Training

```bash
# PPO — Policy Gradient (Actor-Critic)
python navigation_train.py --algo ppo --steps 200000

# DQN — Deep Value-based RL
python navigation_train.py --algo dqn --steps 200000

# Q-Learning — Tabular RL
python navigation_train.py --algo qlearning --episodes 500000
```

Each run saves the model and a training plot (`training_plot_nav_<algo>.png`) with 3 subplots:
1. Mean Episode Reward
2. Success Rate (both agents arrived at target)
3. Agreement with nearest-human baseline

Evaluate a trained model without retraining:

```bash
python navigation_train.py --algo ppo --no-train
```

## Demo

4-panel visualization: all three algorithms + baseline run the **same scenario** simultaneously.

```bash
python demo.py                 # 5 scenarios
python demo.py --scenarios 10
python demo.py --scenarios 0   # Infinite until window closed
```

Color coding per panel:
- **1** (blue) = agent1, **2** (red) = agent2
- **T** (green) = target POI
- **P1/P2/P3** (grey) = non-target POIs
- **#** (dark) = obstacles

## Project structure

```
thesslink-rl/
├── cost_function.py        # astar_distance, nearest_human_baseline
├── poi_environment.py      # PoINavigationEnv (multi-step, obstacles, shared policy)
├── navigation_train.py     # Training: PPO, DQN, Q-Learning
├── demo.py                 # 4-panel navigation demo
├── models/
│   ├── ppo/                # nav_ppo.zip
│   ├── dqn/                # nav_dqn.zip
│   └── qlearning/          # nav_qtable.pkl
├── training_plot_nav_ppo.png
├── training_plot_nav_dqn.png
├── training_plot_nav_qlearning.png
├── lb-foraging/            # lb-foraging env (visualization)
├── requirements.txt
└── README.md
```

## Observation space

Per agent (19 floats):

| Features | Size | Description |
|----------|------|-------------|
| `self_pos` | 2 | Normalized (row, col) of this agent |
| `other_pos` | 2 | Normalized (row, col) of the other agent |
| `cost_components × 3 POIs` | 15 | A\* cost components for each POI |

## Related improvements / ideas

- **Fairness (minimax):** Minimize max effort instead of sum — balance burden between agents.
- **User preferences:** Learn or adapt weights per user (preference-based RL).
- **Privacy variants:** Crowd exposure, anonymity, distance from home.
- **Energy variants:** Terrain, elevation, vehicle type.
- **Negotiation:** Alternating offers, Pareto-optimal compromise.

## License

Uses [lb-foraging](https://github.com/semitable/lb-foraging) (MIT) for visualization. The `lb-foraging/` folder is a full copy with modifications: `allow_agent_on_food` and `allow_agent_on_agent` so agents can move onto POIs and share cells.
