# ThessLink RL

**Cooperative multi-agent navigation** for meeting point suggestion. A centralized controller learns to navigate two agents to the optimal POI on a grid with static obstacles.

![lb-foraging environment](example.gif)

## Overview

- **Architecture:** Centralized "god-camera" controller — a single policy observes both agents and outputs a joint action
- **Environment:** Grid with ~10% static obstacles (connectivity guaranteed), supports 8×8, 32×32, 64×64
- **Goal:** Both agents navigate to the same target POI (out of 3 candidates)
- **Target POI:** The policy selects which POI to navigate to each step, guided by the cost function

## Cost function

The optimal POI minimizes a weighted cost over BFS path distances:

$$\text{cost} = w_{TE_a} \cdot d_A + w_{TE_h} \cdot d_H + w_e \cdot e + w_p \cdot p + w_{TTM} \cdot ttm$$

| Symbol | Formula | Description |
|--------|---------|-------------|
| $d_A$ | $\frac{\text{BFS}(\text{agent1}, \text{POI})}{D_{\max}}$ | Travel Effort (agent1 → POI), normalized |
| $d_H$ | $\frac{\text{BFS}(\text{agent2}, \text{POI})}{D_{\max}}$ | Travel Effort (agent2 → POI), normalized |
| $D_{\max}$ | $\text{rows} + \text{cols}$ | Max distance on grid |
| $e$ | $0.2 + 0.6 \cdot d_H$ | Energy expenditure, range $[0.2, 0.8]$ |
| $p$ | $1 - d_H$ | Privacy (higher when POI near agent2) |
| $ttm$ | $\max(d_A, d_H)$ | Time-to-Meet |

Default weights: $w_{TE_a} = 0.20,\ w_{TE_h} = 0.35,\ w_e = 0.10,\ w_p = 0.10,\ w_{TTM} = 0.25$

The highest weight is on $d_H$ (human travel effort), prioritising the human's comfort.

## Reward

Each step the agents receive a shared reward with four components:

$$r = \text{progress} - \text{step\_penalty} - \text{switch\_penalty} + \text{terminal\_bonus}$$

| Component | Value | Description |
|-----------|-------|-------------|
| Progress | $\frac{\Delta d_1 + \Delta d_2}{2 \cdot D_{\max}} \times S$ | Reward for reducing BFS distance to target ($S = 2.0$) |
| Step penalty | $\frac{B}{2 \cdot T_{\max}}$ | Per-step cost encouraging speed ($B = 5.0$, $T_{\max}$ = max steps) |
| Switch penalty | $0.05 \times B$ | Penalty each time the target POI changes |
| Terminal bonus | $B$ | Bonus when both agents reach the target |

The terminal bonus ($B = 5.0$) always outweighs the total accumulated step penalties, ensuring arrival is always rewarded.

## Observation space

Global observation (35 floats, position-invariant):

| Features | Size | Description |
|----------|------|-------------|
| `delta_a1_to_pois` | 6 | Relative vectors agent1 → each POI (dr, dc × 3), normalized to [-1, 1] |
| `delta_a2_to_pois` | 6 | Relative vectors agent2 → each POI (dr, dc × 3), normalized to [-1, 1] |
| `wall_bits_a1` | 4 | Binary: N/S/W/E blocked for agent1 |
| `wall_bits_a2` | 4 | Binary: N/S/W/E blocked for agent2 |
| `cost_components × 3 POIs` | 15 | (te_a, te_h, energy, privacy, ttm) per POI |

Relative vectors make the policy position-invariant: it sees "POI is 3 cells north" instead of absolute coordinates.

## Action space

`MultiDiscrete([3, 5, 5])` — joint action controlling both agents:

| Index | Range | Description |
|-------|-------|-------------|
| 0 | {0, 1, 2} | Target POI index |
| 1 | {0, 1, 2, 3, 4} | Agent1 movement (NONE, N, S, W, E) |
| 2 | {0, 1, 2, 3, 4} | Agent2 movement (NONE, N, S, W, E) |

DQN and Q-Learning use a `FlatActionWrapper` that maps `Discrete(75)` to the equivalent MultiDiscrete encoding: `flat = target × 25 + move1 × 5 + move2`.

## Algorithms

Three RL categories are compared on the same environment:

| Category | Algorithm | Notes |
|----------|-----------|-------|
| **Tabular RL** | Q-Learning | Compact discrete state (442,368 states); parallel workers with Q-table merging |
| **Deep Value-based RL** | DQN | Off-policy, replay buffer, ε-greedy; `Discrete(75)` via FlatActionWrapper |
| **Policy Gradient (Actor-Critic)** | PPO | On-policy, 6 parallel envs via SubprocVecEnv; native `MultiDiscrete([3,5,5])` |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e lb-foraging/
pip install -r requirements.txt
```

## Training

```bash
# PPO — 1M steps, 6 parallel envs
python navigation_train.py --algo ppo --grid-size 8

# DQN — 1M steps, 6 parallel envs
python navigation_train.py --algo dqn --grid-size 8

# Q-Learning — 400k episodes, 6 parallel workers
python navigation_train.py --algo qlearning --grid-size 8 --workers 6
```

Supported grid sizes: `8`, `32`, `64`. Each run saves the model and training history under `models/<algo>/`.

Evaluate a trained model without retraining:

```bash
python navigation_train.py --algo ppo --grid-size 8 --no-train
```

## Demo

3-panel visualization: Q-Learning, DQN, and PPO run the **same scenario** simultaneously.

```bash
python demo.py --grid-size 8    # 5 scenarios (default)
python demo.py --scenarios 10
python demo.py --scenarios 0    # Infinite until window closed
```

POI colour coding per panel:

| Colour | Meaning |
|--------|---------|
| Green | POI chosen by the model (model's target) |
| Blue | POI chosen by the cost-optimal baseline |
| Cyan | Model and baseline agree on this POI |
| Grey | Other POIs |
| Dark | Obstacles |
| Blue label **1** | agent1 |
| Red label **2** | agent2 |

## Project structure

```
thesslink-rl/
├── cost_function.py        # bfs_distance, cost_components, cost_optimal_baseline
├── poi_environment.py      # PoINavigationEnv (35-float obs, MultiDiscrete, FlatActionWrapper)
├── navigation_train.py     # Training: PPO, DQN (SubprocVecEnv), Q-Learning (parallel workers)
├── demo.py                 # 3-panel navigation demo
├── training.ipynb          # Interactive training notebook
├── models/
│   ├── ppo/                # nav_ppo_<size>.zip + training_history
│   ├── dqn/                # nav_dqn_<size>.zip + training_history
│   └── qlearning/          # nav_qtable_<size>.pkl + training_history
├── lb-foraging/            # lb-foraging env (visualization only)
├── requirements.txt
└── README.md
```

## Related improvements / ideas

- **Fairness (minimax):** Minimize max effort instead of sum — balance burden between agents.
- **User preferences:** Learn or adapt weights per user (preference-based RL).
- **Privacy variants:** Crowd exposure, anonymity, distance from home.
- **Energy variants:** Terrain, elevation, vehicle type.
- **Negotiation:** Alternating offers, Pareto-optimal compromise.

## License

Uses [lb-foraging](https://github.com/semitable/lb-foraging) (MIT) for visualization. The `lb-foraging/` folder is a full copy with modifications: `allow_agent_on_food` and `allow_agent_on_agent` so agents can move onto POIs and share cells.
