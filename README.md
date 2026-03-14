# ThessLink RL

**Cooperative multi-agent navigation** for meeting point suggestion. A centralized controller learns to navigate two agents to the optimal POI on a grid with static obstacles.

![lb-foraging environment](example.gif)

## Overview

- **Architecture:** Centralized controller — a single policy observes both agents and outputs a joint movement action
- **Environment:** Grid with ~10% static obstacles (connectivity guaranteed), supports 8×8, 32×32, 64×64
- **Goal:** Both agents navigate to the same POI (out of 3 candidates)
- **Target POI:** No explicit target selection — the reward function guides agents toward the cost-optimal POI via progress reward + cost-scaled terminal bonus

## Cost function

The optimal POI minimizes a weighted cost over BFS path distances:

$$\text{cost} = w_{TE_a} \cdot d_A + w_{TE_h} \cdot d_H + w_e \cdot e + w_p \cdot p + w_{TTM} \cdot ttm$$

| Symbol | Formula | Range | Description |
|--------|---------|-------|-------------|
| $d_A$ | $\frac{\text{BFS}(\text{agent1}, \text{POI})}{D_{\max}}$ | $[0, 1]$ | Travel Effort — agent1 (robot) → POI |
| $d_H$ | $\frac{\text{BFS}(\text{agent2}, \text{POI})}{D_{\max}}$ | $[0, 1]$ | Travel Effort — agent2 (human) → POI |
| $D_{\max}$ | $\text{rows} + \text{cols}$ | — | Max distance on grid |
| $e$ | $0.6 \cdot d_A + 0.4 \cdot d_H$ | $[0, 1]$ | Combined locomotion energy (robot weighs more: batteries) |
| $p$ | $1 - d_H$ | $[0, 1]$ | Privacy — location-disclosure risk. POI near human → reveals their location → high cost |
| $ttm$ | $\max(d_A, d_H)$ | $[0, 1]$ | Time-to-Meet |

Default weights: $w_{TE_a} = 0.15,\ w_{TE_h} = 0.25,\ w_e = 0.15,\ w_p = 0.15,\ w_{TTM} = 0.30$

The highest weight is on $ttm$ (time-to-meet), ensuring both agents arrive quickly. Privacy and energy have equal weight ($0.15$), so the policy avoids POIs that would reveal the human's location or drain the robot's battery.

## Reward

Each step the agents receive a shared reward with three components:

$$r = \text{progress} - \text{step\_penalty} + \text{terminal\_bonus}$$

| Component | Value | Description |
|-----------|-------|-------------|
| Progress | $\max\!\bigl(0,\;\frac{\Delta d_1 + \Delta d_2}{2 \cdot D_{\max}}\bigr) \times S$ | Reward for reducing BFS distance to the **cost-optimal** POI ($S = 2.0$). Only positive progress counts — no reward for moving away. |
| Step penalty | $\frac{B}{T_{\max}}$ | Per-step cost encouraging speed ($B = 5.0$, $T_{\max}$ = max steps) |
| Terminal bonus | $B \times (1 + 2 \cdot (1 - \text{cost}))$ | Cost-scaled bonus when both agents meet at **any** POI. Cheaper POI → bigger bonus ($5\text{–}15$). |

The progress reward only measures distance to the cost-optimal POI, giving the agents a **directional signal** toward the best destination. The terminal bonus provides a **3× reward gap** between the cheapest and most expensive POI, so reaching the optimal one is strongly reinforced.

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

`MultiDiscrete([5, 5])` — joint movement action controlling both agents:

| Index | Range | Description |
|-------|-------|-------------|
| 0 | {0, 1, 2, 3, 4} | Agent1 movement (NONE, N, S, W, E) |
| 1 | {0, 1, 2, 3, 4} | Agent2 movement (NONE, N, S, W, E) |

No explicit target selection — the policy learns which POI to navigate to from the observation (cost components + relative vectors) and the reward signal.

DQN and Q-Learning use a `FlatActionWrapper` that maps `Discrete(25)` to the equivalent MultiDiscrete encoding: `flat = move1 × 5 + move2`.

## Algorithms

Three RL categories are compared on the same environment:

| Category | Algorithm | Notes |
|----------|-----------|-------|
| **Tabular RL** | Q-Learning | Compact discrete state (442,368 states); parallel workers with visit-weighted Q-table merging |
| **Deep Value-based RL** | DQN | Off-policy, replay buffer, ε-greedy; `Discrete(25)` via FlatActionWrapper |
| **Policy Gradient (Actor-Critic)** | PPO | On-policy, 6 parallel envs via SubprocVecEnv; native `MultiDiscrete([5,5])` |

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
| Green | POI where both agents arrived (non-optimal) |
| Blue | Cost-optimal baseline POI |
| Cyan | Both agents arrived at the cost-optimal POI |
| Grey | Other POIs |
| Dark | Obstacles |
| Blue label **1** | agent1 |
| Red label **2** | agent2 |

## Project structure

```
thesslink-rl/
├── cost_function.py        # bfs_distance, cost_components, cost_optimal_baseline
├── poi_environment.py      # PoINavigationEnv (35-float obs, MultiDiscrete([5,5]), FlatActionWrapper)
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
