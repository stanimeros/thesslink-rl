# ThessLink RL

**Cooperative multi-agent navigation** for meeting point suggestion. Two agents learn to navigate together to the optimal POI on a grid with static obstacles, using a shared RL policy.

![lb-foraging environment](example.gif)

## Overview

- **Agents:** Two symmetric agents (agent1, agent2) share a single policy
- **Environment:** Grid with ~10% static obstacles (connectivity guaranteed), supports 8×8, 32×32, 64×64
- **Goal:** Both agents navigate to the same target POI (out of 3 candidates)
- **Target POI:** The policy selects which POI to navigate to each step, guided by the cost function

## Cost function

The optimal POI minimizes a weighted cost over A\* path distances:

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

The highest weight is on $d_H$ (human travel effort), prioritising the human's comfort.

## Reward

Each step the agents receive a shared reward:

$$r = -\text{cost}(\text{chosen POI}) - 0.01$$

The step penalty encourages reaching the target quickly. The policy is penalised proportionally to how suboptimal its POI choice is at every step.

## Observation space

Per agent (27 floats):

| Features | Size | Description |
|----------|------|-------------|
| `self_pos` | 2 | Normalized (row, col) of this agent |
| `other_pos` | 2 | Normalized (row, col) of the other agent |
| `wall_bits_self` | 4 | Binary: N/S/W/E directions blocked for this agent |
| `wall_bits_other` | 4 | Binary: N/S/W/E directions blocked for the other agent |
| `cost_components × 3 POIs` | 15 | BFS cost components (te_a, te_h, energy, privacy, ttm) for each POI |

Wall bits prevent the model from getting stuck in deterministic loops at obstacles.

## Action space

`Discrete(15)` — composite action: `target_idx * 5 + move`

- `target_idx ∈ {0, 1, 2}` — which POI to navigate to
- `move ∈ {NONE, N, S, W, E}` — movement direction

## Algorithms

Three RL categories are compared on the same environment:

| Category | Algorithm | Notes |
|----------|-----------|-------|
| **Tabular RL** | Q-Learning | Compact discrete state (1,152 states); tractable for tabular RL |
| **Deep Value-based RL** | DQN | Off-policy, replay buffer, ε-greedy exploration |
| **Policy Gradient (Actor-Critic)** | PPO | On-policy, 8 parallel envs via SubprocVecEnv |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e lb-foraging/
pip install -r requirements.txt
```

## Training

```bash
# PPO — 500k steps, 8 parallel envs (~2 min on Apple M4)
python navigation_train.py --algo ppo --grid-size 8

# DQN — 500k steps (~2.5 min on Apple M4)
python navigation_train.py --algo dqn --grid-size 8

# Q-Learning — 200k episodes
python navigation_train.py --algo qlearning --grid-size 8
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
├── cost_function.py        # astar_distance, cost_components, cost_optimal_baseline
├── poi_environment.py      # PoINavigationEnv (27-float obs, wall bits, shared policy)
├── navigation_train.py     # Training: PPO (SubprocVecEnv), DQN, Q-Learning
├── demo.py                 # 3-panel navigation demo
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
