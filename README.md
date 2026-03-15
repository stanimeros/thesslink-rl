# ThessLink RL

**Cooperative multi-agent navigation** for meeting point suggestion. A centralized RL controller learns to guide two agents to the cost-optimal meeting point on a grid with static obstacles.

![Demo](example.gif)

---

## Environment, Agents & Goal

### Environment

A 2D grid world with static obstacles (~10% of cells). Supported sizes: **8×8**, **32×32**, **64×64**. Obstacles are randomly generated each episode with a guaranteed connected free space. Three candidate meeting points (POIs) are placed at random free cells each episode.

### Agents

| Agent | Role | Description |
|-------|------|-------------|
| **Agent 1** | Robot | Autonomous mobile device navigating the grid |
| **Agent 2** | Human | Person navigating toward the meeting point |

Both agents are controlled by a **single centralized policy** that observes the full global state and outputs a joint movement action — one move per agent per step. This "god-camera" design avoids communication overhead and simplifies training.

### Goal

The policy must guide **both agents to meet at the cost-optimal POI** — the candidate meeting point that minimizes a weighted cost over travel effort, energy consumption, privacy risk, and time-to-meet (see [Cost function](#cost-function) below).

The policy is **not told which POI is optimal**. It must infer this from the observation (cost components per POI) and learn the preference through the reward signal.

An episode ends successfully when both agents occupy the same optimal POI cell simultaneously. If they do not meet within `max_steps`, the episode is truncated.

---

## Cost function

The optimal POI minimizes a weighted cost over BFS path distances:

$$\text{cost} = w_{TE_a} \cdot d_A + w_{TE_h} \cdot d_H + w_e \cdot e + w_p \cdot p + w_{TTM} \cdot ttm$$

| Symbol | Formula | Range | Description |
|--------|---------|-------|-------------|
| $d_A$ | $\frac{\text{BFS}(\text{agent1}, \text{POI})}{D_{\max}}$ | $[0, 1]$ | Travel Effort — agent1 (robot) → POI |
| $d_H$ | $\frac{\text{BFS}(\text{agent2}, \text{POI})}{D_{\max}}$ | $[0, 1]$ | Travel Effort — agent2 (human) → POI |
| $D_{\max}$ | $\text{rows} + \text{cols}$ | — | Max distance on grid |
| $e$ | $0.6 \cdot d_A + 0.4 \cdot d_H$ | $[0, 1]$ | Combined locomotion energy (robot weighs more: batteries vs human effort) |
| $p$ | $1 - d_H$ | $[0, 1]$ | Privacy — location-disclosure risk. POI near human → reveals their location → high cost |
| $ttm$ | $\max(d_A, d_H)$ | $[0, 1]$ | Time-to-Meet |

Default weights: $w_{TE_a} = 0.15,\ w_{TE_h} = 0.25,\ w_e = 0.15,\ w_p = 0.15,\ w_{TTM} = 0.30$

The highest weight is on $ttm$ (time-to-meet), ensuring both agents arrive quickly. Privacy and human travel effort together account for 40% of the cost, so the policy avoids POIs that would reveal the human's location or require excessive walking.

---

## Reward

Each step the agents receive a shared reward:

$$r = \text{progress} - \text{step\_penalty} + \text{terminal\_bonus}$$

| Component | Value | Description |
|-----------|-------|-------------|
| Progress | $\max\!\bigl(0,\;\Delta d_{\text{best}}\bigr) \times S$ | Reward for setting a new **personal best** combined BFS distance to the optimal POI ($S = 2.0$). Only positive progress counts — no reward for standing still or moving away. |
| Step penalty | $\frac{B}{T_{\max}}$ | Per-step cost encouraging speed ($B = 5.0$, $T_{\max}$ = max steps) |
| Terminal bonus | $B \times (1 + 2 \cdot (1 - \text{cost}))$ | Cost-scaled bonus when both agents meet at the **optimal POI** ($5$–$15$ depending on cost). Cheaper POI → larger bonus. |

**Anti-oscillation:** Progress is tracked as a running best (`_best_combined_dist`). An agent that oscillates between cells never improves its best distance, so it only accumulates step penalties — naturally discouraging back-and-forth loops.

**Freeze:** Once an agent reaches the optimal POI it stops moving, so the other agent must navigate to join it there.

---

## Observation space

Global observation (35 floats, position-invariant):

| Features | Size | Description |
|----------|------|-------------|
| `delta_a1_to_pois` | 6 | Relative vectors agent1 → each POI (dr, dc × 3), normalized to [-1, 1] |
| `delta_a2_to_pois` | 6 | Relative vectors agent2 → each POI (dr, dc × 3), normalized to [-1, 1] |
| `wall_bits_a1` | 4 | Binary: N/S/W/E blocked for agent1 |
| `wall_bits_a2` | 4 | Binary: N/S/W/E blocked for agent2 |
| `cost_components × 3 POIs` | 15 | (te_a, te_h, energy, privacy, ttm) per POI, in [0, 1] |

Relative vectors make the policy position-invariant: it sees "POI is 3 cells north" instead of absolute coordinates. The cost components give the policy everything it needs to identify the optimal POI without being told explicitly.

---

## Action space

`MultiDiscrete([5, 5])` — joint movement action controlling both agents independently:

| Index | Range | Description |
|-------|-------|-------------|
| 0 | {0..4} | Agent1 movement (NONE, N, S, W, E) |
| 1 | {0..4} | Agent2 movement (NONE, N, S, W, E) |

DQN and Q-Learning use a `FlatActionWrapper` that maps `Discrete(25)` → `[move1, move2]` via `flat = move1 × 5 + move2`.

---

## Algorithms

Three RL categories are compared on the same environment:

| Category | Algorithm | Notes |
|----------|-----------|-------|
| **Tabular RL** | Q-Learning | Compact discrete state (442,368 states); parallel workers with visit-weighted Q-table merging |
| **Deep Value-based RL** | DQN | Off-policy, replay buffer, ε-greedy; `Discrete(25)` via FlatActionWrapper |
| **Policy Gradient (Actor-Critic)** | PPO | On-policy, 6 parallel envs via SubprocVecEnv; native `MultiDiscrete([5,5])` |

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e lb-foraging/
pip install -r requirements.txt
```

---

## Training

```bash
# PPO — 1M steps, 6 parallel envs
python navigation_train.py --algo ppo --grid-size 8

# DQN — 1M steps, 6 parallel envs
python navigation_train.py --algo dqn --grid-size 8

# Q-Learning — 400k episodes, 6 parallel workers
python navigation_train.py --algo qlearning --grid-size 8 --workers 6
```

Supported grid sizes: `8`, `32`, `64`. Each run saves the model and training history under `models/<algo>/`. Training resumes automatically if a saved model is found.

Evaluate a trained model without retraining:

```bash
python navigation_train.py --algo ppo --grid-size 8 --no-train
```

---

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
| Cyan | Both agents arrived at the cost-optimal POI (success) |
| Blue | Cost-optimal POI (not yet reached by both) |
| Grey | Non-optimal POIs |
| Dark | Obstacles |
| Blue label **1** | Agent1 (robot) |
| Red label **2** | Agent2 (human) |

---

## Project structure

```
thesslink-rl/
├── cost_function.py        # bfs_distance, cost_components, cost_optimal_baseline
├── poi_environment.py      # PoINavigationEnv (35-float obs, MultiDiscrete([5,5]), FlatActionWrapper)
├── navigation_train.py     # Training: PPO, DQN (SubprocVecEnv), Q-Learning (parallel workers)
├── demo.py                 # 3-panel navigation demo
├── training.ipynb          # Combined training curves plot
├── models/
│   ├── ppo/                # nav_ppo_<size>.zip + training_history
│   ├── dqn/                # nav_dqn_<size>.zip + training_history
│   └── qlearning/          # nav_qtable_<size>.pkl + training_history
├── lb-foraging/            # lb-foraging env (visualization only)
├── requirements.txt
└── README.md
```

---

## Related improvements / ideas

- **Fairness (minimax):** Minimize max effort instead of sum — balance burden between agents.
- **User preferences:** Learn or adapt weights per user (preference-based RL).
- **Privacy variants:** Crowd exposure, anonymity, distance from home.
- **Energy variants:** Terrain, elevation, vehicle type.
- **Negotiation:** Alternating offers, Pareto-optimal compromise.

---

## License

Uses [lb-foraging](https://github.com/semitable/lb-foraging) (MIT) for visualization. The `lb-foraging/` folder is a full copy with modifications: `allow_agent_on_food` and `allow_agent_on_agent` so agents can move onto POIs and share cells.
