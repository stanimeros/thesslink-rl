#!/usr/bin/env python3
"""
ThessLink: Cost/reward function + lb-foraging for visualization & evaluation.

1. train_weights.py  - Train cost function weights (distance, privacy, energy)
2. cost_function.py  - Rank POIs, suggest meeting point
3. lb-foraging       - Visualize and evaluate results

Usage:
  python run_thesslink_demo.py              # Cost function + lb-foraging window
  python run_thesslink_demo.py --no-visualize  # Skip window
"""
import argparse
import time
from typing import cast

import gymnasium as gym
import lbforaging
from cost_function import rank_pois, cost_function, suggest_poi
from lbforaging.foraging.environment import ForagingEnv

parser = argparse.ArgumentParser()
parser.add_argument("--no-visualize", action="store_true", help="Skip lb-foraging window")
args = parser.parse_args()

# --- 1. Cost function ---
print("=" * 50)
print("1. Cost function (Distance + Privacy + Energy)")
print("=" * 50)
agent_pos = (2, 3)
human_pos = (5, 5)
pois = [(1, 1), (4, 4), (6, 2)]
grid_size = (64, 64)

ranked = rank_pois(pois, agent_pos, human_pos, grid_size)
suggested = suggest_poi(pois, agent_pos, human_pos)
print(f"Agent: {agent_pos}, Human: {human_pos}")
print(f"POIs: {pois}")
print("Ranked (by cost, lower=better):")
for rank, poi, cost in ranked:
    print(f"  #{rank}: POI {poi} -> cost = {cost:.4f}")
print(f"Suggested: POI {pois[suggested]}")

# --- 2. lb-foraging: visualize & evaluate ---
print("\n" + "=" * 50)
print("2. lb-foraging (visualization & evaluation)")
print("=" * 50)
render_mode = None if args.no_visualize else "human"
env = gym.make("Foraging-8x8-2p-2f-v3", render_mode=render_mode)
obs, info = env.reset(seed=42)
lbf = cast(ForagingEnv, env.unwrapped)
agent_positions = [(p.position[0], p.position[1]) for p in lbf.players if p.position]
print(f"Agents at: {agent_positions}, field: {lbf.field_size}")

if render_mode == "human":
    print("\nRunning episode (close window to exit)...")
    done, truncated = False, False
    while not (done or truncated):
        actions = env.action_space.sample()
        obs, rewards, done, truncated, info = env.step(actions)
        env.render()
        time.sleep(0.2)
env.close()

print("\nDone. Run train_weights.py to train cost function weights.")
