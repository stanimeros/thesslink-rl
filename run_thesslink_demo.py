#!/usr/bin/env python3
"""
ThessLink Demo: Distance + Privacy + Energy

1. Run lb-foraging (8x8 quick test) or extend to 64x64
2. Cost function: rank 3 POIs by distance, privacy, energy
3. A* for pathfinding (placeholder), RL for cost optimization (future)

Usage:
  python run_thesslink_demo.py              # Run demo (no window)
  python run_thesslink_demo.py --visualize  # ThessLink env: 1 human, 1 agent, 3 POIs
  python human_play_thesslink.py            # Interactive: pick POI (1,2,3), move agent
"""
import argparse
import time
from typing import cast

import gymnasium as gym
import lbforaging  # Registers LBF environments
from cost_function import rank_pois, cost_function
from lbforaging.foraging.environment import ForagingEnv
from thesslink_env import ThessLinkEnv

parser = argparse.ArgumentParser()
parser.add_argument("--visualize", action="store_true", help="Open LBF visualization window")
parser.add_argument("--env", default="Foraging-8x8-2p-2f-v3", help="Environment name")
args = parser.parse_args()

# --- 1. ThessLink env (1 human, 1 agent, 3 POIs) ---
print("=" * 50)
print("1. ThessLink environment (1 human, 1 agent, 3 POIs)")
print("=" * 50)
pois = [(2, 2), (4, 5), (6, 3)]
thesslink_env = ThessLinkEnv(
    grid_size=(8, 8),
    pois=pois,
    render_mode="human" if args.visualize else None,
)
obs, _ = thesslink_env.reset(seed=42)
print(f"Human: {thesslink_env.human_pos}, Agent: {thesslink_env.agent_pos}")
print(f"POIs: {pois}")

if args.visualize:
    print("\nLaunching interactive play - run human_play_thesslink.py for full controls.")
    print("(1,2,3=select POI, arrows=move agent)")
    for _ in range(30):
        obs, reward, _, _, _ = thesslink_env.step(thesslink_env.action_space.sample())
        thesslink_env.render()
        time.sleep(0.15)
        if reward > 0:
            break
    input("Press Enter to close...")
thesslink_env.close()

# --- 2. Cost function: rank 3 POIs ---
print("\n" + "=" * 50)
print("2. Cost function (Distance + Privacy + Energy)")
print("=" * 50)
agent_pos = (2, 3)
human_pos = (5, 5)
pois = [(1, 1), (4, 4), (6, 2)]  # 3 static POI suggestions
grid_size = (64, 64)

ranked = rank_pois(pois, agent_pos, human_pos, grid_size)
print(f"Agent: {agent_pos}, Human: {human_pos}")
print(f"POIs: {pois}")
print("Ranked (by cost, lower=better):")
for rank, poi, cost in ranked:
    print(f"  #{rank}: POI {poi} -> cost = {cost:.4f}")

# --- 3. Extend lb-foraging for 64x64 (custom registration) ---
print("\n" + "=" * 50)
print("3. Custom 64x64 environment (ThessLink)")
print("=" * 50)

# Register 64x64 variant
from gymnasium import register
register(
    id="ThessLink-64x64-2p-3f-v0",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 2,
        "min_player_level": 1,
        "max_player_level": 2,
        "field_size": (64, 64),
        "min_food_level": 1,
        "max_food_level": None,
        "max_num_food": 3,  # 3 POIs
        "sight": 64,
        "max_episode_steps": 200,
        "force_coop": False,
        "grid_observation": False,
    },
)

env_64 = gym.make("ThessLink-64x64-2p-3f-v0")
obs, info = env_64.reset(seed=123)
lbf_64 = cast(ForagingEnv, env_64.unwrapped)
print(f"64x64 env: agents={lbf_64.n_agents}, field={lbf_64.field_size}")
env_64.close()

print("\nDone. Next: A* pathfinding + RL cost optimization.")
