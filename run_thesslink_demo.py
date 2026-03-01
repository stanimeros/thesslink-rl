#!/usr/bin/env python3
"""
ThessLink: Multiple scenarios. Cost function suggests POI, lb-foraging shows H and A movement.

Usage:
  python run_thesslink_demo.py                 # 5 scenarios
  python run_thesslink_demo.py --scenarios 10
  python run_thesslink_demo.py --scenarios 0   # Infinite until window closed
  python run_thesslink_demo.py --no-visualize
"""
import argparse
import math
import random
import time

import gymnasium as gym
import lbforaging
import numpy as np
import pyglet
from pyglet.gl import GL_LINE_LOOP, GL_POLYGON, glColor3ub

from cost_function import cost_components, cost_function, load_weights, rank_pois, suggest_poi
from lbforaging.foraging.environment import Action, ForagingEnv
from lbforaging.foraging.rendering import Viewer

parser = argparse.ArgumentParser()
parser.add_argument("--no-visualize", action="store_true", help="Skip lb-foraging window")
parser.add_argument("--scenarios", type=int, default=5, help="Scenarios to run (0=infinite until window closed)")
args = parser.parse_args()


def sample_positions(grid_size: tuple[int, int], n: int, seed: int | None = None) -> list[tuple[int, int]]:
    """Sample n distinct random positions on grid."""
    rng = random.Random(seed)
    rows, cols = grid_size
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    return rng.sample(cells, n)


def step_toward(pos: tuple[int, int], target: tuple[int, int]) -> Action:
    """Return action (NONE,NORTH,SOUTH,WEST,EAST) to move one step toward target."""
    r, c = pos
    tr, tc = target
    if (r, c) == target:
        return Action.NONE
    if r > tr:
        return Action.NORTH
    if r < tr:
        return Action.SOUTH
    if c > tc:
        return Action.WEST
    if c < tc:
        return Action.EAST
    return Action.NONE


def run_episode(
    env: gym.Env,
    lbf: ForagingEnv,
    viewer: Viewer,
    grid_size: tuple[int, int],
    human_pos: tuple[int, int],
    agent_pos: tuple[int, int],
    pois: list[tuple[int, int]],
    suggested_poi: tuple[int, int],
    weights: tuple[float, float, float],
) -> bool:
    """Run one episode. Returns True if completed, False if window closed."""
    rows, cols = grid_size

    lbf.current_step = 0
    lbf._game_over = False

    # Set state: H and A at positions, 3 food at POIs
    lbf.field = np.zeros((rows, cols), dtype=np.int32)  # type: ignore[assignment]
    for i, poi in enumerate(pois):
        level = 2 if poi == suggested_poi else 1
        lbf.field[poi[0], poi[1]] = level
    lbf.players[0].position = tuple(human_pos)
    lbf.players[1].position = tuple(agent_pos)
    lbf.players[0].level = 1
    lbf.players[1].level = 2
    lbf._gen_valid_moves()

    env.render()
    time.sleep(1.0)

    poi_costs = {
        poi: cost_function(poi, agent_pos, human_pos, grid_size, *weights)
        for poi in pois
    }
    poi_to_label = {
        poi: f"P{i+1}" + ("*" if poi == suggested_poi else "") + f" {poi_costs[poi]:.2f}"
        for i, poi in enumerate(pois)
    }

    def draw_badge_h_a(row: int, col: int, level: int):
        if (row, col) in poi_to_label:
            label = poi_to_label[(row, col)]
        elif level == 1:
            label = "H"
        elif level == 2:
            label = "A"
        else:
            label = str(level)
        resolution = 6
        radius = viewer.grid_size / 7
        badge_x = int(col * (viewer.grid_size + 1) + (3 / 4) * (viewer.grid_size + 1))
        badge_y = int(viewer.height - (viewer.grid_size + 1) * (row + 1) + (1 / 4) * (viewer.grid_size + 1))
        verts = []
        for i in range(resolution):
            angle = 2 * math.pi * i / resolution
            verts += [radius * math.cos(angle) + badge_x, radius * math.sin(angle) + badge_y]
        circle = pyglet.graphics.vertex_list(resolution, ("v2f", verts))
        glColor3ub(255, 255, 255)
        circle.draw(GL_POLYGON)
        glColor3ub(0, 0, 0)
        circle.draw(GL_LINE_LOOP)
        pyglet.text.Label(
            label,
            font_size=8,
            bold=True,
            x=badge_x,
            y=badge_y + 2,
            anchor_x="center",
            anchor_y="center",
            color=(0, 0, 0, 255),
        ).draw()

    viewer._draw_badge = draw_badge_h_a

    def at_or_adjacent(p: tuple[int, int], t: tuple[int, int]) -> bool:
        """True if p is on t or one step away (Manhattan)."""
        return abs(p[0] - t[0]) + abs(p[1] - t[1]) <= 1

    steps_with_both_adjacent = 0
    for _ in range(100):
        if not viewer.isopen:
            return False
        h_pos = (int(lbf.players[0].position[0]), int(lbf.players[0].position[1]))
        a_pos = (int(lbf.players[1].position[0]), int(lbf.players[1].position[1]))
        both_adjacent_not_on = (
            at_or_adjacent(h_pos, suggested_poi)
            and at_or_adjacent(a_pos, suggested_poi)
            and h_pos != suggested_poi
            and a_pos != suggested_poi
        )
        if both_adjacent_not_on:
            act_h = step_toward(h_pos, suggested_poi)
            act_a = Action.NONE
        else:
            act_h = step_toward(h_pos, suggested_poi)
            act_a = step_toward(a_pos, suggested_poi)
        obs, rewards, done, truncated, info = env.step([int(act_h.value), int(act_a.value)])
        env.render()
        time.sleep(0.15)
        h_pos = (int(lbf.players[0].position[0]), int(lbf.players[0].position[1]))
        a_pos = (int(lbf.players[1].position[0]), int(lbf.players[1].position[1]))
        both_on_poi = (h_pos == suggested_poi) and (a_pos == suggested_poi)
        if both_on_poi:
            return True
        both_reached = at_or_adjacent(h_pos, suggested_poi) and at_or_adjacent(a_pos, suggested_poi)
        if both_reached:
            steps_with_both_adjacent += 1
            if steps_with_both_adjacent >= 5:
                return True
        else:
            steps_with_both_adjacent = 0
    return True


def run_with_movement(
    grid_size: tuple[int, int],
    n_scenarios: int,
    rng: random.Random,
):
    """Run multiple scenarios. After each episode completes, start next."""
    env = gym.make(
        "Foraging-8x8-2p-3f-v3",
        render_mode="human",
        allow_agent_on_food=True,
        allow_agent_on_agent=True,
    )
    env.reset(seed=42)
    lbf = env.unwrapped
    assert isinstance(lbf, ForagingEnv)
    lbf._init_render()
    viewer = lbf.viewer
    assert isinstance(viewer, Viewer)

    scenario = 0
    while viewer.isopen:
        if n_scenarios > 0 and scenario >= n_scenarios:
            break
        scenario += 1
        positions = sample_positions(grid_size, 5, rng.randint(0, 2**31 - 1))
        human_pos, agent_pos = positions[0], positions[1]
        pois = positions[2:5]
        weights = load_weights()
        suggested_idx = suggest_poi(pois, agent_pos, human_pos, weights=weights, grid_size=grid_size)
        suggested_poi = pois[suggested_idx]

        cost_lines = []
        for i, poi in enumerate(pois):
            cost = cost_function(poi, agent_pos, human_pos, grid_size, *weights)
            d, p, e = cost_components(poi, agent_pos, human_pos, grid_size)
            marker = " *" if poi == suggested_poi else ""
            cost_lines.append(f"  P{i+1} {poi}: cost={cost:.4f} (d={d:.3f} p={p:.3f} e={e:.3f}){marker}")

        print(f"\n--- Scenario {scenario} ---")
        print(f"H: {human_pos}  A: {agent_pos}  POIs: {pois}")
        print("Costs (lower=better):")
        print("\n".join(cost_lines))
        print(f"Suggested: P{suggested_idx+1} {suggested_poi}")

        completed = run_episode(
            env, lbf, viewer, grid_size, human_pos, agent_pos, pois, suggested_poi, weights
        )
        if not completed:
            break
        for _ in range(5):
            env.render()
            time.sleep(0.1)
        time.sleep(0.5)  # Brief pause between scenarios

    env.close()


def main():
    grid_size = (8, 8)
    rng = random.Random(42)
    n_scenarios = args.scenarios

    print("=" * 50)
    print("ThessLink: Multiple scenarios")
    print("=" * 50)

    if args.no_visualize:
        n_show = min(n_scenarios, 5) if n_scenarios > 0 else 5
        for s in range(n_show):
            positions = sample_positions(grid_size, 5, rng.randint(0, 2**31 - 1))
            human_pos, agent_pos = positions[0], positions[1]
            pois = positions[2:5]
            ranked = rank_pois(pois, agent_pos, human_pos, grid_size)
            suggested_idx = suggest_poi(pois, agent_pos, human_pos, grid_size=grid_size)
            suggested_poi = pois[suggested_idx]
            print(f"Scenario {s+1}: H={human_pos} A={agent_pos} POIs={pois} -> P{suggested_idx+1} {suggested_poi}")
        print("\nRun without --no-visualize to see movement.")
    else:
        print(f"Running {n_scenarios if n_scenarios > 0 else 'infinite'} scenarios. Close window to exit.")
        run_with_movement(grid_size, n_scenarios, rng)


if __name__ == "__main__":
    main()
