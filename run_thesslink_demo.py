#!/usr/bin/env python3
"""
ThessLink: Cost function suggests best meeting POI. lb-foraging shows H and A movement.

Usage:
  python run_thesslink_demo.py              # lb-foraging window: agents move to suggested POI
  python run_thesslink_demo.py --no-visualize  # Skip window
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

from cost_function import rank_pois, suggest_poi
from lbforaging.foraging.environment import Action, ForagingEnv
from lbforaging.foraging.rendering import Viewer

parser = argparse.ArgumentParser()
parser.add_argument("--no-visualize", action="store_true", help="Skip lb-foraging window")
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


def run_with_movement(
    grid_size: tuple[int, int],
    human_pos: tuple[int, int],
    agent_pos: tuple[int, int],
    pois: list[tuple[int, int]],
    suggested_poi: tuple[int, int],
):
    """Use lb-foraging to show H and A moving toward suggested POI."""
    rows, cols = grid_size

    # Create env with 2 players, 3 food (all 3 POIs)
    env = gym.make(
        "Foraging-8x8-2p-3f-v3",
        render_mode="human",
    )
    env.reset(seed=42)
    lbf = env.unwrapped
    assert isinstance(lbf, ForagingEnv)

    # Override: H and A at our positions, 3 food at all POIs (suggested=2, others=1)
    lbf.field = np.zeros((rows, cols), dtype=np.int32)  # type: ignore[assignment]
    for i, poi in enumerate(pois):
        level = 2 if poi == suggested_poi else 1  # suggested=2 (P*), others=1 (P1,P2,P3)
        lbf.field[poi[0], poi[1]] = level
    lbf.players[0].position = list(human_pos)
    lbf.players[1].position = list(agent_pos)
    lbf.players[0].level = 1  # H
    lbf.players[1].level = 2  # A
    lbf._gen_valid_moves()

    lbf._init_render()
    viewer = lbf.viewer
    assert viewer is not None

    # Replace badge labels: H, A, P1/P2/P3 (suggested=*)
    poi_to_label = {poi: f"P{i+1}" + ("*" if poi == suggested_poi else "") for i, poi in enumerate(pois)}

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
        radius = viewer.grid_size / 5
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
            font_size=12,
            bold=True,
            x=badge_x,
            y=badge_y + 2,
            anchor_x="center",
            anchor_y="center",
            color=(0, 0, 0, 255),
        ).draw()

    viewer._draw_badge = draw_badge_h_a

    done = False
    step_count = 0
    max_steps = 100

    while viewer.isopen and not done and step_count < max_steps:
        # Move both toward suggested POI
        h_pos = (int(lbf.players[0].position[0]), int(lbf.players[0].position[1]))
        a_pos = (int(lbf.players[1].position[0]), int(lbf.players[1].position[1]))
        act_h = step_toward(h_pos, suggested_poi)
        act_a = step_toward(a_pos, suggested_poi)

        actions = [int(act_h.value), int(act_a.value)]
        obs, rewards, done, truncated, info = env.step(actions)
        step_count += 1

        env.render()
        time.sleep(0.3)

        # Both adjacent to POI (agents can't stand on food cell)
        def adjacent(p, t):
            return abs(p[0] - t[0]) + abs(p[1] - t[1]) == 1
        if adjacent(h_pos, suggested_poi) and adjacent(a_pos, suggested_poi):
            done = True

    env.close()


def main():
    grid_size = (8, 8)
    seed = 42

    # Random human, agent, 3 POIs
    positions = sample_positions(grid_size, 5, seed)
    human_pos = positions[0]
    agent_pos = positions[1]
    pois = positions[2:5]

    # Cost function suggests best POI
    ranked = rank_pois(pois, agent_pos, human_pos, grid_size)
    suggested_idx = suggest_poi(pois, agent_pos, human_pos, grid_size=grid_size)
    suggested_poi = pois[suggested_idx]

    print("=" * 50)
    print("ThessLink: Cost function suggests meeting point")
    print("=" * 50)
    print(f"H (human): {human_pos}")
    print(f"A (agent): {agent_pos}")
    print(f"POIs: P1={pois[0]}, P2={pois[1]}, P3={pois[2]}")
    print("Ranked (by cost, lower=better):")
    for rank, poi, cost in ranked:
        mark = " <- suggested" if poi == suggested_poi else ""
        print(f"  #{rank}: {poi} cost={cost:.4f}{mark}")
    print(f"Suggested meeting point: P{suggested_idx + 1} {suggested_poi}")

    if not args.no_visualize:
        print("\nlb-foraging: H and A move toward P* (suggested). P1, P2, P3 = ignored. Close window to exit.")
        run_with_movement(grid_size, human_pos, agent_pos, pois, suggested_poi)
    else:
        print("\nRun without --no-visualize to see movement.")


if __name__ == "__main__":
    main()
