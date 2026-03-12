#!/usr/bin/env python3
"""
ThessLink: Multiple scenarios. RL-based POI suggestion (3 POIs), lb-foraging shows H and A movement.

Usage:
  python demo.py                    # 5 scenarios, PPO model (default)
  python demo.py --model ppo        # Policy-based (PPO)
  python demo.py --model dqn        # Value-based (DQN)
  python demo.py --model cost       # Cost baseline (no RL)
  python demo.py --scenarios 10
  python demo.py --scenarios 0      # Infinite until window closed
  python demo.py --no-visualize
"""
import argparse
import random
import time

import gymnasium as gym
import lbforaging  # pyright: ignore[reportMissingImports]
import numpy as np
import pyglet

from cost_function import cost_components, nearest_human_baseline
from lbforaging.foraging.environment import Action, ForagingEnv  # pyright: ignore[reportMissingImports]
from lbforaging.foraging.rendering import Viewer  # pyright: ignore[reportMissingImports]

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["ppo", "dqn", "qlearning", "cost"], default="ppo", help="Model for POI suggestion (default: ppo)")
parser.add_argument("--no-visualize", action="store_true", help="Skip lb-foraging window")
parser.add_argument("--scenarios", type=int, default=5, help="Scenarios to run (0=infinite until window closed)")
args = parser.parse_args()

N_POIS = 3


def get_suggestor(model: str):
    """Return suggest function for given model: ppo, dqn, or cost."""
    if model == "cost":
        return lambda pois, agent_pos, human_pos, grid_size=(64, 64): nearest_human_baseline(
            pois, human_pos
        )
    if model == "ppo":
        from policy_based_train import suggest_poi_rl
        return suggest_poi_rl
    if model == "dqn":
        from value_based_train import suggest_poi_dqn
        return suggest_poi_dqn
    if model == "qlearning":
        from tabular_train import suggest_poi_qlearning
        return suggest_poi_qlearning
    raise ValueError(f"Unknown model: {model}")


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


COLOR_MODEL = (0, 140, 0, 255)      # green: model suggestion
COLOR_BASELINE = (200, 0, 0, 255)   # red: nearest-human baseline
COLOR_NEUTRAL = (100, 100, 100, 255) # grey: neither
COLOR_HUMAN = (0, 0, 0, 255)        # black for H
COLOR_AGENT = (128, 128, 128, 255)   # grey for A


def run_episode(
    env: gym.Env,
    lbf: ForagingEnv,
    viewer: Viewer,
    grid_size: tuple[int, int],
    human_pos: tuple[int, int],
    agent_pos: tuple[int, int],
    pois: list[tuple[int, int]],
    suggested_poi: tuple[int, int],
    baseline_poi: tuple[int, int],
) -> bool:
    """Run one episode. Returns True if completed, False if window closed."""
    rows, cols = grid_size

    lbf.current_step = 0
    lbf._game_over = False

    # Set state: H and A at positions, N food at POIs
    lbf.field = np.zeros((rows, cols), dtype=np.int32)  # type: ignore[assignment]
    for i, poi in enumerate(pois):
        level = 2 if poi == suggested_poi else 1
        lbf.field[poi[0], poi[1]] = level
    lbf.players[0].position = tuple(human_pos)
    lbf.players[1].position = tuple(agent_pos)
    lbf.players[0].level = 1
    lbf.players[1].level = 2
    lbf._gen_valid_moves()

    # Color: green = model suggestion, red = nearest-human baseline, grey = neither
    # If model and baseline agree, show green (model wins)
    poi_to_label_and_color: dict[tuple[int, int], tuple[str, tuple[int, int, int, int]]] = {}
    for i, poi in enumerate(pois):
        label = f"P{i+1}"
        if poi == suggested_poi:
            color = COLOR_MODEL
        elif poi == baseline_poi:
            color = COLOR_BASELINE
        else:
            color = COLOR_NEUTRAL
        poi_to_label_and_color[poi] = (label, color)

    def draw_badge_h_a(row: int, col: int, level: int):
        p0, p1 = lbf.players[0].position, lbf.players[1].position
        pos0 = (int(p0[0]), int(p0[1])) if p0 is not None else (-1, -1)
        pos1 = (int(p1[0]), int(p1[1])) if p1 is not None else (-1, -1)
        cell = (row, col)
        # POI cells always show POI label (H/A hidden when they arrive)
        if cell in poi_to_label_and_color:
            label, color = poi_to_label_and_color[cell]
        elif pos0 == cell or pos1 == cell:
            if pos0 == cell:
                label, color = "H", COLOR_HUMAN
            else:
                label, color = "A", COLOR_AGENT
        else:
            label, color = str(level), COLOR_AGENT

        gs = viewer.grid_size + 1
        badge_x = col * gs + gs / 2
        badge_y = viewer.height - gs * (row + 1) + gs / 2
        # Offset label for edge cells so it stays visible (avoid clipping)
        dx = 0
        dy = 0
        if row == 0:
            dy = -8   # top row: move label down
        elif row == rows - 1:
            dy = 8    # bottom row: move label up
        if col == 0:
            dx = 12   # left edge: move label right
        elif col == cols - 1:
            dx = -12  # right edge: move label left
        x = int(badge_x + dx)
        y = int(badge_y + dy)
        pyglet.text.Label(
            label,
            font_size=10,
            bold=False,
            x=x,
            y=y,
            anchor_x="center",
            anchor_y="center",
            color=color,
        ).draw()

    viewer._draw_badge = draw_badge_h_a

    env.render()
    time.sleep(1)

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
    suggest_poi,
):
    """Run multiple scenarios. After each episode completes, start next."""
    n_pois = N_POIS
    env_id = "Foraging-64x64-2p-3f-v3"
    env = gym.make(
        env_id,
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
    # Cell size for 64x64 grid (smaller to fit window)
    viewer.grid_size = 9
    viewer.width = 1 + viewer.cols * (viewer.grid_size + 1)
    viewer.height = 1 + viewer.rows * (viewer.grid_size + 1)
    viewer.window.set_size(viewer.width, viewer.height)

    scenario = 0
    while viewer.isopen:
        if n_scenarios > 0 and scenario >= n_scenarios:
            break
        scenario += 1
        n_positions = 2 + n_pois
        positions = sample_positions(grid_size, n_positions, rng.randint(0, 2**31 - 1))
        human_pos, agent_pos = positions[0], positions[1]
        pois = positions[2:n_positions]
        suggested_idx = suggest_poi(pois, agent_pos, human_pos, grid_size=grid_size)
        suggested_poi = pois[suggested_idx]
        baseline_idx = nearest_human_baseline(pois, human_pos)
        baseline_poi = pois[baseline_idx]

        lines = []
        for i, poi in enumerate(pois):
            te_a, te_h, e, p, ttm = cost_components(poi, agent_pos, human_pos, grid_size)
            tags = []
            if poi == suggested_poi:
                tags.append("MODEL")
            if poi == baseline_poi:
                tags.append("BASELINE")
            marker = f" [{', '.join(tags)}]" if tags else ""
            lines.append(f"  P{i+1} {poi}: dA={te_a:.2f} dH={te_h:.2f} ttm={ttm:.2f}{marker}")
        print(f"\n--- Scenario {scenario} ---")
        print(f"H: {human_pos}  A: {agent_pos}  POIs: {pois}")
        print("\n".join(lines))

        completed = run_episode(
            env, lbf, viewer, grid_size, human_pos, agent_pos, pois, suggested_poi, baseline_poi,
        )
        if not completed:
            break
        for _ in range(5):
            env.render()
            time.sleep(0.1)
        time.sleep(0.5)  # Brief pause between scenarios

    env.close()


def _run_with_movement(grid_size, n_scenarios, rng):
    suggest_poi = get_suggestor(args.model)
    run_with_movement(grid_size, n_scenarios, rng, suggest_poi)


def main():
    grid_size = (64, 64)
    rng = random.Random(42)
    n_scenarios = args.scenarios

    print("=" * 50)
    print(f"ThessLink: Multiple scenarios (model={args.model}, 3 POIs)")
    print("=" * 50)

    if args.no_visualize:
        n_show = min(n_scenarios, 5) if n_scenarios > 0 else 5
        suggest_poi = get_suggestor(args.model)
        for s in range(n_show):
            n_positions = 2 + N_POIS
            positions = sample_positions(grid_size, n_positions, rng.randint(0, 2**31 - 1))
            human_pos, agent_pos = positions[0], positions[1]
            pois = positions[2:n_positions]
            suggested_idx = suggest_poi(pois, agent_pos, human_pos, grid_size=grid_size)
            suggested_poi = pois[suggested_idx]
            baseline_idx = nearest_human_baseline(pois, human_pos)
            baseline_poi = pois[baseline_idx]
            lines = []
            for i, poi in enumerate(pois):
                tags = []
                if poi == suggested_poi:
                    tags.append("MODEL")
                if poi == baseline_poi:
                    tags.append("BASELINE")
                marker = f" [{', '.join(tags)}]" if tags else ""
                lines.append(f"  P{i+1} {poi}{marker}")
            print(f"Scenario {s+1}: H={human_pos} A={agent_pos} POIs={pois}")
            print("\n".join(lines))
        print("\nRun without --no-visualize to see movement.")
    else:
        print(f"Running {n_scenarios if n_scenarios > 0 else 'infinite'} scenarios, {N_POIS} POIs (model={args.model}). Close window to exit.")
        _run_with_movement(grid_size, n_scenarios, rng)


if __name__ == "__main__":
    main()
