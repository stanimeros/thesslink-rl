#!/usr/bin/env python3
"""
ThessLink Navigation Demo — 3-panel cooperative navigation with obstacles.

Three panels run the same scenario (same positions + obstacles) simultaneously,
each driven by a different algorithm: Q-Learning, DQN, PPO.

POI colours:
  Green  — POI chosen by the model (model's target)
  Blue   — POI chosen by the cost-optimal baseline
  Cyan   — POI where model and baseline agree

Usage:
  python demo.py                          # 5 scenarios, 64x64 grid
  python demo.py --grid-size 8            # 8x8 grid
  python demo.py --grid-size 32           # 32x32 grid
  python demo.py --scenarios 10
  python demo.py --scenarios 0            # Infinite until window closed
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path
from typing import Callable

import gymnasium as gym
import lbforaging  # pyright: ignore[reportMissingImports]
import numpy as np
import pyglet
from pyglet.gl import GL_COLOR_BUFFER_BIT, glClear, glClearColor, glPopMatrix, glPushMatrix, glTranslatef

from lbforaging.foraging.environment import ForagingEnv  # pyright: ignore[reportMissingImports]
from lbforaging.foraging.rendering import Viewer  # pyright: ignore[reportMissingImports]
from navigation_train import _NAV_ACTIONS, _discretize_nav
from poi_environment import PoINavigationEnv

parser = argparse.ArgumentParser()
parser.add_argument("--scenarios", type=int, default=5, help="Scenarios to run (0=infinite until window closed)")
parser.add_argument("--grid-size", type=int, choices=[8, 32, 64], default=8,
                    help="Grid size to use (8, 32, or 64). Loads the corresponding trained models.")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_DIR = Path(__file__).parent / "models"
TITLE_H = 28

_CELL_PX_MAP = {8: 28, 32: 10, 64: 4}

_ENV_ID_MAP = {
    8:  "Foraging-8x8-2p-3f-v3",
    32: "Foraging-32x32-2p-3f-v3",
    64: "Foraging-64x64-2p-3f-v3",
}

COLOR_AGENT1 = (0, 0, 180, 255)         # blue: agent1
COLOR_AGENT2 = (180, 0, 0, 255)         # red: agent2
COLOR_MODEL_TARGET = (0, 160, 0, 255)   # green: model's chosen POI
COLOR_OPTIMAL_POI = (0, 100, 220, 255)  # blue: cost-optimal POI
COLOR_AGREE_POI = (0, 200, 200, 255)    # cyan: model and baseline agree
COLOR_OTHER_POI = (110, 110, 110, 255)  # grey: other POIs
COLOR_OBSTACLE = (50, 50, 50, 255)      # dark: obstacles

PANEL_TITLES = ["Q-Learning", "DQN", "PPO"]

# ---------------------------------------------------------------------------
# Load navigation policies
# ---------------------------------------------------------------------------

PredictFn = Callable[[np.ndarray], int]


def _load_ppo(size_tag: str) -> PredictFn:
    from stable_baselines3 import PPO
    model = PPO.load(str(MODEL_DIR / "ppo" / f"nav_ppo_{size_tag}"))
    def predict(obs: np.ndarray) -> int:
        a, _ = model.predict(obs, deterministic=True)
        return int(a)
    return predict


def _load_dqn(size_tag: str) -> PredictFn:
    from stable_baselines3 import DQN
    model = DQN.load(str(MODEL_DIR / "dqn" / f"nav_dqn_{size_tag}"))
    def predict(obs: np.ndarray) -> int:
        a, _ = model.predict(obs, deterministic=True)
        return int(a)
    return predict


def _load_qlearning(size_tag: str) -> PredictFn:
    model_path = MODEL_DIR / "qlearning" / f"nav_qtable_{size_tag}.pkl"
    with open(model_path, "rb") as f:
        q_table: dict = pickle.load(f)
    def predict(obs: np.ndarray) -> int:
        return int(np.argmax(q_table.get(_discretize_nav(obs), np.zeros(_NAV_ACTIONS))))
    return predict


def _get_optimal_idx(nav_env: PoINavigationEnv) -> int:
    """Return the cost-optimal POI index for the current env state."""
    from cost_function import cost_optimal_baseline, DEFAULT_WEIGHTS
    return cost_optimal_baseline(
        nav_env._pois, nav_env._init_agent1_pos, nav_env._init_agent2_pos,
        nav_env._obstacles, DEFAULT_WEIGHTS, nav_env.grid_size,
    )


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _make_badge_fn(
    nav_env: PoINavigationEnv,
    viewer: Viewer,
    panel_h: int,
    optimal_idx: int,
) -> Callable:
    """Return a _draw_badge function for one panel, using panel-local coordinates.

    POI colours:
      Green  — model's chosen target
      Blue   — cost-optimal baseline target
      Cyan   — both agree on this POI
      Grey   — other POIs
    """
    rows, cols = nav_env.grid_size

    def draw_badge(row: int, col: int, level: int) -> None:
        cell = (row, col)
        obstacles = nav_env._obstacles
        model_target = nav_env._target_poi
        optimal_poi = nav_env._pois[optimal_idx]
        pois = nav_env._pois
        a1 = nav_env._agent1_pos
        a2 = nav_env._agent2_pos

        if cell in obstacles:
            label, color = "#", COLOR_OBSTACLE
        elif cell in pois:
            idx = pois.index(cell)
            is_model = (cell == model_target)
            is_optimal = (cell == optimal_poi)
            if is_model and is_optimal:
                label, color = f"P{idx+1}", COLOR_AGREE_POI
            elif is_model:
                label, color = f"P{idx+1}", COLOR_MODEL_TARGET
            elif is_optimal:
                label, color = f"P{idx+1}", COLOR_OPTIMAL_POI
            else:
                label, color = f"P{idx+1}", COLOR_OTHER_POI
        elif cell == a1:
            label, color = "1", COLOR_AGENT1
        elif cell == a2:
            label, color = "2", COLOR_AGENT2
        else:
            return

        gs = viewer.grid_size + 1
        x = int(col * gs + gs / 2)
        y = int(panel_h - gs * (row + 1) + gs / 2)
        pyglet.text.Label(
            label,
            font_size=10,
            bold=(cell == model_target or cell == optimal_poi),
            x=x,
            y=y,
            anchor_x="center",
            anchor_y="center",
            color=color,
        ).draw()

    return draw_badge


def _draw_panel(
    viewer: Viewer,
    lbf: ForagingEnv,
    nav_env: PoINavigationEnv,
    ox: int,
    oy: int,
    panel_h: int,
) -> None:
    """Draw one grid panel at pixel offset (ox, oy), including obstacles."""
    real_height = viewer.height
    viewer.height = panel_h
    glPushMatrix()
    glTranslatef(ox, oy, 0)
    viewer._draw_grid()
    for (r, c) in nav_env._obstacles:
        viewer._draw_cell_fill(r, c, 0.2, 0.2, 0.2, 1.0)
    viewer._draw_food(lbf)
    viewer._draw_players(lbf)
    glPopMatrix()
    viewer.height = real_height


def _sync_lbf(lbf: ForagingEnv, nav_env: PoINavigationEnv) -> None:
    """Mirror PoINavigationEnv state into a ForagingEnv for rendering."""
    rows, cols = nav_env.grid_size
    lbf.field = np.zeros((rows, cols), dtype=np.int32)  # type: ignore[assignment]
    for poi in nav_env._pois:
        lbf.field[poi[0], poi[1]] = 2 if poi == nav_env._target_poi else 1
    lbf.players[0].position = tuple(nav_env._agent1_pos)
    lbf.players[1].position = tuple(nav_env._agent2_pos)
    lbf.players[0].level = 1
    lbf.players[1].level = 2
    lbf._gen_valid_moves()


# ---------------------------------------------------------------------------
# Main demo loop
# ---------------------------------------------------------------------------

N_PANELS = 3  # Q-Learning, DQN, PPO


def run_demo(n_scenarios: int, grid_size: tuple[int, int] = (64, 64)) -> None:
    rows, cols = grid_size
    size_tag = str(rows)
    env_id = _ENV_ID_MAP[rows]

    cell_px = _CELL_PX_MAP.get(rows, 4)
    panel_w = 1 + cols * (cell_px + 1)
    panel_h = 1 + rows * (cell_px + 1)
    win_w = panel_w * N_PANELS
    win_h = panel_h + TITLE_H

    # Single row: Q-Learning | DQN | PPO  (pyglet y=0 is bottom)
    panel_offsets = [(i * panel_w, 0) for i in range(N_PANELS)]

    # One PoINavigationEnv per panel (same scenario, independent step state)
    # Use a generous max_steps so agents have enough time to navigate
    max_steps = max(200, rows * cols)
    nav_envs = [PoINavigationEnv(seed=42, grid_size=grid_size, max_steps=max_steps) for _ in range(N_PANELS)]

    # Three lbforaging envs for rendering (share one Pyglet window)
    lbf_envs_raw: list[tuple[gym.Env, ForagingEnv]] = []
    for _ in range(N_PANELS):
        e = gym.make(env_id, render_mode="human", allow_agent_on_food=True, allow_agent_on_agent=True)
        e.reset(seed=42)
        lbf = e.unwrapped
        assert isinstance(lbf, ForagingEnv)
        lbf_envs_raw.append((e, lbf))

    # Init render on first, share viewer with others
    lbf_envs_raw[0][1]._init_render()
    viewer = lbf_envs_raw[0][1].viewer
    assert isinstance(viewer, Viewer)
    viewer.grid_size = cell_px
    viewer.width = win_w
    viewer.height = win_h
    viewer.window.set_size(win_w, win_h)
    for _, lbf in lbf_envs_raw[1:]:
        lbf.viewer = viewer

    lbfs = [lbf for _, lbf in lbf_envs_raw]

    policies: list[PredictFn] = [
        _load_qlearning(size_tag),
        _load_dqn(size_tag),
        _load_ppo(size_tag),
    ]

    # optimal_idxs[i] = cost-optimal POI index for panel i (updated each scenario reset)
    optimal_idxs: list[int] = [0] * N_PANELS

    def render_frame(badge_fns: list[Callable]) -> None:
        viewer.window.switch_to()
        viewer.window.dispatch_events()
        glClearColor(1, 1, 1, 1)
        glClear(GL_COLOR_BUFFER_BIT)
        for i, (lbf, badge_fn) in enumerate(zip(lbfs, badge_fns)):
            ox, oy = panel_offsets[i]
            lbf.viewer = viewer
            viewer._draw_badge = badge_fn
            _draw_panel(viewer, lbf, nav_envs[i], ox, oy, panel_h)
        for i, title in enumerate(PANEL_TITLES):
            ox, oy = panel_offsets[i]
            pyglet.text.Label(
                title,
                font_size=16,
                bold=True,
                x=ox + panel_w // 2,
                y=oy + panel_h + TITLE_H // 2,
                anchor_x="center",
                anchor_y="center",
                color=(30, 30, 30, 255),
            ).draw()
        viewer.window.flip()

    scenario = 0
    while viewer.isopen:
        if n_scenarios > 0 and scenario >= n_scenarios:
            break
        scenario += 1

        # Reset all envs with the same seed so they get identical scenarios
        scenario_seed = scenario * 1000
        obs_list: list[np.ndarray] = []
        for nav_env in nav_envs:
            obs, _ = nav_env.reset(seed=scenario_seed)
            obs_list.append(obs)

        # Compute cost-optimal POI index (same scenario → same result for all panels)
        for i, nav_env in enumerate(nav_envs):
            optimal_idxs[i] = _get_optimal_idx(nav_env)

        # Sync lbf state and build badge functions
        for lbf, nav_env in zip(lbfs, nav_envs):
            _sync_lbf(lbf, nav_env)
        badge_fns = [
            _make_badge_fn(nav_env, viewer, panel_h, optimal_idxs[i])
            for i, nav_env in enumerate(nav_envs)
        ]

        print(f"\n--- Scenario {scenario} ({size_tag}x{size_tag}) ---")
        print(f"Agent1: {nav_envs[0]._agent1_pos}  Agent2: {nav_envs[0]._agent2_pos}")
        print(f"Optimal POI: {nav_envs[0]._pois[optimal_idxs[0]]}  All POIs: {nav_envs[0]._pois}")
        print(f"Obstacles: {len(nav_envs[0]._obstacles)} cells")

        render_frame(badge_fns)
        time.sleep(1)

        # Lock each panel's target to the first action the model picks, then only move
        locked_targets: list[int | None] = [None] * N_PANELS

        # Step all envs until all are done
        done_flags = [False] * N_PANELS
        while not all(done_flags):
            if not viewer.isopen:
                for e, _ in lbf_envs_raw:
                    e.close()
                return

            for i, (nav_env, policy) in enumerate(zip(nav_envs, policies)):
                if done_flags[i]:
                    continue
                obs = obs_list[i]
                a_raw = policy(obs)
                # Lock target on first step; afterwards keep it fixed
                if locked_targets[i] is None:
                    locked_targets[i] = a_raw // 25
                target = locked_targets[i]
                # Keep model's move choices but enforce the locked target
                a = target * 25 + (a_raw % 25)
                obs, _, terminated, truncated, info = nav_env.step(a)
                obs_list[i] = obs
                done_flags[i] = terminated or truncated
                _sync_lbf(lbfs[i], nav_env)

            render_frame(badge_fns)
            time.sleep(0.15)

        results = [
            f"{PANEL_TITLES[i]}: {'arrived' if nav_envs[i]._agent1_pos == nav_envs[i]._target_poi and nav_envs[i]._agent2_pos == nav_envs[i]._target_poi else 'timeout'} ({nav_envs[i]._step_count} steps)"
            for i in range(N_PANELS)
        ]
        print("  " + "  |  ".join(results))

        # Hold 5 seconds before next scenario
        end_time = time.time() + 5.0
        while time.time() < end_time:
            if not viewer.isopen:
                for e, _ in lbf_envs_raw:
                    e.close()
                return
            render_frame(badge_fns)
            time.sleep(0.1)

    for e, _ in lbf_envs_raw:
        e.close()


def main() -> None:
    grid_size = (args.grid_size, args.grid_size)
    size_tag = str(args.grid_size)
    print("=" * 50)
    print(f"ThessLink: 3-panel navigation ({size_tag}x{size_tag}) — Q-Learning | DQN | PPO")
    print("=" * 50)
    n = args.scenarios
    print(f"Running {'infinite' if n == 0 else n} scenarios. Close window to exit.")
    run_demo(n, grid_size=grid_size)


if __name__ == "__main__":
    main()
