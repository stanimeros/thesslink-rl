#!/usr/bin/env python3
"""
ThessLink Navigation Demo — 4-panel cooperative navigation with obstacles.

Four panels run the same scenario (same positions + obstacles) simultaneously,
each driven by a different algorithm: Q-Learning, DQN, PPO, Baseline.

Usage:
  python demo.py                    # 5 scenarios
  python demo.py --scenarios 10
  python demo.py --scenarios 0      # Infinite until window closed
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

from cost_function import nearest_human_baseline
from lbforaging.foraging.environment import ForagingEnv  # pyright: ignore[reportMissingImports]
from lbforaging.foraging.rendering import Viewer  # pyright: ignore[reportMissingImports]
from navigation_train import _NAV_ACTIONS, _discretize_nav
from poi_environment import PoINavigationEnv

parser = argparse.ArgumentParser()
parser.add_argument("--scenarios", type=int, default=5, help="Scenarios to run (0=infinite until window closed)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENV_ID = "Foraging-64x64-2p-3f-v3"
MODEL_DIR = Path(__file__).parent / "models"
CELL_PX = 4
TITLE_H = 28

COLOR_AGENT1 = (0, 0, 180, 255)      # blue: agent1
COLOR_AGENT2 = (180, 0, 0, 255)      # red: agent2
COLOR_TARGET = (0, 160, 0, 255)      # green: target POI
COLOR_OTHER_POI = (110, 110, 110, 255)  # grey: non-target POIs
COLOR_OBSTACLE = (50, 50, 50, 255)   # dark: obstacles

PANEL_TITLES = ["Q-Learning", "DQN", "PPO", "Baseline"]

# ---------------------------------------------------------------------------
# Load navigation policies
# ---------------------------------------------------------------------------

PredictFn = Callable[[np.ndarray], int]


def _load_ppo() -> PredictFn:
    from stable_baselines3 import PPO
    model = PPO.load(str(MODEL_DIR / "ppo" / "nav_ppo"))
    def predict(obs: np.ndarray) -> int:
        a, _ = model.predict(obs, deterministic=True)
        return int(a)
    return predict


def _load_dqn() -> PredictFn:
    from stable_baselines3 import DQN
    model = DQN.load(str(MODEL_DIR / "dqn" / "nav_dqn"))
    def predict(obs: np.ndarray) -> int:
        a, _ = model.predict(obs, deterministic=True)
        return int(a)
    return predict


def _load_qlearning() -> PredictFn:
    model_path = MODEL_DIR / "qlearning" / "nav_qtable.pkl"
    with open(model_path, "rb") as f:
        q_table: dict = pickle.load(f)
    def predict(obs: np.ndarray) -> int:
        return int(np.argmax(q_table.get(_discretize_nav(obs), np.zeros(_NAV_ACTIONS))))
    return predict


def _baseline_policy(nav_env: PoINavigationEnv) -> PredictFn:
    """
    Greedy baseline: always move toward nearest-human POI using A*.
    Re-evaluates target each step based on agent1's position.
    """
    from cost_function import astar_distance

    def predict(obs: np.ndarray) -> int:
        # Determine which agent this obs belongs to by checking self_pos
        rows, cols = nav_env.grid_size
        self_r = round(float(obs[0]) * (rows - 1))
        self_c = round(float(obs[1]) * (cols - 1))
        self_pos = (self_r, self_c)
        target = nav_env._target_poi
        obstacles = nav_env._obstacles

        if self_pos == target:
            return 0  # NONE

        # A*-guided greedy: pick action that minimises distance to target
        best_action = 0
        best_dist = astar_distance(self_pos, target, obstacles, nav_env.grid_size)
        for action, (dr, dc) in enumerate([(0,0),(-1,0),(1,0),(0,-1),(0,1)]):
            nr, nc = self_r + dr, self_c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if (nr, nc) in obstacles:
                continue
            d = astar_distance((nr, nc), target, obstacles, nav_env.grid_size)
            if d < best_dist:
                best_dist = d
                best_action = action
        return best_action

    return predict


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _make_badge_fn(
    nav_env: PoINavigationEnv,
    viewer: Viewer,
    panel_h: int,
) -> Callable:
    """Return a _draw_badge function for one panel, using panel-local coordinates."""
    rows, cols = nav_env.grid_size

    def draw_badge(row: int, col: int, level: int) -> None:
        cell = (row, col)
        obstacles = nav_env._obstacles
        target = nav_env._target_poi
        pois = nav_env._pois
        a1 = nav_env._agent1_pos
        a2 = nav_env._agent2_pos

        if cell in obstacles:
            label, color = "#", COLOR_OBSTACLE
        elif cell == target:
            label, color = "T", COLOR_TARGET
        elif cell in pois:
            label, color = f"P{pois.index(cell)+1}", COLOR_OTHER_POI
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
            bold=(cell == target),
            x=x,
            y=y,
            anchor_x="center",
            anchor_y="center",
            color=color,
        ).draw()

    return draw_badge


def _draw_panel(viewer: Viewer, lbf: ForagingEnv, ox: int, oy: int, panel_h: int) -> None:
    """Draw one grid panel at pixel offset (ox, oy)."""
    real_height = viewer.height
    viewer.height = panel_h
    glPushMatrix()
    glTranslatef(ox, oy, 0)
    viewer._draw_grid()
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

def run_demo(n_scenarios: int) -> None:
    rows, cols = 64, 64
    panel_w = 1 + cols * (CELL_PX + 1)
    panel_h = 1 + rows * (CELL_PX + 1)
    win_w = panel_w * 2
    win_h = (panel_h + TITLE_H) * 2

    # Panel offsets (ox, oy) — pyglet y=0 is bottom
    # Layout: top-left=Q-Learning, top-right=DQN, bottom-left=PPO, bottom-right=Baseline
    panel_offsets = [
        (0,       panel_h + TITLE_H),   # top-left:     Q-Learning
        (panel_w, panel_h + TITLE_H),   # top-right:    DQN
        (0,       0),                   # bottom-left:  PPO
        (panel_w, 0),                   # bottom-right: Baseline
    ]

    # One PoINavigationEnv per panel (same scenario, independent step state)
    nav_envs = [PoINavigationEnv(seed=42) for _ in range(4)]

    # Four lbforaging envs for rendering (share one Pyglet window)
    lbf_envs_raw: list[tuple[gym.Env, ForagingEnv]] = []
    for _ in range(4):
        e = gym.make(ENV_ID, render_mode="human", allow_agent_on_food=True, allow_agent_on_agent=True)
        e.reset(seed=42)
        lbf = e.unwrapped
        assert isinstance(lbf, ForagingEnv)
        lbf_envs_raw.append((e, lbf))

    # Init render on first, share viewer with others
    lbf_envs_raw[0][1]._init_render()
    viewer = lbf_envs_raw[0][1].viewer
    assert isinstance(viewer, Viewer)
    viewer.grid_size = CELL_PX
    viewer.width = win_w
    viewer.height = win_h
    viewer.window.set_size(win_w, win_h)
    for _, lbf in lbf_envs_raw[1:]:
        lbf.viewer = viewer

    lbfs = [lbf for _, lbf in lbf_envs_raw]

    # Load policies (Baseline uses nav_envs[3] for obstacle/target lookup)
    policies: list[PredictFn] = [
        _load_qlearning(),
        _load_dqn(),
        _load_ppo(),
        _baseline_policy(nav_envs[3]),
    ]

    def render_frame(badge_fns: list[Callable]) -> None:
        viewer.window.switch_to()
        viewer.window.dispatch_events()
        glClearColor(1, 1, 1, 1)
        glClear(GL_COLOR_BUFFER_BIT)
        for i, (lbf, badge_fn) in enumerate(zip(lbfs, badge_fns)):
            ox, oy = panel_offsets[i]
            lbf.viewer = viewer
            viewer._draw_badge = badge_fn
            _draw_panel(viewer, lbf, ox, oy, panel_h)
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
        obs_list: list[tuple[np.ndarray, np.ndarray]] = []
        for nav_env in nav_envs:
            obs_pair, _ = nav_env.reset(seed=scenario_seed)
            obs_list.append(obs_pair)

        # Sync lbf state and build badge functions
        for lbf, nav_env in zip(lbfs, nav_envs):
            _sync_lbf(lbf, nav_env)
        badge_fns = [_make_badge_fn(nav_env, viewer, panel_h) for nav_env in nav_envs]

        print(f"\n--- Scenario {scenario} ---")
        print(f"Agent1: {nav_envs[0]._agent1_pos}  Agent2: {nav_envs[0]._agent2_pos}")
        print(f"Target POI: {nav_envs[0]._target_poi}  All POIs: {nav_envs[0]._pois}")
        print(f"Obstacles: {len(nav_envs[0]._obstacles)} cells")

        render_frame(badge_fns)
        time.sleep(1)

        # Step all envs until all are done
        done_flags = [False] * 4
        while not all(done_flags):
            if not viewer.isopen:
                for e, _ in lbf_envs_raw:
                    e.close()
                return

            for i, (nav_env, policy) in enumerate(zip(nav_envs, policies)):
                if done_flags[i]:
                    continue
                obs1, obs2 = obs_list[i]
                a1 = policy(obs1)
                a2 = policy(obs2)
                obs_pair, _, terminated, truncated, info = nav_env.step((a1, a2))
                obs_list[i] = obs_pair
                done_flags[i] = terminated or truncated
                _sync_lbf(lbfs[i], nav_env)

            render_frame(badge_fns)
            time.sleep(0.15)

        results = [
            f"{PANEL_TITLES[i]}: {'arrived' if nav_envs[i]._agent1_pos == nav_envs[i]._target_poi and nav_envs[i]._agent2_pos == nav_envs[i]._target_poi else 'timeout'} ({nav_envs[i]._step_count} steps)"
            for i in range(4)
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
    print("=" * 50)
    print("ThessLink: 4-panel navigation (Q-Learning | DQN | PPO | Baseline)")
    print("=" * 50)
    n = args.scenarios
    print(f"Running {'infinite' if n == 0 else n} scenarios. Close window to exit.")
    run_demo(n)


if __name__ == "__main__":
    main()
