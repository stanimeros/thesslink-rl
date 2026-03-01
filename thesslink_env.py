"""
ThessLink Environment: 1 Human, 1 Agent, 3 POIs

User picks one POI (1, 2, 3) to meet the agent at.
"""
import gymnasium as gym
import numpy as np

try:
    import pyglet
    from pyglet.gl import *
except ImportError:
    pyglet = None

_BLACK = (0, 0, 0)
_WHITE = (255, 255, 255)


class Action:
    NONE, NORTH, SOUTH, WEST, EAST = 0, 1, 2, 3, 4


class ThessLinkEnv(gym.Env):
    """1 human, 1 agent, 3 static POIs. User selects meeting POI."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        grid_size: tuple[int, int] = (8, 8),
        pois: list[tuple[int, int]] | None = None,
        max_episode_steps: int = 100,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.rows, self.cols = grid_size
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode

        # 3 static POIs (meeting point suggestions)
        if pois is None:
            pois = [(2, 2), (4, 5), (6, 3)]
        self.pois = pois[:3]
        self.selected_poi: int | None = None  # 0, 1, or 2 - set by user

        # Human and Agent positions
        self.human_pos: tuple[int, int] = (0, 0)
        self.agent_pos: tuple[int, int] = (0, 0)
        self._step_count = 0

        self.action_space = gym.spaces.Discrete(5)  # NONE, N, S, E, W
        self.observation_space = gym.spaces.Box(
            low=0,
            high=max(self.rows, self.cols),
            shape=(10,),  # human_xy, agent_xy, poi0_xy, poi1_xy, poi2_xy
            dtype=np.float32,
        )
        self.viewer = None

    def _is_valid_cell(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _can_move_to(self, r: int, c: int) -> bool:
        if not self._is_valid_cell(r, c):
            return False
        if (r, c) == self.human_pos or (r, c) == self.agent_pos:
            return False
        if (r, c) in self.pois:
            return False
        return True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self._np_random = np.random.default_rng(seed)
        self._step_count = 0
        self.selected_poi = None

        # Spawn human and agent in empty cells
        available = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if (r, c) not in self.pois
        ]
        self._np_random.shuffle(available)
        self.human_pos = tuple(available[0])
        self.agent_pos = tuple(available[1])
        return self._obs(), {}

    def step(self, action: int):
        self._step_count += 1
        r, c = self.agent_pos
        if action == Action.NORTH and self._can_move_to(r - 1, c):
            self.agent_pos = (r - 1, c)
        elif action == Action.SOUTH and self._can_move_to(r + 1, c):
            self.agent_pos = (r + 1, c)
        elif action == Action.WEST and self._can_move_to(r, c - 1):
            self.agent_pos = (r, c - 1)
        elif action == Action.EAST and self._can_move_to(r, c + 1):
            self.agent_pos = (r, c + 1)

        reward = 0.0
        done = False
        truncated = self._step_count >= self.max_episode_steps
        if self.selected_poi is not None:
            target = self.pois[self.selected_poi]
            if self.agent_pos == target:
                reward = 1.0
                done = True

        return self._obs(), reward, done, truncated, {}

    def _obs(self):
        flat = [
            *self.human_pos,
            *self.agent_pos,
            *self.pois[0],
            *self.pois[1],
            *self.pois[2],
        ]
        return np.array(flat, dtype=np.float32)

    def set_selected_poi(self, idx: int):
        """User selects POI 1, 2, or 3 (0-indexed: 0, 1, 2)."""
        if 0 <= idx <= 2:
            self.selected_poi = idx

    def _init_render(self):
        if pyglet is None:
            raise ImportError("pyglet required for rendering")
        self.viewer = ThessLinkViewer((self.rows, self.cols))

    def render(self):
        if self.viewer is None:
            self._init_render()
        return self.viewer.render(self)

    def close(self):
        if self.viewer:
            self.viewer.close()


class ThessLinkViewer:
    """Custom viewer: H, A, P1, P2, P3 as text labels in grid cells."""

    def __init__(self, world_size: tuple[int, int]):
        self.rows, self.cols = world_size
        self.grid_size = 50
        self.width = 1 + self.cols * (self.grid_size + 1)
        self.height = 1 + self.rows * (self.grid_size + 1)
        self.window = pyglet.window.Window(width=self.width, height=self.height)
        self.window.on_close = self._on_close
        self.isopen = True

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def _on_close(self):
        self.isopen = False

    def close(self):
        if self.window:
            self.window.close()

    def render(self, env: ThessLinkEnv):
        glClearColor(*_WHITE, 0)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        self._draw_grid()
        self._draw_labels(env)

        self.window.flip()
        return self.isopen

    def _draw_grid(self):
        for r in range(self.rows + 1):
            pyglet.graphics.draw(
                2,
                GL_LINES,
                ("v2f", (0, (self.grid_size + 1) * r + 1, (self.grid_size + 1) * self.cols, (self.grid_size + 1) * r + 1)),
                ("c3B", (*_BLACK, *_BLACK)),
            )
        for c in range(self.cols + 1):
            pyglet.graphics.draw(
                2,
                GL_LINES,
                ("v2f", ((self.grid_size + 1) * c + 1, 0, (self.grid_size + 1) * c + 1, (self.grid_size + 1) * self.rows)),
                ("c3B", (*_BLACK, *_BLACK)),
            )

    def _draw_labels(self, env: ThessLinkEnv):
        self._draw_label(*env.human_pos, "H")
        self._draw_label(*env.agent_pos, "A")
        for i, (row, col) in enumerate(env.pois):
            label = f"P{i + 1}"
            if env.selected_poi == i:
                label += "*"
            self._draw_label(row, col, label)

    def _draw_label(self, row: int, col: int, text: str):
        x = col * (self.grid_size + 1) + (self.grid_size + 1) * 0.5
        y = self.height - (self.grid_size + 1) * (row + 1) + (self.grid_size + 1) * 0.5
        pyglet.text.Label(
            text,
            font_size=14,
            bold=True,
            x=x,
            y=y + 2,
            anchor_x="center",
            anchor_y="center",
            color=(*_BLACK, 255),
        ).draw()
