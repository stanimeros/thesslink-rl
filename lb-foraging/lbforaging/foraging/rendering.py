"""
2D rendering of the level based foraging domain
"""

import math
import os
import sys

import numpy as np
import six
from gymnasium import error

if "Apple" in sys.version:
    if "DYLD_FALLBACK_LIBRARY_PATH" in os.environ:
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] += ":/usr/lib"
        # (JDS 2016/04/15): avoid bug on Anaconda 2.3.0 / Yosemite


try:
    import pyglet
except ImportError:
    raise ImportError(
        """
    Cannot import pyglet.
    HINT: you can install pyglet directly via 'pip install pyglet'.
    But if you really just want to install all Gym dependencies and not have to think about it,
    'pip install -e .[all]' or 'pip install gym[all]' will do it.
    """
    )

try:
    from pyglet.gl import *
except ImportError:
    raise ImportError(
        """
    Error occured while running `from pyglet.gl import *`
    HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'.
    If you're running on a server, you may need a virtual frame buffer; something like this should work:
    'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'
    """
    )


RAD2DEG = 57.29577951308232
# # Define some colors
_BLACK = (0, 0, 0)
_WHITE = (255, 255, 255)
_GREEN = (0, 255, 0)
_RED = (255, 0, 0)

# ThessLink: soft fill colors (R,G,B,A 0-1) instead of icons
_COLOR_POI = (0.95, 0.94, 0.88, 0.65)   # warm cream for POIs
_COLOR_HUMAN = (0.88, 0.92, 1.0, 0.75)   # soft blue for Human
_COLOR_AGENT = (0.9, 0.94, 0.92, 0.75)   # soft green-gray for Agent
_COLOR_BOTH = (0.88, 0.96, 0.92, 0.8)    # soft mint when H+A meet

_BACKGROUND_COLOR = _WHITE
_GRID_COLOR = _BLACK


def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display
    object.
    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return None
    elif isinstance(spec, six.string_types):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error(
            "Invalid display specification: {}. (Must be a string like :0 or None.)".format(
                spec
            )
        )


class Viewer(object):
    def __init__(self, world_size):
        display = get_display(None)
        self.rows, self.cols = world_size

        self.grid_size = 50
        self.icon_size = 20

        self.width = 1 + self.cols * (self.grid_size + 1)
        self.height = 1 + self.rows * (self.grid_size + 1)
        self.window = pyglet.window.Window(
            width=self.width, height=self.height, display=display
        )
        self.window.on_close = self.window_closed_by_user
        self.isopen = True

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        script_dir = os.path.dirname(__file__)

        pyglet.resource.path = [os.path.join(script_dir, "icons")]
        pyglet.resource.reindex()

        self.img_apple = pyglet.resource.image("apple.png")
        self.img_agent = pyglet.resource.image("agent.png")

    def close(self):
        self.window.close()

    def window_closed_by_user(self):
        self.isopen = False
        exit()

    def set_bounds(self, left, right, bottom, top):
        assert right > left and top > bottom
        scalex = self.width / (right - left)
        scaley = self.height / (top - bottom)
        self.transform = Transform(
            translation=(-left * scalex, -bottom * scaley), scale=(scalex, scaley)
        )

    def render(self, env, return_rgb_array=False):
        glClearColor(*_WHITE, 0)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        self._draw_grid()
        self._draw_food(env)
        self._draw_players(env)

        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1, :, 0:3]
        self.window.flip()
        return arr if return_rgb_array else self.isopen

    def _draw_grid(self):
        batch = pyglet.graphics.Batch()
        # vertical lines
        for r in range(self.rows + 1):
            batch.add(
                2,
                gl.GL_LINES,
                None,
                (
                    "v2f",
                    (
                        0,  # LEFT X
                        (self.grid_size + 1) * r + 1,  # Y
                        (self.grid_size + 1) * self.cols,  # RIGHT X
                        (self.grid_size + 1) * r + 1,  # Y
                    ),
                ),
                ("c3B", (*_BLACK, *_BLACK)),
            )

        # horizontal lines
        for c in range(self.cols + 1):
            batch.add(
                2,
                gl.GL_LINES,
                None,
                (
                    "v2f",
                    (
                        (self.grid_size + 1) * c + 1,  # X
                        0,  # BOTTOM Y
                        (self.grid_size + 1) * c + 1,  # X
                        (self.grid_size + 1) * self.rows,  # TOP X
                    ),
                ),
                ("c3B", (*_BLACK, *_BLACK)),
            )
        batch.draw()

    def _draw_cell_fill(self, row, col, r, g, b, a):
        """Draw a soft-filled rectangle for a cell (ThessLink: no icons)."""
        gs = self.grid_size + 1
        x1 = gs * col + 1
        y1 = self.height - gs * (row + 1)
        x2 = gs * (col + 1)
        y2 = self.height - gs * row - 1
        verts = [x1, y1, x2, y1, x2, y2, x1, y2]
        colors = [r, g, b, a] * 4
        pyglet.graphics.draw(4, gl.GL_QUADS, ("v2f", verts), ("c4f", colors))

    def _draw_food(self, env):
        idxes = list(zip(*env.field.nonzero()))
        r, g, b, a = _COLOR_POI
        for row, col in idxes:
            self._draw_cell_fill(row, col, r, g, b, a)
        for row, col in idxes:
            self._draw_badge(row, col, env.field[row, col])

    def _draw_players(self, env):
        pos0 = env.players[0].position
        pos1 = env.players[1].position if len(env.players) > 1 else None
        cell0 = (int(pos0[0]), int(pos0[1]))
        cell1 = (int(pos1[0]), int(pos1[1])) if pos1 is not None else None
        both_on_same = cell1 is not None and cell0 == cell1
        if both_on_same:
            self._draw_cell_fill(cell0[0], cell0[1], *_COLOR_BOTH)
        else:
            self._draw_cell_fill(cell0[0], cell0[1], *_COLOR_HUMAN)
            if cell1 is not None:
                self._draw_cell_fill(cell1[0], cell1[1], *_COLOR_AGENT)
        for p in env.players:
            self._draw_badge(*p.position, p.level)

    def _draw_badge(self, row, col, level):
        resolution = 6
        radius = self.grid_size / 5

        badge_x = col * (self.grid_size + 1) + (3 / 4) * (self.grid_size + 1)
        badge_y = (
            self.height
            - (self.grid_size + 1) * (row + 1)
            + (1 / 4) * (self.grid_size + 1)
        )

        # make a circle
        verts = []
        for i in range(resolution):
            angle = 2 * math.pi * i / resolution
            x = radius * math.cos(angle) + badge_x
            y = radius * math.sin(angle) + badge_y
            verts += [x, y]
        circle = pyglet.graphics.vertex_list(resolution, ("v2f", verts))
        glColor3ub(*_WHITE)
        circle.draw(GL_POLYGON)
        glColor3ub(*_BLACK)
        circle.draw(GL_LINE_LOOP)
        label = pyglet.text.Label(
            str(level),
            font_name="Times New Roman",
            font_size=12,
            bold=True,
            x=badge_x,
            y=badge_y + 2,
            anchor_x="center",
            anchor_y="center",
            color=(*_BLACK, 255),
        )
        label.draw()
