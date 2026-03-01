#!/usr/bin/env python3
"""
Interactive ThessLink: Pick a POI (1,2,3) and move the agent to meet there.

Controls:
  1, 2, 3 - Select POI 1, 2, or 3 as meeting point
  Arrow keys - Move the agent
  R - Reset
  H - Help
  ESC - Exit
"""
import time

from pyglet.window import key

from thesslink_env import Action, ThessLinkEnv


def main():
    env = ThessLinkEnv(
        grid_size=(8, 8),
        pois=[(2, 2), (4, 5), (6, 3)],
        max_episode_steps=100,
        render_mode="human",
    )
    obs, _ = env.reset(seed=42)
    env.render()

    class PlayState:
        last_action: int = Action.NONE
        reset: bool = False
        running: bool = True
    state = PlayState()

    def on_key(symbol: int, modifiers: int, s: PlayState) -> None:
        if symbol in (key.NUM_1, ord("1")):
            env.set_selected_poi(0)
            print("Selected POI 1")
        elif symbol in (key.NUM_2, ord("2")):
            env.set_selected_poi(1)
            print("Selected POI 2")
        elif symbol in (key.NUM_3, ord("3")):
            env.set_selected_poi(2)
            print("Selected POI 3")
        elif symbol == key.LEFT:
            s.last_action = Action.WEST
        elif symbol == key.UP:
            s.last_action = Action.NORTH
        elif symbol == key.RIGHT:
            s.last_action = Action.EAST
        elif symbol == key.DOWN:
            s.last_action = Action.SOUTH
        elif symbol == key.R:
            s.reset = True
        elif symbol == key.H:
            print("1,2,3: Select POI | Arrows: Move agent | R: Reset | ESC: Exit")
        elif symbol == key.ESCAPE:
            s.running = False

    if env.viewer and env.viewer.window:
        def _on_key(symbol: int, modifiers: int) -> None:
            on_key(symbol, modifiers, state)
        env.viewer.window.on_key_press = _on_key

    print("Press 1, 2, or 3 to pick a meeting POI. Use arrows to move the agent.")
    print("Press H for help.")

    while state.running:
        if state.reset:
            obs, _ = env.reset()
            state.reset = False
            state.last_action = Action.NONE

        action = state.last_action
        state.last_action = Action.NONE

        obs, reward, done, truncated, _ = env.step(action)
        env.render()

        if reward > 0:
            print("Agent reached the meeting point!")
        if done or truncated:
            state.reset = True

        time.sleep(0.1)

    env.close()


if __name__ == "__main__":
    main()
