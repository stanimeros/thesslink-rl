"""
Gymnasium environment for POI suggestion. Used for Reinforcement Learning training.

State: cost components for each POI (travel_effort_agent, travel_effort_human, energy, privacy, time_to_meet) = 5*3 = 15 floats
Action: Discrete(3) - which POI to suggest (0, 1, 2)
Reward: -cost (Travel Effort, energy expenditure, privacy, Time-to-Meet)
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from cost_function import cost_components, cost_function, DEFAULT_WEIGHTS


def build_observation(
    human_pos: tuple[int, int],
    agent_pos: tuple[int, int],
    pois: list[tuple[int, int]],
    grid_size: tuple[int, int] = (64, 64),
) -> np.ndarray:
    """Build observation vector for RL policy. Same format as PoISuggestionEnv._get_obs."""
    parts = []
    for poi in pois:
        comps = cost_components(poi, agent_pos, human_pos, grid_size)
        parts.append(np.array(comps, dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)


class PoISuggestionEnv(gym.Env):
    """
    RL environment for learning which POI to suggest.

    Each episode: sample random (human_pos, agent_pos, 3 POIs).
    Agent chooses one POI. Reward = -cost (Travel Effort, energy expenditure, privacy, Time-to-Meet).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        grid_size: tuple[int, int] = (64, 64),
        weights: tuple[float, ...] | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.rows, self.cols = grid_size
        self.weights = DEFAULT_WEIGHTS if weights is None else weights

        # State: cost_components for each POI (travel_effort_agent, travel_effort_human, energy, privacy, time_to_meet) * 3 POIs = 15 floats
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(15,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(3)

        self._rng = np.random.default_rng(seed)
        self._human_pos: tuple[int, int] | None = None
        self._agent_pos: tuple[int, int] | None = None
        self._pois: list[tuple[int, int]] | None = None

    def _sample_scenario(self) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]]:
        """Sample distinct positions for human, agent, and 3 POIs."""
        cells = [(r, c) for r in range(self.rows) for c in range(self.cols)]
        chosen = self._rng.choice(len(cells), size=5, replace=False)
        positions = [cells[j] for j in chosen]
        return positions[0], positions[1], positions[2:5]

    def _get_obs(self) -> np.ndarray:
        """Build observation vector from current state."""
        assert self._human_pos is not None and self._agent_pos is not None and self._pois is not None
        parts = []
        for poi in self._pois:
            comps = cost_components(poi, self._agent_pos, self._human_pos, self.grid_size)
            parts.append(np.array(comps, dtype=np.float32))
        return np.concatenate(parts).astype(np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._human_pos, self._agent_pos, self._pois = self._sample_scenario()
        return self._get_obs(), {}

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._pois is None:
            raise RuntimeError("Call reset() before step()")
        action = int(np.clip(action, 0, 2))
        chosen_poi = self._pois[action]

        cost = cost_function(
            chosen_poi,
            self._agent_pos,
            self._human_pos,
            self.grid_size,
            *self.weights,
        )
        reward = -cost

        terminated = True
        truncated = False
        info = {
            "chosen_poi": chosen_poi,
            "reward": reward,
            "human_pos": self._human_pos,
            "agent_pos": self._agent_pos,
            "pois": self._pois.copy(),
        }
        return self._get_obs(), reward, terminated, truncated, info
