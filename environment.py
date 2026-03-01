"""
Gymnasium environment for POI suggestion. Used for Reinforcement Learning training.

State: (human_pos, agent_pos, poi1, poi2, poi3) normalized to [0,1], plus cost components
       for each POI (d_agent, d_human, energy, privacy) = 10 + 12 = 22 floats
Action: Discrete(3) - which POI to suggest (0, 1, 2)
Reward: configurable - "cost" (reward=-cost) or "steps" (reward=-steps to both arrive)
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from cost_function import cost_components, cost_function, load_weights


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _normalize_pos(pos: tuple[int, int], rows: int, cols: int) -> np.ndarray:
    """Normalize (row, col) to [0, 1] range."""
    return np.array([pos[0] / max(1, rows - 1), pos[1] / max(1, cols - 1)], dtype=np.float32)


def build_observation(
    human_pos: tuple[int, int],
    agent_pos: tuple[int, int],
    pois: list[tuple[int, int]],
    grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Build observation vector for RL policy. Same format as PoISuggestionEnv._get_obs."""
    rows, cols = grid_size
    parts = [
        _normalize_pos(human_pos, rows, cols),
        _normalize_pos(agent_pos, rows, cols),
        _normalize_pos(pois[0], rows, cols),
        _normalize_pos(pois[1], rows, cols),
        _normalize_pos(pois[2], rows, cols),
    ]
    for poi in pois:
        comps = cost_components(poi, agent_pos, human_pos, grid_size)
        parts.append(np.array(comps, dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)


class PoISuggestionEnv(gym.Env):
    """
    RL environment for learning which POI to suggest.

    Each episode: sample random (human_pos, agent_pos, 3 POIs).
    Agent chooses one POI. Reward depends on reward_type:
    - "cost": reward = -cost (uses cost formula with d_agent, d_human, energy, privacy)
    - "steps": reward = -max(d_agent, d_human) (steps for both to reach POI)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        grid_size: tuple[int, int] = (8, 8),
        weights: tuple[float, float, float, float] | None = None,
        reward_type: str = "cost",
        seed: int | None = None,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.rows, self.cols = grid_size
        self.weights = load_weights() if weights is None else weights
        self.reward_type = reward_type


        # State: human(2) + agent(2) + poi1(2) + poi2(2) + poi3(2) + cost_components for each POI (4*3=12) = 22 floats
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(22,),
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
        parts = [
            _normalize_pos(self._human_pos, self.rows, self.cols),
            _normalize_pos(self._agent_pos, self.rows, self.cols),
            _normalize_pos(self._pois[0], self.rows, self.cols),
            _normalize_pos(self._pois[1], self.rows, self.cols),
            _normalize_pos(self._pois[2], self.rows, self.cols),
        ]
        # Add cost components for each POI (d_agent, d_human, energy, privacy) - already in [0,1]
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

        if self.reward_type == "cost":
            cost = cost_function(
                chosen_poi,
                self._agent_pos,
                self._human_pos,
                self.grid_size,
                *self.weights,
            )
            reward = -cost
        else:
            # "steps": steps for both to reach POI (Manhattan)
            steps = max(
                _manhattan(self._agent_pos, chosen_poi),
                _manhattan(self._human_pos, chosen_poi),
            )
            max_dist = self.rows + self.cols
            reward = -steps / max_dist

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
