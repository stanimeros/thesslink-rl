"""
Environment wrappers for ThessLink-RL.
Provides a continuous action space variant for SAC, TD3, DDPG (Box action space).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from geopy.distance import geodesic


class ThessLinkContinuousWrapper(gym.Wrapper):
    """
    Wraps ThessLinkEnv to use a continuous action space (Box) instead of Discrete.
    Action is [lat_norm, lon_norm] in [0, 1]².
    The nearest POI to the chosen point is used for reward computation.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(2,), dtype=np.float32
        )
        # Expose bbox for denormalization
        self._lat_min = env.unwrapped._lat_min
        self._lat_max = env.unwrapped._lat_max
        self._lon_min = env.unwrapped._lon_min
        self._lon_max = env.unwrapped._lon_max
        self._pois = env.unwrapped.pois

    def _action_to_poi_index(self, action: np.ndarray) -> int:
        lat = self._lat_min + action[0] * (self._lat_max - self._lat_min)
        lon = self._lon_min + action[1] * (self._lon_max - self._lon_min)
        point = (lat, lon)
        distances = self._pois.apply(
            lambda r: geodesic(point, (r["lat"], r["lon"])).km, axis=1
        )
        return int(distances.argmin())

    def step(self, action):
        discrete_action = self._action_to_poi_index(
            np.clip(action, 0.0, 1.0).astype(np.float32)
        )
        return self.env.step(discrete_action)
