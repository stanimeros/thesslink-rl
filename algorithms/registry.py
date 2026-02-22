"""
Algorithm registry: name -> (factory_fn, needs_continuous_action).
Factories take (env, seed, **common_kwargs) and return a model.
"""

from __future__ import annotations

from typing import Any, Callable

import gymnasium as gym


def _make_dqn(env: gym.Env, seed: int | None, **kwargs: Any):
    from stable_baselines3 import DQN
    return DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        target_update_interval=1000,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        verbose=1,
        seed=seed,
        **kwargs,
    )


def _make_ppo(env: gym.Env, seed: int | None, **kwargs: Any):
    from stable_baselines3 import PPO
    return PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        seed=seed,
        **kwargs,
    )


def _make_a2c(env: gym.Env, seed: int | None, **kwargs: Any):
    from stable_baselines3 import A2C
    return A2C(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        gamma=0.99,
        verbose=1,
        seed=seed,
        **kwargs,
    )


def _make_trpo(env: gym.Env, seed: int | None, **kwargs: Any):
    from sb3_contrib import TRPO
    return TRPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        verbose=1,
        seed=seed,
        **kwargs,
    )


def _make_sac(env: gym.Env, seed: int | None, **kwargs: Any):
    from stable_baselines3 import SAC
    return SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=100_000,
        batch_size=64,
        gamma=0.99,
        verbose=1,
        seed=seed,
        **kwargs,
    )


def _make_td3(env: gym.Env, seed: int | None, **kwargs: Any):
    from stable_baselines3 import TD3
    return TD3(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=100_000,
        batch_size=64,
        gamma=0.99,
        policy_delay=2,
        verbose=1,
        seed=seed,
        **kwargs,
    )


def _make_ddpg(env: gym.Env, seed: int | None, **kwargs: Any):
    from stable_baselines3 import DDPG
    return DDPG(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=100_000,
        batch_size=64,
        gamma=0.99,
        verbose=1,
        seed=seed,
        **kwargs,
    )


# Registry: name -> (factory, needs_continuous_action_space)
ALGORITHMS: dict[str, tuple[Callable[..., Any], bool]] = {
    "DQN": (_make_dqn, False),
    "PPO": (_make_ppo, False),
    "A2C": (_make_a2c, False),
    "TRPO": (_make_trpo, False),
    "SAC": (_make_sac, True),
    "TD3": (_make_td3, True),
    "DDPG": (_make_ddpg, True),
}


def get_continuous_algos() -> list[str]:
    """Return algorithm names that require a continuous action space."""
    return [name for name, (_, cont) in ALGORITHMS.items() if cont]


def create_model(
    name: str,
    env: gym.Env,
    seed: int | None = None,
    **kwargs: Any,
):
    """Create a model for the given algorithm name and environment."""
    if name not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {name}. Choose from {list(ALGORITHMS.keys())}")
    factory, _ = ALGORITHMS[name]
    return factory(env, seed, **kwargs)
