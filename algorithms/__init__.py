"""
Modular RL algorithm registry for ThessLink-RL.
Each algorithm is defined with a factory and action-space type.
"""

from __future__ import annotations

from .registry import ALGORITHMS, create_model, get_continuous_algos

__all__ = ["ALGORITHMS", "create_model", "get_continuous_algos"]
