"""Registers `FlappyBird-v0` with Gymnasium; `max_episode_steps=None` avoids a
second `TimeLimit` truncation source since `FlappyBirdEnv` manages its own.
"""

from __future__ import annotations

import gymnasium as gym

from .flappy_env import FlappyBirdEnv
from .wrappers import FrameSkip

gym.register(
    id="FlappyBird-v0",
    entry_point="flappy_ai.env.flappy_env:FlappyBirdEnv",
    max_episode_steps=None,
)

__all__ = ["FlappyBirdEnv", "FrameSkip"]
