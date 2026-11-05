"""Đăng ký `FlappyBird-v0` với Gymnasium khi import `flappy_ai.env` (spec §5.4).

`max_episode_steps=None` ở `gym.register`: `FlappyBirdEnv` tự quản lý
`truncated` qua tham số `max_episode_steps` riêng, không cần `TimeLimit`
wrapper của Gymnasium chồng lên (sẽ tạo ra 2 nguồn truncation khác nhau).
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
