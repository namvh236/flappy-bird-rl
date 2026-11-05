"""Configurable reward strategies for `FlappyBirdEnv`."""

from __future__ import annotations

from typing import Literal

from ..game.config import GameConfig
from ..game.core import GameState, StepResult
from .observation import next_pipes

RewardMode = Literal["default", "sparse", "shaped"]
REWARD_MODES: tuple[RewardMode, ...] = ("default", "sparse", "shaped")

ALIVE_REWARD = 0.1
PIPE_REWARD = 1.0
DEATH_REWARD = -1.0
SHAPING_WEIGHT = 0.01


def compute_reward(
    mode: RewardMode, state: GameState, result: StepResult, config: GameConfig
) -> float:
    if result.dead:
        return DEATH_REWARD

    if mode == "sparse":
        return PIPE_REWARD if result.passed_pipe else 0.0

    reward = ALIVE_REWARD + (PIPE_REWARD if result.passed_pipe else 0.0)

    if mode == "shaped":
        pipe1, _ = next_pipes(state, config)
        gap_center = (pipe1.gap_top + pipe1.gap_bottom) / 2
        reward -= SHAPING_WEIGHT * abs(state.bird_rect.centery - gap_center) / config.screen_height

    return reward
