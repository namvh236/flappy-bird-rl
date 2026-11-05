"""8-dim feature vector for `FlappyBirdEnv`; uses virtual pipes when fewer than
2 real ones are on screen, avoiding special-case branches in the env.
"""

from __future__ import annotations

import numpy as np

from ..game.config import GameConfig
from ..game.core import GameState, PipePairState

OBS_SIZE = 8


def _virtual_pipe(config: GameConfig, x: float) -> PipePairState:
    gap_center = config.screen_height / 2
    return PipePairState(
        x=x,
        gap_top=gap_center - config.pipe_gap / 2,
        gap_bottom=gap_center + config.pipe_gap / 2,
        passed=False,
    )


def next_pipes(state: GameState, config: GameConfig) -> tuple[PipePairState, PipePairState]:
    """Next two pipes ahead of the bird, falling back to virtual pipes if needed."""
    bird_left = state.bird_rect.left
    upcoming = [p for p in state.pipes if p.x + config.pipe_width >= bird_left]
    pipe1 = upcoming[0] if upcoming else _virtual_pipe(config, config.screen_width)
    pipe2_x = pipe1.x + config.screen_width / 2
    pipe2 = upcoming[1] if len(upcoming) > 1 else _virtual_pipe(config, pipe2_x)
    return pipe1, pipe2


def build_features(state: GameState, config: GameConfig) -> np.ndarray:
    bird = state.bird_rect
    pipe1, pipe2 = next_pipes(state, config)
    pipe1_right = pipe1.x + config.pipe_width
    pipe2_right = pipe2.x + config.pipe_width
    pipe2_gap_center = (pipe2.gap_top + pipe2.gap_bottom) / 2

    features = np.array(
        [
            state.bird_y / config.ground_y,
            state.bird_vel / 10,
            (pipe1_right - bird.left) / config.screen_width,
            (pipe1.gap_top - bird.top) / config.screen_height,
            (pipe1.gap_bottom - bird.bottom) / config.screen_height,
            (pipe2_right - bird.left) / config.screen_width,
            (pipe2_gap_center - bird.centery) / config.screen_height,
            (config.ground_y - bird.bottom) / config.ground_y,
        ],
        dtype=np.float32,
    )
    return np.clip(features, -1.0, 1.0)
