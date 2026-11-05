"""Tests for `PixelObs` wrapper (configs/dqn_pixels.yaml)."""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pytest

from flappy_ai.env.flappy_env import FlappyBirdEnv
from flappy_ai.env.wrappers import PixelObs


def test_requires_rgb_array_render_mode():
    with pytest.raises(ValueError):
        PixelObs(FlappyBirdEnv())


def test_reset_stacks_same_frame_four_times():
    env = PixelObs(FlappyBirdEnv(render_mode="rgb_array"))
    obs, _info = env.reset(seed=0)

    assert obs.shape == (4, 84, 84)
    assert obs.dtype == np.uint8
    for i in range(1, 4):
        assert np.array_equal(obs[0], obs[i])


def test_step_pushes_newest_frame_to_end_and_changes_it():
    env = PixelObs(FlappyBirdEnv(render_mode="rgb_array"))
    obs, _info = env.reset(seed=0)
    before = obs.copy()

    obs, _reward, _terminated, _truncated, _info = env.step(1)

    assert obs.shape == (4, 84, 84)
    assert np.array_equal(obs[:3], before[1:])
    assert not np.array_equal(obs[3], before[3])
