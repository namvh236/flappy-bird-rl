"""Tests for `FlappyBirdEnv` (checklist P2)."""

from __future__ import annotations

import numpy as np
from gymnasium.utils.env_checker import check_env

from flappy_ai.env.flappy_env import FlappyBirdEnv
from flappy_ai.env.wrappers import FrameSkip


def test_check_env_passes():
    env = FlappyBirdEnv()
    check_env(env.unwrapped, skip_render_check=True)


def test_obs_stays_in_bounds_over_1000_random_episodes():
    env = FlappyBirdEnv(max_episode_steps=200)
    rng = np.random.default_rng(0)

    for episode in range(1000):
        obs, _info = env.reset(seed=episode)
        assert env.observation_space.contains(obs)
        done = False
        while not done:
            action = int(rng.integers(0, 2))
            obs, _reward, terminated, truncated, _info = env.step(action)
            assert env.observation_space.contains(obs)
            done = terminated or truncated


def test_same_seed_is_deterministic():
    def run(seed: int, actions: list[int]) -> list[float]:
        env = FlappyBirdEnv()
        obs, _info = env.reset(seed=seed)
        trace = [float(obs[0])]
        for action in actions:
            obs, _reward, terminated, truncated, _info = env.step(action)
            trace.append(float(obs[0]))
            if terminated or truncated:
                break
        return trace

    actions = [1, 0, 0, 0, 1, 0, 0, 0, 0, 1] * 20
    assert run(42, actions) == run(42, actions)


def test_frame_skip_repeats_action_and_only_flaps_first_frame():
    base_env = FlappyBirdEnv()
    base_env.reset(seed=0)
    base_env.step(1)
    base_env.step(0)
    base_env.step(0)
    expected_vel = base_env.game.state.bird_vel

    skipped_env = FrameSkip(FlappyBirdEnv(), skip=3)
    skipped_env.reset(seed=0)
    skipped_env.step(1)

    assert skipped_env.env.game.state.bird_vel == expected_vel
