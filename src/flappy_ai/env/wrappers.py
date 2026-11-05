"""`FrameSkip`: repeats the action for `skip` core steps; flap only applies on
the first frame since repeating it has no extra physical effect.
"""

from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
import pygame


class FrameSkip(gym.Wrapper):
    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        if skip < 1:
            raise ValueError("skip must be >= 1")
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info: dict = {}
        obs = None
        for i in range(self.skip):
            frame_action = action if i == 0 else 0
            obs, reward, terminated, truncated, info = self.env.step(frame_action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class PixelObs(gym.ObservationWrapper):
    """Stacks 4 grayscale 84x84 frames, channel-first to match SB3's `CnnPolicy`
    (like Atari).
    """

    FRAME_SIZE = 84
    N_FRAMES = 4
    _WEIGHTS = (0.299, 0.587, 0.114)

    def __init__(self, env: gym.Env):
        if env.render_mode != "rgb_array":
            raise ValueError(
                f"PixelObs requires env.render_mode == 'rgb_array', got {env.render_mode!r}"
            )
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.N_FRAMES, self.FRAME_SIZE, self.FRAME_SIZE),
            dtype=np.uint8,
        )
        self._frames: deque[np.ndarray] = deque(maxlen=self.N_FRAMES)

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        r, g, b = self._WEIGHTS
        gray = (frame[..., 0] * r + frame[..., 1] * g + frame[..., 2] * b).astype(np.uint8)
        gray_rgb = np.stack([gray] * 3, axis=-1)
        surface = pygame.surfarray.make_surface(np.transpose(gray_rgb, (1, 0, 2)))
        scaled = pygame.transform.smoothscale(surface, (self.FRAME_SIZE, self.FRAME_SIZE))
        return np.transpose(pygame.surfarray.array3d(scaled), (1, 0, 2))[..., 0]

    def observation(self, observation):
        return np.stack(self._frames, axis=0)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        frame = self._preprocess(self.env.render())
        self._frames.clear()
        self._frames.extend([frame] * self.N_FRAMES)
        return self.observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(self._preprocess(self.env.render()))
        return self.observation(obs), reward, terminated, truncated, info
