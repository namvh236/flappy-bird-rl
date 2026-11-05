"""`FrameSkip`: lặp lại action trong `skip` step lõi mỗi lần `step()` (spec
§5.4, sequence diagram §3.3) — flap chỉ áp dụng ở frame đầu tiên, các frame
bỏ qua nhận action `0` (không làm gì) vì giữ flap liên tục không có ý nghĩa
vật lý (mỗi flap chỉ đặt lại `vel`, gọi lại không đổi gì thêm trong cùng loạt).
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
            raise ValueError("skip phải >= 1")
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
    """Stack 4 frame grayscale 84×84 channel-first (spec `docs/plan/experiments.md`
    stretch goal CNN DQN) — tương thích `CnnPolicy` của SB3, vốn kỳ vọng
    `(C, H, W)` như Atari thay vì `(H, W, C)` của ảnh raw.
    """

    FRAME_SIZE = 84
    N_FRAMES = 4
    _WEIGHTS = (0.299, 0.587, 0.114)

    def __init__(self, env: gym.Env):
        if env.render_mode != "rgb_array":
            raise ValueError(
                f"PixelObs yêu cầu env.render_mode == 'rgb_array', nhận {env.render_mode!r}"
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
