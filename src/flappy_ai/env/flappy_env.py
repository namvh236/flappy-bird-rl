"""`FlappyBirdEnv(gymnasium.Env)`: bọc `FlappyGame` theo interface Gymnasium
chuẩn (spec §5.4). Headless mặc định (`render_mode=None`) — không đụng
`pygame.display` cho tới khi thật sự cần vẽ (`human`/`rgb_array`).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..game.config import GameConfig
from ..game.core import FlappyGame
from .observation import OBS_SIZE, build_features
from .reward import REWARD_MODES, RewardMode, compute_reward

DEFAULT_MAX_EPISODE_STEPS = 20_000


class FlappyBirdEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        config: GameConfig = GameConfig(),
        render_mode: str | None = None,
        reward_mode: RewardMode = "default",
        max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
        debug_overlay: bool = False,
    ):
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode không hợp lệ: {render_mode!r}")
        if reward_mode not in REWARD_MODES:
            raise ValueError(f"reward_mode không hợp lệ: {reward_mode!r}")

        self.config = config
        self.render_mode = render_mode
        self.reward_mode = reward_mode
        self.max_episode_steps = max_episode_steps
        self.debug_overlay = debug_overlay
        self.hud: dict[str, Any] = {}

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32)

        self.game = FlappyGame(config)
        self._renderer = None
        self._steps = 0

    def _get_obs(self) -> np.ndarray:
        return build_features(self.game.state, self.config)

    def _get_info(self) -> dict[str, Any]:
        state = self.game.state
        return {
            "score": state.score,
            "frame": state.frame,
            "death_cause": self.game.death_cause,
            "pipes_passed": state.score,
        }

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.game.reset(seed=seed)
        self._steps = 0
        obs = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        result = self.game.step(flap=bool(action == 1))
        self._steps += 1

        reward = compute_reward(self.reward_mode, self.game.state, result, self.config)
        terminated = result.dead
        truncated = not terminated and self._steps >= self.max_episode_steps

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        return None

    @property
    def renderer(self):
        """Renderer của env (tạo lazily ở lần vẽ đầu) — play/record cần nó để
        chỉnh tốc độ/tạm dừng."""
        if self._renderer is None:
            self._render_frame()
        return self._renderer

    def _render_frame(self):
        if self._renderer is None:
            from ..render.pygame_renderer import PygameRenderer

            overlay = None
            if self.debug_overlay:
                from ..render.overlay import DebugOverlay

                overlay = DebugOverlay(self.config)
            self._renderer = PygameRenderer(
                self.config, render_mode=self.render_mode, overlay=overlay
            )
        self._renderer.draw(self.game.state, extra=self.hud if self.debug_overlay else None)
        if self.render_mode == "rgb_array":
            return self._renderer.to_rgb_array()
        return None

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
