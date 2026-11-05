"""`Bird` and `PipePair`: pure state. `Bird.y` is the top edge (not center),
matching legacy physics exactly (see test_parity.py).
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import GameConfig
from .rect import Rect


@dataclass
class Bird:
    y: float  # top edge of rect, matches legacy's `self.rect.y += int(vel)`
    vel: float = 0.0

    @classmethod
    def spawn(cls, config: GameConfig) -> Bird:
        # legacy sets rect.center = bird_start; convert center to top edge here
        center_y = config.bird_start[1]
        top_y = center_y - config.bird_size[1] // 2
        return cls(y=float(top_y), vel=0.0)

    def rect(self, config: GameConfig) -> Rect:
        # rect.x never changes after creation, matching legacy
        x = config.bird_start[0] - config.bird_size[0] // 2
        return Rect(x=x, y=self.y, width=config.bird_size[0], height=config.bird_size[1])

    def step_physics(self, config: GameConfig, flap: bool) -> None:
        # Gravity+move runs before flap in legacy; flap only takes effect next frame
        self.vel = min(self.vel + config.gravity, config.max_fall_speed)
        # Legacy only applies movement if rect.bottom < ground_y before adding
        if self.rect(config).bottom < config.ground_y:
            dy = self.vel if config.float_physics else float(int(self.vel))
            self.y += dy
        if flap:
            self.vel = config.flap_velocity


@dataclass
class PipePair:
    x: float
    gap_center: float
    passed: bool = False

    def top_rect(self, config: GameConfig) -> Rect:
        bottom = self.gap_center - config.pipe_gap // 2
        top = bottom - config.pipe_height
        return Rect(x=self.x, y=top, width=config.pipe_width, height=config.pipe_height)

    def bottom_rect(self, config: GameConfig) -> Rect:
        top = self.gap_center + config.pipe_gap // 2
        return Rect(x=self.x, y=top, width=config.pipe_width, height=config.pipe_height)

    def scroll(self, config: GameConfig) -> None:
        self.x -= config.scroll_speed

    def offscreen(self, config: GameConfig) -> bool:
        return self.x + config.pipe_width < 0
