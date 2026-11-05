"""`Bird` và `PipePair`: state thuần (không render, không input trực tiếp).

Bản gốc di chuyển bird qua `rect.y` (mép trên), không phải tâm — `Bird.y` ở
đây theo đúng ngữ nghĩa đó để giữ quỹ đạo vật lý y hệt (xem test_parity.py).
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import GameConfig
from .rect import Rect


@dataclass
class Bird:
    y: float  # mép trên (top) của rect, khớp bản gốc: `self.rect.y += int(vel)`
    vel: float = 0.0

    @classmethod
    def spawn(cls, config: GameConfig) -> Bird:
        # Bản gốc: Bird(x, y) đặt `rect.center = [x, y]` với bird_start là tâm.
        center_y = config.bird_start[1]
        top_y = center_y - config.bird_size[1] // 2
        return cls(y=float(top_y), vel=0.0)

    def rect(self, config: GameConfig) -> Rect:
        # rect.x cố định suốt game (bản gốc không bao giờ đổi rect.x sau khi tạo).
        x = config.bird_start[0] - config.bird_size[0] // 2
        return Rect(x=x, y=self.y, width=config.bird_size[0], height=config.bird_size[1])

    def step_physics(self, config: GameConfig, flap: bool) -> None:
        # Bản gốc: gravity + di chuyển chạy TRƯỚC, click-to-flap chạy SAU trong
        # cùng một frame ⇒ `vel = flap_velocity` chỉ có hiệu lực từ frame kế
        # tiếp. Đảo thứ tự này sẽ làm sai quỹ đạo so với bản gốc (test_parity).
        self.vel = min(self.vel + config.gravity, config.max_fall_speed)
        # Bản gốc chỉ cộng dịch chuyển nếu rect.bottom < ground_y TRƯỚC khi cộng.
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
