"""Hằng số vật lý/kích thước của game, gom vào một chỗ (spec §5.1).

Toàn bộ giá trị mặc định bằng đúng bản gốc `game/legacy.py`, để `FlappyGame`
với `GameConfig()` mặc định tái tạo đúng "cảm giác" của bản gốc.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GameConfig:
    screen_width: int = 864
    screen_height: int = 936
    ground_y: int = 768
    fps: int = 60
    gravity: float = 0.5
    max_fall_speed: float = 8.0
    flap_velocity: float = -10.0
    bird_start: tuple[int, int] = (100, 468)
    bird_size: tuple[int, int] = (51, 36)
    pipe_width: int = 78
    pipe_height: int = 560
    pipe_gap: int = 150
    pipe_offset_range: tuple[int, int] = (-100, 100)
    scroll_speed: int = 4
    pipe_interval_frames: int = 90  # = 1500 ms * 60 fps (bản gốc dùng get_ticks(), P3 fix)
    ceiling_kills: bool = True  # rect.top < 0 => chết (như bản gốc)
    float_physics: bool = False  # False = giữ int(vel) như bản gốc (P8)
