"""`FlappyGame`: logic thuần, không phụ thuộc `pygame.display` (spec §5.2).

Thứ tự xử lý trong một `step()` (giữ đúng spec, xem ghi chú §3.1 về khác biệt
có chủ đích so với vòng lặp frame của bản gốc — §7 liệt kê rõ):

1. Flap (nếu còn sống) → đặt vận tốc.
2. Trọng lực + di chuyển bird.
3. Sinh pipe mới nếu đã đủ `pipe_interval_frames`.
4. Cuộn pipe sang trái, xoá pipe ra khỏi màn hình.
5. Tính điểm: pipe đã qua hẳn bird và chưa tính ⇒ +1, đánh dấu `passed`.
6. Kiểm tra va chạm: pipe → ceiling → ground (đúng thứ tự ưu tiên trong spec).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

from .config import GameConfig
from .entities import Bird, PipePair
from .rect import Rect

DeathCause = Literal["pipe", "ceiling", "ground"]


@dataclass
class PipePairState:
    x: float
    gap_top: float
    gap_bottom: float
    passed: bool


@dataclass
class GameState:
    frame: int
    score: int
    bird_y: float
    bird_vel: float
    bird_rect: Rect
    pipes: list[PipePairState]
    ground_scroll: int
    dead: bool


@dataclass
class StepResult:
    passed_pipe: bool
    dead: bool
    death_cause: DeathCause | None


class FlappyGame:
    def __init__(self, config: GameConfig = GameConfig(), seed: int | None = None):
        self.config = config
        self._rng = random.Random(seed)
        self.reset(seed)

    def reset(self, seed: int | None = None) -> GameState:
        """Khởi tạo lại TOÀN BỘ state (sửa P6: bản gốc quên reset vel/pass_pipe/...)."""
        if seed is not None:
            self._rng = random.Random(seed)
        self.frame = 0
        self.score = 0
        self.bird = Bird.spawn(self.config)
        self.pipes: list[PipePair] = []
        self.ground_scroll = 0
        self.dead = False
        self.death_cause: DeathCause | None = None
        # Bản gốc backdate `last_pipe` để pipe đầu tiên sinh ngay frame đang bay
        # đầu tiên; ở đây tương đương: coi như đã "đến hạn" tại frame 0.
        self._last_spawn_frame = -self.config.pipe_interval_frames
        return self.state

    def step(self, flap: bool) -> StepResult:
        if self.dead:
            return StepResult(passed_pipe=False, dead=True, death_cause=self.death_cause)

        cfg = self.config

        self.bird.step_physics(cfg, flap)

        if self.frame - self._last_spawn_frame >= cfg.pipe_interval_frames:
            offset = self._rng.randint(*cfg.pipe_offset_range)
            gap_center = float(cfg.screen_height // 2 + offset)
            self.pipes.append(PipePair(x=float(cfg.screen_width), gap_center=gap_center))
            self._last_spawn_frame = self.frame

        for pipe in self.pipes:
            pipe.scroll(cfg)
        self.pipes = [p for p in self.pipes if not p.offscreen(cfg)]

        self.ground_scroll -= cfg.scroll_speed
        if abs(self.ground_scroll) > 35:
            self.ground_scroll = 0

        bird_rect = self.bird.rect(cfg)

        passed_pipe = False
        for pipe in self.pipes:
            if not pipe.passed and pipe.x + cfg.pipe_width < bird_rect.left:
                pipe.passed = True
                self.score += 1
                passed_pipe = True

        death_cause: DeathCause | None = None
        for pipe in self.pipes:
            if bird_rect.colliderect(pipe.top_rect(cfg)) or bird_rect.colliderect(
                pipe.bottom_rect(cfg)
            ):
                death_cause = "pipe"
                break
        if death_cause is None and cfg.ceiling_kills and bird_rect.top < 0:
            death_cause = "ceiling"
        if death_cause is None and bird_rect.bottom >= cfg.ground_y:
            death_cause = "ground"

        if death_cause is not None:
            self.dead = True
            self.death_cause = death_cause

        self.frame += 1
        return StepResult(passed_pipe=passed_pipe, dead=self.dead, death_cause=death_cause)

    @property
    def state(self) -> GameState:
        cfg = self.config
        bird_rect = self.bird.rect(cfg)
        pipes_state = [
            PipePairState(
                x=p.x,
                gap_top=p.gap_center - cfg.pipe_gap / 2,
                gap_bottom=p.gap_center + cfg.pipe_gap / 2,
                passed=p.passed,
            )
            for p in self.pipes
        ]
        return GameState(
            frame=self.frame,
            score=self.score,
            bird_y=self.bird.y,
            bird_vel=self.bird.vel,
            bird_rect=bird_rect,
            pipes=pipes_state,
            ground_scroll=self.ground_scroll,
            dead=self.dead,
        )
