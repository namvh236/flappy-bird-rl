"""Vẽ `GameState` lên màn hình (spec §5.3). Không chứa logic game — chỉ đọc
state và blit asset. Asset nạp một lần duy nhất ở `__init__` (khác bản gốc,
vốn nạp lại ảnh pipe cho mỗi instance — điểm kém hiệu quả đã biết, không copy
theo, xem AGENTS.md).
"""

from __future__ import annotations

from pathlib import Path

import pygame

from ..game.config import GameConfig
from ..game.core import GameState

ASSETS = Path(__file__).resolve().parent.parent / "assets" / "img"
WHITE = (255, 255, 255)


class PygameRenderer:
    """`render_mode`: "human" (cửa sổ + tick fps) hoặc "rgb_array"/None (vẽ lên
    `Surface` ngoài màn hình, không tick fps — dùng cho ghi hình hoặc pixel obs).
    """

    def __init__(self, config: GameConfig, render_mode: str | None = "human", overlay=None):
        self.config = config
        self.render_mode = render_mode
        self.overlay = overlay
        self.paused = False
        self.fps_scale = 1.0

        if not pygame.get_init():
            pygame.init()

        # Cả hai chế độ đều đi qua display.set_mode: .convert()/.convert_alpha()
        # cần một display mode đã khởi tạo. "Headless" đạt được bằng cách đặt
        # biến môi trường SDL_VIDEODRIVER=dummy TRƯỚC khi renderer khởi tạo
        # (xem architecture.md:260), không phải bằng cách né display.set_mode.
        self.screen = pygame.display.set_mode((config.screen_width, config.screen_height))
        if render_mode == "human":
            pygame.display.set_caption("Flappy Bird")

        self.clock = pygame.time.Clock()

        self.bg = pygame.image.load(str(ASSETS / "bg.png")).convert()
        self.ground_img = pygame.image.load(str(ASSETS / "ground.png")).convert_alpha()
        self.button_img = pygame.image.load(str(ASSETS / "restart.png")).convert_alpha()
        self.pipe_img = pygame.image.load(str(ASSETS / "pipe.png")).convert_alpha()
        self.pipe_img_flipped = pygame.transform.flip(self.pipe_img, False, True)
        self.bird_imgs = [
            pygame.image.load(str(ASSETS / f"bird{n}.png")).convert_alpha() for n in (1, 2, 3)
        ]

        # P10 fix: 'Bauhaus 93' không có trên Linux; SysFont tự fallback nhưng
        # in cảnh báo — dùng font mặc định pygame thẳng nếu không tìm thấy.
        if pygame.font.match_font("Bauhaus 93"):
            self.font = pygame.font.SysFont("Bauhaus 93", 60)
        else:
            self.font = pygame.font.Font(None, 60)

        self._anim_index = 0
        self._anim_counter = 0

    def _bird_image(self, state: GameState) -> pygame.Surface:
        self._anim_counter += 1
        if self._anim_counter > 5:
            self._anim_counter = 0
            self._anim_index = (self._anim_index + 1) % len(self.bird_imgs)
        base = self.bird_imgs[self._anim_index]
        angle = -90.0 if state.dead else state.bird_vel * -2
        return pygame.transform.rotate(base, angle)

    def draw(self, state: GameState, extra: dict | None = None) -> None:
        cfg = self.config
        self.screen.blit(self.bg, (0, 0))

        for pipe in state.pipes:
            top_topleft = (pipe.x, pipe.gap_top - cfg.pipe_height)
            bottom_topleft = (pipe.x, pipe.gap_bottom)
            self.screen.blit(self.pipe_img_flipped, top_topleft)
            self.screen.blit(self.pipe_img, bottom_topleft)

        self.screen.blit(self._bird_image(state), (state.bird_rect.x, state.bird_rect.y))
        self.screen.blit(self.ground_img, (state.ground_scroll, cfg.ground_y))

        score_img = self.font.render(str(state.score), True, WHITE)
        self.screen.blit(score_img, (cfg.screen_width // 2, 20))

        if self.overlay is not None and extra is not None:
            self.overlay.draw(self.screen, state, extra)

        if self.render_mode == "human":
            pygame.display.update()
            if self.paused:
                pygame.time.delay(10)
            else:
                self.clock.tick(max(1, int(cfg.fps * self.fps_scale)))

    def draw_restart_button(self) -> pygame.Rect:
        cfg = self.config
        rect = self.button_img.get_rect()
        rect.topleft = (cfg.screen_width // 2 - 50, cfg.screen_height // 2 - 100)
        self.screen.blit(self.button_img, rect.topleft)
        return rect

    def to_rgb_array(self):
        import numpy as np

        return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))

    def close(self) -> None:
        pygame.quit()
