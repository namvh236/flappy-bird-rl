"""Draws `GameState` to the screen; no game logic, just reads state and blits
assets. Assets load once in `__init__`, unlike legacy which reloads them per
instance (a known inefficiency, not copied here).
"""

from __future__ import annotations

from pathlib import Path

import pygame

from ..game.config import GameConfig
from ..game.core import GameState

ASSETS = Path(__file__).resolve().parent.parent / "assets" / "img"
WHITE = (255, 255, 255)


class PygameRenderer:
    """`render_mode`: "human" shows a window and ticks fps; "rgb_array"/None draws
    off-screen without ticking, for recording or pixel obs.
    """

    def __init__(self, config: GameConfig, render_mode: str | None = "human", overlay=None):
        self.config = config
        self.render_mode = render_mode
        self.overlay = overlay
        self.paused = False
        self.fps_scale = 1.0

        if not pygame.get_init():
            pygame.init()

        # Both modes need display.set_mode for .convert(); "headless" means setting
        # SDL_VIDEODRIVER=dummy before renderer init, not skipping display.set_mode.
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

        # 'Bauhaus 93' isn't on Linux; checking first avoids SysFont's fallback warning.
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
