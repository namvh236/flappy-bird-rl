"""Chơi tay trên `FlappyGame` + `PygameRenderer` (P1). Click chuột hoặc phím
Space để flap (edge-triggered, mỗi lần nhấn = một cú flap). Sau khi chết,
click vào nút restart để chơi lại. Đóng cửa sổ để thoát.
"""

from __future__ import annotations

import pygame

from ..render.pygame_renderer import PygameRenderer
from .config import GameConfig
from .core import FlappyGame


def run_human_play(fps: int = 60, seed: int | None = None) -> None:
    config = GameConfig(fps=fps)
    game = FlappyGame(config, seed=seed)
    renderer = PygameRenderer(config, render_mode="human")

    running = True
    while running:
        flap = False
        restart_click = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                flap = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                flap = True
                restart_click = True  # cùng một click: flap khi sống, restart khi đã chết

        if not running:
            break

        if game.state.dead:
            renderer.draw(game.state)
            button_rect = renderer.draw_restart_button()
            pygame.display.update()
            renderer.clock.tick(config.fps)
            if restart_click and button_rect.collidepoint(pygame.mouse.get_pos()):
                game.reset(seed=seed)
        else:
            game.step(flap)
            renderer.draw(game.state)

    renderer.close()


if __name__ == "__main__":
    run_human_play()
