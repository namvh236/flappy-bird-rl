"""`DebugOverlay`: vẽ đường cảm biến tới pipe kế tiếp + panel debug (action,
explain, episode, fps) lên trên `PygameRenderer.draw()` (spec §5.3).
"""

from __future__ import annotations

import pygame

from ..env.observation import next_pipes
from ..game.config import GameConfig
from ..game.core import GameState

GREEN = (0, 220, 0)
GRAY = (160, 160, 160)
PANEL_BG = (0, 0, 0)
PANEL_FG = (255, 255, 255)
PANEL_ALPHA = 160


class DebugOverlay:
    def __init__(self, config: GameConfig):
        self.config = config
        pygame.font.init()
        self.font = pygame.font.Font(None, 22)

    def _draw_sensor_line(self, surface: pygame.Surface, state: GameState) -> None:
        cfg = self.config
        pipe1, _pipe2 = next_pipes(state, cfg)
        bird_center = (state.bird_rect.centerx, state.bird_rect.centery)
        gap_center_y = (pipe1.gap_top + pipe1.gap_bottom) / 2
        pipe_center = (pipe1.x + cfg.pipe_width / 2, gap_center_y)
        is_real = any(p.x == pipe1.x for p in state.pipes)
        color = GREEN if is_real else GRAY
        pygame.draw.line(surface, color, bird_center, pipe_center, 2)
        pygame.draw.circle(surface, color, (int(pipe_center[0]), int(pipe_center[1])), 4)

    def _format_explain(self, explain: dict) -> list[str]:
        lines: list[str] = []
        if "q_values" in explain:
            values = ", ".join(f"{v:.2f}" for v in explain["q_values"])
            lines.append(f"q: [{values}]")
        if "probs" in explain:
            values = ", ".join(f"{v:.2f}" for v in explain["probs"])
            lines.append(f"probs: [{values}]")
        if "output" in explain:
            lines.append(f"output: {explain['output']:.3f}")
        for key, value in explain.items():
            if key in ("q_values", "probs", "output"):
                continue
            lines.append(f"{key}: {value}")
        return lines

    def _draw_panel(self, surface: pygame.Surface, hud: dict) -> None:
        lines: list[str] = []
        if "action" in hud:
            lines.append(f"action: {'flap' if hud['action'] == 1 else '-'}")
        lines.extend(self._format_explain(hud.get("explain", {})))
        if "episode" in hud:
            episode_line = f"episode: {hud['episode']}"
            if "total_episodes" in hud:
                episode_line += f"/{hud['total_episodes']}"
            lines.append(episode_line)
        if "best_score" in hud:
            lines.append(f"best: {hud['best_score']}")
        if "fps" in hud:
            lines.append(f"fps: {hud['fps']:.1f}")
        if hud.get("paused"):
            lines.append("PAUSED")
        if not lines:
            return

        text_surfaces = [self.font.render(line, True, PANEL_FG) for line in lines]
        width = max(s.get_width() for s in text_surfaces) + 12
        height = sum(s.get_height() for s in text_surfaces) + 12

        panel = pygame.Surface((width, height), pygame.SRCALPHA)
        panel.fill((*PANEL_BG, PANEL_ALPHA))
        y = 6
        for text_surface in text_surfaces:
            panel.blit(text_surface, (6, y))
            y += text_surface.get_height()
        surface.blit(panel, (8, 8))

    def draw(self, surface: pygame.Surface, state: GameState, hud: dict) -> None:
        self._draw_sensor_line(surface, state)
        self._draw_panel(surface, hud)
