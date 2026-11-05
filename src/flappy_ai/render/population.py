"""Xem cả quần thể NEAT bay cùng lúc (docs/plan/plan.md:131, checklist.md:85).

Mọi game dùng cùng seed nên bố cục pipe giống hệt nhau giữa các cá thể — chỉ
cần vẽ bg/pipe/ground một lần (theo game đầu tiên) rồi chồng bird của từng
game lên, tint màu khác nhau để phân biệt.
"""

from __future__ import annotations

from pathlib import Path

import pygame

from ..env.observation import build_features
from ..game.config import GameConfig
from ..game.core import FlappyGame, GameState
from .pygame_renderer import WHITE, PygameRenderer

_COLORS = [
    (255, 90, 90),
    (90, 255, 90),
    (90, 160, 255),
    (255, 255, 90),
    (255, 90, 255),
    (90, 255, 255),
    (255, 160, 90),
    (170, 90, 255),
    (220, 220, 220),
    (90, 220, 150),
]
_DEAD_TINT = (120, 120, 120, 110)


class PopulationRenderer(PygameRenderer):
    """Vẽ nhiều `GameState` (cùng seed) chồng lên một nền pipe/ground duy nhất."""

    def __init__(self, config: GameConfig, render_mode: str | None = "human"):
        super().__init__(config, render_mode)
        self.label_font = pygame.font.Font(None, 20)

    def draw_population(self, states: list[GameState]) -> None:
        cfg = self.config
        self.screen.blit(self.bg, (0, 0))

        first = states[0]
        for pipe in first.pipes:
            top_topleft = (pipe.x, pipe.gap_top - cfg.pipe_height)
            bottom_topleft = (pipe.x, pipe.gap_bottom)
            self.screen.blit(self.pipe_img_flipped, top_topleft)
            self.screen.blit(self.pipe_img, bottom_topleft)

        self._anim_counter += 1
        if self._anim_counter > 5:
            self._anim_counter = 0
            self._anim_index = (self._anim_index + 1) % len(self.bird_imgs)
        base_img = self.bird_imgs[self._anim_index]

        for i, state in enumerate(states):
            angle = -90.0 if state.dead else state.bird_vel * -2
            bird_img = pygame.transform.rotate(base_img, angle).copy()
            tint = _DEAD_TINT if state.dead else _COLORS[i % len(_COLORS)]
            bird_img.fill(tint, special_flags=pygame.BLEND_RGBA_MULT)
            self.screen.blit(bird_img, (state.bird_rect.x, state.bird_rect.y))

            label = self.label_font.render(f"#{i} {state.score}", True, WHITE)
            self.screen.blit(label, (state.bird_rect.x, state.bird_rect.y - label.get_height()))

        self.screen.blit(self.ground_img, (first.ground_scroll, cfg.ground_y))

        if self.render_mode == "human":
            pygame.display.update()
            self.clock.tick(cfg.fps)


def _latest_neat_model_dir() -> Path:
    candidates = sorted(
        p
        for p in Path("models").glob("neat_*")
        if p.is_dir() and (p / "best_genome.pkl").exists() and (p / "config.cfg").exists()
    )
    if not candidates:
        raise RuntimeError(
            "Không tìm thấy model NEAT nào trong models/neat_*/ — chạy "
            "`flappy-ai train --algo neat` trước, hoặc truyền genome_path/config_path."
        )
    return candidates[-1]


def run_population_view(
    config_path: str | Path | None = None,
    genome_path: str | Path | None = None,
    n: int = 10,
    seed: int = 0,
    fps: int = 60,
    max_frames: int | None = 3600,
) -> None:
    """Thả `n` bản cùng genome NEAT (từ P5) bay song song trên cùng một seed.

    `config_path`/`genome_path` mặc định lấy từ model NEAT mới nhất trong
    `models/neat_*/` (file `config.cfg` + `best_genome.pkl`, xem `train_neat.py`).
    """
    try:
        from ..agents.neat_agent import NeatAgent
    except ImportError as exc:
        raise RuntimeError(
            "Chưa có flappy_ai.agents.neat_agent.NeatAgent — chạy train_neat.py "
            "(P5) để train và lưu genome trước khi xem population."
        ) from exc

    if config_path is None or genome_path is None:
        model_dir = _latest_neat_model_dir()
        if config_path is None:
            config_path = model_dir / "config.cfg"
        if genome_path is None:
            genome_path = model_dir / "best_genome.pkl"

    config = GameConfig(fps=fps)
    games = [FlappyGame(config, seed=seed) for _ in range(n)]
    agents = [NeatAgent(genome_path, config_path) for _ in range(n)]
    renderer = PopulationRenderer(config, render_mode="human")

    frame = 0
    try:
        while max_frames is None or frame < max_frames:
            states = [game.state for game in games]
            if all(state.dead for state in states):
                break
            for game, agent, state in zip(games, agents, states, strict=True):
                if state.dead:
                    continue
                obs = build_features(state, config)
                game.step(bool(agent.act(obs)))
            renderer.draw_population([game.state for game in games])
            frame += 1
    finally:
        renderer.close()
