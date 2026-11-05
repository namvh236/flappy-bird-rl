"""Watches an agent play on `FlappyBirdEnv` with `human` rendering. The human
agent (`--agent human`) still uses `game/play_human.py`, controlling directly
via mouse/keyboard instead of the `Agent` interface.

Keys while watching: `+`/`-` speed up/down, `P` pause, `R` restart the
episode, `Esc`/close window to quit.
"""

from __future__ import annotations

import pygame

from ..agents.base import Agent
from ..agents.factory import build_agent
from ..env.flappy_env import FlappyBirdEnv
from ..game.config import GameConfig
from ..render.pygame_renderer import PygameRenderer

SPEED_STEP = 2.0
SPEED_MIN = 0.25
SPEED_MAX = 8.0


def run_agent_play(
    agent_name: str,
    fps: int = 60,
    seed: int | None = None,
    model_path: str | None = None,
    genome_path: str | None = None,
    neat_config: str | None = None,
    debug: bool = False,
    pixel_obs: bool = False,
) -> None:
    agent: Agent = build_agent(
        agent_name, model_path=model_path, genome_path=genome_path, neat_config=neat_config
    )
    config = GameConfig(fps=fps)
    overlay = None
    if debug:
        from ..render.overlay import DebugOverlay

        overlay = DebugOverlay(config)

    viewer: PygameRenderer | None = None
    if pixel_obs:
        from ..env.wrappers import PixelObs

        env = PixelObs(FlappyBirdEnv(config=config, render_mode="rgb_array"))
        viewer = PygameRenderer(config, render_mode="human", overlay=overlay)
        viewer_ref = viewer
    else:
        env = FlappyBirdEnv(config=config, render_mode="human", debug_overlay=debug)
        viewer_ref = env.renderer

    obs, info = env.reset(seed=seed)
    episode = 0
    best_score = 0
    hud: dict = {}

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                viewer_ref.paused = not viewer_ref.paused
            elif event.type == pygame.KEYDOWN and event.key in (
                pygame.K_PLUS,
                pygame.K_EQUALS,
                pygame.K_KP_PLUS,
            ):
                viewer_ref.fps_scale = min(SPEED_MAX, viewer_ref.fps_scale * SPEED_STEP)
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                viewer_ref.fps_scale = max(SPEED_MIN, viewer_ref.fps_scale / SPEED_STEP)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                episode += 1
                obs, info = env.reset(seed=None if seed is None else seed + episode)
        if not running:
            break

        if viewer is not None:
            viewer.draw(env.unwrapped.game.state, extra=hud)

        if viewer_ref.paused:
            continue

        action = agent.act(obs, info)
        hud = {
            "action": action,
            "explain": agent.explain(),
            "episode": episode,
            "best_score": best_score,
            "fps": viewer_ref.config.fps * viewer_ref.fps_scale,
            "paused": viewer_ref.paused,
        }
        env.unwrapped.hud = hud
        obs, _reward, terminated, truncated, info = env.step(action)
        best_score = max(best_score, int(info["score"]))
        if terminated or truncated:
            episode += 1
            obs, info = env.reset(seed=None if seed is None else seed + episode)

    env.close()
    if viewer is not None:
        viewer.close()
