"""Compares `bird.y` trajectory between `FlappyGame` and `game/legacy.py`
over 300 free-fall frames (no flaps) to verify physics parity."""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

from flappy_ai.game.config import GameConfig
from flappy_ai.game.core import FlappyGame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

FRAMES_TO_COMPARE = 300
LEGACY_MODULE = "flappy_ai.game.legacy"


def _run_legacy_freefall(frames: int) -> list[float]:
    import pygame

    trajectory: list[float] = []
    counter = {"n": 0}

    class _FakeClock:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def tick(self, *args, **kwargs) -> int:
            # Skip the real fps wait and force flying immediately; Clock is a
            # C type so we replace the whole class instead of patching a method.
            mod = sys.modules.get(LEGACY_MODULE)
            if mod is not None:
                mod.flying = True
            return 0

    def fake_get_pressed():
        return (0, 0, 0)  # never flap

    def fake_event_get(*args, **kwargs):
        mod = sys.modules.get(LEGACY_MODULE)
        if mod is not None:
            trajectory.append(float(mod.flappy.rect.y))
        counter["n"] += 1
        if counter["n"] >= frames:
            return [pygame.event.Event(pygame.QUIT)]
        return []

    with (
        patch("pygame.time.Clock", _FakeClock),
        patch("pygame.mouse.get_pressed", fake_get_pressed),
        patch("pygame.event.get", fake_event_get),
    ):
        sys.modules.pop(LEGACY_MODULE, None)
        import flappy_ai.game.legacy  # noqa: F401 (runs the loop then exits via a fake QUIT event)

    return trajectory


def _run_core_freefall(frames: int) -> list[float]:
    game = FlappyGame(GameConfig())
    game.reset(seed=0)
    trajectory = []
    for _ in range(frames):
        game.step(False)
        trajectory.append(game.state.bird_y)
    return trajectory


def test_freefall_trajectory_matches_legacy_for_300_frames():
    legacy_traj = _run_legacy_freefall(FRAMES_TO_COMPARE)
    core_traj = _run_core_freefall(FRAMES_TO_COMPARE)

    assert len(legacy_traj) == FRAMES_TO_COMPARE
    assert len(core_traj) == FRAMES_TO_COMPARE

    mismatches = [i for i, (a, b) in enumerate(zip(legacy_traj, core_traj)) if a != b]
    assert not mismatches, (
        f"trajectory mismatch at frame {mismatches[:5]}: "
        f"legacy={[legacy_traj[i] for i in mismatches[:5]]} "
        f"core={[core_traj[i] for i in mismatches[:5]]}"
    )

    # Also confirms post-death behavior matches: both land and freeze at the same value.
    assert legacy_traj[-1] == core_traj[-1]
