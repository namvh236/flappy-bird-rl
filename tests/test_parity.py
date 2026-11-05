"""So sánh quỹ đạo `bird.y` giữa `FlappyGame` (core mới) và `game/legacy.py`
(bản gốc, đã vá đường dẫn asset) trong 300 frame đầu (docs/plan/checklist.md
P1, spec §7).

`legacy.py` là một script nguyên khối chạy vòng lặp `while run:` ngay khi
import (không có `reset()`/`step()`), nên để "chơi thử" nó từ test, ta patch
input/thời gian (đúng như spec §7 mô tả) rồi để nó tự thoát vòng lặp sau đúng
300 frame bằng một sự kiện QUIT giả.

Kịch bản chọn: **rơi tự do, không bao giờ flap**. Với `bird.x` cố định = 75
và pipe đầu tiên chỉ tới gần vị trí đó ở khoảng frame ~274 (pipe_frequency
1500 ms ⇒ frame 90, cách bird ~184 frame di chuyển), trong khi rơi tự do
chạm đất ở khoảng frame ~43 và đứng yên vĩnh viễn sau đó — nên toàn bộ 300
frame so sánh chỉ phụ thuộc gravity/flap/ground, hoàn toàn tách biệt khỏi
khác biệt cố ý về thứ tự pipe/scoring giữa bản gốc và refactor (spec §7).
"""

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
            # Bỏ giới hạn fps thật (không cần chờ wall-clock trong test) và ép
            # bay ngay từ frame đầu, bỏ qua pha "chờ click" (P9, cũng là hành
            # vi env sau refactor). `pygame.time.Clock` là kiểu C bất biến,
            # không patch được method trực tiếp nên thay cả class.
            mod = sys.modules.get(LEGACY_MODULE)
            if mod is not None:
                mod.flying = True
            return 0

    def fake_get_pressed():
        return (0, 0, 0)  # không bao giờ flap

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
        import flappy_ai.game.legacy  # noqa: F401 (chạy vòng lặp rồi tự thoát qua QUIT giả)

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
        f"lệch quỹ đạo tại frame {mismatches[:5]}: "
        f"legacy={[legacy_traj[i] for i in mismatches[:5]]} "
        f"core={[core_traj[i] for i in mismatches[:5]]}"
    )

    # Chốt luôn hành vi "đứng yên sau khi chết" khớp nhau (cả hai đều chạm đất
    # và đóng băng ở cùng giá trị trong phần đuôi quỹ đạo).
    assert legacy_traj[-1] == core_traj[-1]
