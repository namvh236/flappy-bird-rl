"""Tests for `record.py` (checklist P6)."""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import imageio

from flappy_ai.evaluation.record import record


def test_record_writes_gif_with_frames(tmp_path):
    out_path = tmp_path / "out.gif"
    result = record("random", out=out_path, seed=0, max_frames=30)

    assert result == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0

    frames = imageio.mimread(out_path)
    assert len(frames) >= 1
    assert frames[0].shape[:3] == (936, 864, 3)
