"""Unit tests for `FlappyGame`; each test checks one physics/rule invariant."""

from __future__ import annotations

from flappy_ai.game.config import GameConfig
from flappy_ai.game.core import FlappyGame
from flappy_ai.game.entities import PipePair


def test_spawn_position_matches_legacy_center_to_topleft():
    cfg = GameConfig()
    game = FlappyGame(cfg, seed=0)
    state = game.reset(seed=0)
    # Original: Bird(100, 468) sets rect.center=[100,468] -> topleft (75, 450)
    # with bird_size=(51, 36).
    assert state.bird_rect.x == 75
    assert state.bird_y == 450.0
    assert state.bird_vel == 0.0
    assert state.score == 0
    assert not state.dead


def test_gravity_accumulates_and_clamps_to_max_fall_speed():
    game = FlappyGame(GameConfig(), seed=0)
    game.reset(seed=0)
    for _ in range(3):
        game.step(False)
    assert game.state.bird_vel == 1.5  # 0.5 * 3, not yet touching ground within 3 frames
    for _ in range(40):
        result = game.step(False)
        if result.dead:
            break
    assert game.state.bird_vel <= 8.0


def test_flap_sets_velocity_next_frame_not_current_frame():
    """Original: gravity+move runs before flap in the same frame, so flap
    velocity only takes effect on the next frame."""
    game = FlappyGame(GameConfig(), seed=1)
    game.reset(seed=1)
    y_before = game.state.bird_y
    game.step(True)
    assert game.state.bird_vel == -10.0
    # Flap frame: movement still uses vel BEFORE flap (0.5 -> int 0), so y is unchanged.
    assert game.state.bird_y == y_before
    game.step(False)
    # Next frame: gravity adds to -10 -> -9.5.
    assert game.state.bird_vel == -9.5


def test_dies_on_ground():
    game = FlappyGame(GameConfig(), seed=2)
    game.reset(seed=2)
    result = None
    for _ in range(200):
        result = game.step(False)
        if result.dead:
            break
    assert result is not None and result.dead
    assert result.death_cause == "ground"
    assert game.state.bird_rect.bottom >= game.config.ground_y


def test_dies_on_ceiling():
    game = FlappyGame(GameConfig(), seed=3)
    game.reset(seed=3)
    result = None
    # Flapping every frame pins vel to -10, gravity then adds -9.5 (int
    # -9/frame) => climbs steadily at 9px/frame, hitting y<0 at frame 52.
    for _ in range(80):
        result = game.step(True)
        if result.dead:
            break
    assert result is not None and result.dead
    assert result.death_cause == "ceiling"
    assert game.state.bird_rect.top < 0


def test_dies_on_pipe_collision():
    cfg = GameConfig()
    game = FlappyGame(cfg, seed=4)
    game.reset(seed=4)
    # Place a pipe directly overlapping the bird (bird at y=450..486, a
    # narrow gap shifted high so the bird is guaranteed to hit the top pipe).
    game.pipes = [PipePair(x=75.0, gap_center=900.0)]
    result = game.step(False)
    assert result.dead
    assert result.death_cause == "pipe"


def test_scores_once_per_pipe_no_double_count():
    cfg = GameConfig()
    game = FlappyGame(cfg, seed=5)
    game.reset(seed=5)
    # Place the pipe right at the "fully passed the bird" threshold (pipe.right < bird.left).
    game.pipes = [PipePair(x=75 - cfg.pipe_width - cfg.scroll_speed + 1, gap_center=468.0)]
    result = game.step(False)
    assert result.passed_pipe
    assert game.state.score == 1
    result2 = game.step(False)
    assert not result2.passed_pipe
    assert game.state.score == 1  # not double-counted


def test_reset_clears_all_state_between_episodes():
    game = FlappyGame(GameConfig(), seed=6)
    game.reset(seed=6)
    for _ in range(80):
        if game.step(False).dead:
            break
    assert game.state.dead
    assert game.state.frame > 0

    state = game.reset(seed=6)
    assert state.frame == 0
    assert state.score == 0
    assert state.bird_vel == 0.0
    assert state.ground_scroll == 0
    assert not state.dead
    assert state.pipes == []


def test_same_seed_produces_same_pipe_sequence():
    def gap_centers(seed: int, n_steps: int = 400) -> list[float]:
        game = FlappyGame(GameConfig(), seed=seed)
        game.reset(seed=seed)
        seen = []
        prev_count = 0
        for i in range(n_steps):
            game.step(i % 12 == 0)
            if len(game.pipes) > prev_count:
                seen.append(game.pipes[-1].gap_center)
                prev_count = len(game.pipes)
        return seen

    seq_a = gap_centers(99)
    seq_b = gap_centers(99)
    assert seq_a == seq_b
    assert len(seq_a) > 0


def test_different_seed_can_produce_different_pipe_sequence():
    def first_gap(seed: int) -> float:
        game = FlappyGame(GameConfig(), seed=seed)
        game.reset(seed=seed)
        game.step(False)
        return game.pipes[0].gap_center

    gaps = {first_gap(s) for s in range(10)}
    assert len(gaps) > 1  # different seeds don't always produce the same offset
