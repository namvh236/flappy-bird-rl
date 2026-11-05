import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from flappy_ai.game.config import GameConfig  # noqa: E402
from flappy_ai.game.core import FlappyGame  # noqa: E402
from flappy_ai.render.population import PopulationRenderer  # noqa: E402

N_GAMES = 3
N_FRAMES = 5


def test_draw_population_runs_headless_with_distinct_bird_positions():
    config = GameConfig()
    games = [FlappyGame(config, seed=0) for _ in range(N_GAMES)]
    renderer = PopulationRenderer(config, render_mode="rgb_array")

    for frame in range(N_FRAMES):
        for i, game in enumerate(games):
            flap = frame % (i + 2) == 0
            game.step(flap)
        states = [game.state for game in games]
        renderer.draw_population(states)

    assert len(states) == N_GAMES

    bird_ys = [state.bird_y for state in states]
    assert len(set(bird_ys)) == N_GAMES

    renderer.close()
