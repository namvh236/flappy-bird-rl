"""`train_neat.py`: trains a NEAT agent directly on `FlappyGame` +
`build_features` (bypassing `FlappyBirdEnv`/Gymnasium so per-generation
fitness evaluation runs fast). Each run creates `runs/neat_<timestamp>_<pid>/`
(`config.cfg`, `git_commit.txt`, `results.json`, a checkpoint every 10
generations) and `models/neat_<same>/` (`best_genome.pkl` + `config.cfg`, so
`NeatAgent`/CLI can resolve the latest model without touching `runs/`).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import neat
import numpy as np

from ..env.observation import build_features
from ..game.config import GameConfig
from ..game.core import FlappyGame

DEFAULT_NEAT_CONFIG = Path("configs/neat.cfg")
MAX_EPISODE_STEPS = 20_000


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _run_episode(
    net: neat.nn.FeedForwardNetwork, seed: int, config: GameConfig
) -> tuple[float, int]:
    """Runs one episode; fitness = frames_alive + 100 * score."""
    game = FlappyGame(config, seed=seed)
    state = game.state
    frames_alive = 0
    for _ in range(MAX_EPISODE_STEPS):
        obs = build_features(state, config)
        flap = net.activate(obs)[0] > 0.5
        result = game.step(flap)
        frames_alive += 1
        state = game.state
        if result.dead:
            break
    return frames_alive + 100.0 * state.score, frames_alive


def _make_eval_function(episodes_per_genome: int, game_config: GameConfig, results: list[dict]):
    generation = {"n": 0}

    def eval_genomes(
        genomes: list[tuple[int, neat.DefaultGenome]], neat_config: neat.Config
    ) -> None:
        gen = generation["n"]
        generation["n"] += 1
        total_frames = 0
        fitnesses: list[float] = []
        for _, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, neat_config)
            episode_fitnesses = []
            for i in range(episodes_per_genome):
                # Seed varies by generation so genomes can't overfit to fixed seeds.
                episode_seed = 2000 + gen * episodes_per_genome + i
                fitness, frames = _run_episode(net, seed=episode_seed, config=game_config)
                episode_fitnesses.append(fitness)
                total_frames += frames
            genome.fitness = float(np.mean(episode_fitnesses))
            fitnesses.append(genome.fitness)

        best_fitness = max(fitnesses)
        mean_fitness = float(np.mean(fitnesses))
        results.append(
            {"generation": gen, "best_fitness": best_fitness, "mean_fitness": mean_fitness}
        )
        print(f"gen={gen} best={best_fitness:.1f} mean={mean_fitness:.1f} frames={total_frames}")

    return eval_genomes


def train_neat(
    config_path: Path = DEFAULT_NEAT_CONFIG,
    seed: int = 0,
    generations: int | None = None,
    episodes_per_genome: int | None = None,
    pop_size: int | None = None,
    out_root: Path | None = None,
) -> Path:
    episodes_per_genome = episodes_per_genome or 3
    generations = generations if generations is not None else 30
    game_config = GameConfig()

    random.seed(seed)
    np.random.seed(seed)

    neat_config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(config_path),
    )
    if pop_size is not None:
        neat_config.pop_size = pop_size

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"neat_{timestamp}_{os.getpid()}"
    root = out_root if out_root is not None else Path()
    run_dir = root / "runs" / run_name
    model_dir = root / "models" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(config_path, run_dir / "config.cfg")
    shutil.copy(config_path, model_dir / "config.cfg")
    (run_dir / "git_commit.txt").write_text(_git_commit_hash() + "\n")

    population = neat.Population(neat_config)
    population.add_reporter(
        neat.Checkpointer(
            generation_interval=10,
            time_interval_seconds=None,
            filename_prefix=str(run_dir / "neat-checkpoint-"),
        )
    )

    results: list[dict] = []
    eval_genomes = _make_eval_function(episodes_per_genome, game_config, results)
    best_genome = population.run(eval_genomes, n=generations)

    with (model_dir / "best_genome.pkl").open("wb") as f:
        pickle.dump(best_genome, f)
    with (run_dir / "results.json").open("w") as f:
        json.dump(results, f, indent=2)

    return model_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="flappy-ai train --algo neat")
    parser.add_argument("--config", type=str, default=str(DEFAULT_NEAT_CONFIG))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--episodes-per-genome", type=int, default=None)
    parser.add_argument("--pop-size", type=int, default=None)
    args = parser.parse_args(argv)

    model_dir = train_neat(
        Path(args.config),
        seed=args.seed,
        generations=args.generations,
        episodes_per_genome=args.episodes_per_genome,
        pop_size=args.pop_size,
    )
    print(f"Saved model to {model_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
