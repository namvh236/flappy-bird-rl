"""`NeatAgent`: loads a trained NEAT genome (pickle) + `neat.cfg`, builds a
`neat.nn.FeedForwardNetwork` to infer the flap action."""

from __future__ import annotations

import pickle
from pathlib import Path

import neat
import numpy as np


class NeatAgent:
    def __init__(self, genome_path: str | Path, config_path: str | Path):
        with open(genome_path, "rb") as f:
            genome = pickle.load(f)
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            str(config_path),
        )
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)
        self._last_explain: dict = {}

    def act(self, obs: np.ndarray, info: dict | None = None) -> int:
        output = float(self.net.activate(obs)[0])
        flap = output > 0.5
        self._last_explain = {"output": output, "flap": flap}
        return 1 if flap else 0

    def explain(self) -> dict:
        return self._last_explain
