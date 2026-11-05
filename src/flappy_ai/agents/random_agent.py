"""`RandomAgent`: flaps with probability `flap_prob`, a lower-bound baseline
for learned agents."""

from __future__ import annotations

import random

import numpy as np


class RandomAgent:
    def __init__(self, flap_prob: float = 0.5, seed: int | None = None):
        self.flap_prob = flap_prob
        self._rng = random.Random(seed)

    def act(self, obs: np.ndarray, info: dict | None = None) -> int:
        return 1 if self._rng.random() < self.flap_prob else 0

    def explain(self) -> dict:
        return {"flap_prob": self.flap_prob}
