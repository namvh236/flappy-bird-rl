"""`RandomAgent`: flap ngẫu nhiên với xác suất `flap_prob` — sàn điểm (lower
bound) để so sánh với các agent có học (spec §5.6).
"""

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
