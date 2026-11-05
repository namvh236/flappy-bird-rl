"""Interface chung cho mọi agent (spec §5.6)."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Agent(Protocol):
    def act(self, obs: np.ndarray, info: dict | None = None) -> int: ...
    def explain(self) -> dict: ...
