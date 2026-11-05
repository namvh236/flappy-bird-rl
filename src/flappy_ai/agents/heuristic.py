"""`HeuristicAgent`: luật cứng — flap khi bird đang rơi và sắp chạm mép dưới
khe pipe kế tiếp (spec §5.6), để kiểm chứng env "chơi được" mà không cần học.

Ngưỡng tính trên feature #4 đã chuẩn hoá của `observation.py`
(`(gap_bottom - bird.bottom) / screen_height`), không cần biết pixel thật.
"""

from __future__ import annotations

import numpy as np

from ..game.config import GameConfig

_DEFAULT_SCREEN_HEIGHT = GameConfig().screen_height


class HeuristicAgent:
    def __init__(self, margin: float = 20.0, screen_height: float = _DEFAULT_SCREEN_HEIGHT):
        self.margin = margin
        self._margin_norm = margin / screen_height

    def act(self, obs: np.ndarray, info: dict | None = None) -> int:
        falling = obs[1] >= 0
        near_gap_bottom = obs[4] < self._margin_norm
        return 1 if (falling and near_gap_bottom) else 0

    def explain(self) -> dict:
        return {"margin": self.margin}
