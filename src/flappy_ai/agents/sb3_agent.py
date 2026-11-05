"""`SB3Agent`: load model DQN/PPO đã train, cung cấp `explain()` cho HUD
(Q-values hoặc xác suất flap) (spec §5.6).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import DQN, PPO

_LOADERS = {"dqn": DQN, "ppo": PPO}


class SB3Agent:
    def __init__(self, algo: str, model_path: str | Path):
        if algo not in _LOADERS:
            raise ValueError(f"algo không hỗ trợ: {algo!r}")
        self.algo = algo
        self.model = _LOADERS[algo].load(model_path)
        self._last_explain: dict = {}

    def act(self, obs: np.ndarray, info: dict | None = None) -> int:
        action, _ = self.model.predict(obs, deterministic=True)
        self._last_explain = self._compute_explain(obs)
        return int(action)

    def _compute_explain(self, obs: np.ndarray) -> dict:
        obs_tensor = torch.as_tensor(obs, device=self.model.device).float().unsqueeze(0)
        with torch.no_grad():
            if self.algo == "dqn":
                q_values = self.model.q_net(obs_tensor).cpu().numpy()[0]
                return {"q_values": q_values.tolist()}
            distribution = self.model.policy.get_distribution(obs_tensor)
            probs = distribution.distribution.probs.cpu().numpy()[0]
            return {"probs": probs.tolist()}

    def explain(self) -> dict:
        return self._last_explain
