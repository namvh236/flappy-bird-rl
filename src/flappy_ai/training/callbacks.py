"""Callback dùng chung cho huấn luyện SB3 (spec §5.7): eval định kỳ + lưu
best model, log `score`/`pipes_passed` lên TensorBoard (SB3 mặc định chỉ log
reward/episode length, không biết về các field tuỳ ý trong `info`).
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import VecEnv


class ScoreLoggingCallback(BaseCallback):
    def __init__(self, window: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self._scores: deque[int] = deque(maxlen=window)
        self._pipes: deque[int] = deque(maxlen=window)

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"], strict=True):
            if done:
                self._scores.append(info["score"])
                self._pipes.append(info["pipes_passed"])
        if self._scores:
            self.logger.record("rollout/score_mean", float(np.mean(self._scores)))
            self.logger.record("rollout/pipes_passed_mean", float(np.mean(self._pipes)))
        return True


def build_eval_callback(
    eval_env: VecEnv, model_dir: Path, eval_freq: int, n_eval_episodes: int = 20
) -> EvalCallback:
    return EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(model_dir),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
    )
