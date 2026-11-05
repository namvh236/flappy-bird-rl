"""Callback dùng chung cho huấn luyện SB3 (spec §5.7): eval định kỳ + lưu
best model, log `score`/`pipes_passed` lên TensorBoard (SB3 mặc định chỉ log
reward/episode length, không biết về các field tuỳ ý trong `info`), checkpoint
để resume sau crash.
"""

from __future__ import annotations

import os
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


class LatestCheckpointCallback(BaseCallback):
    """Mỗi `save_freq` env step ghi đè `checkpoint.zip` (+ replay buffer nếu có).

    Không dùng `CheckpointCallback` của SB3 vì nó giữ mọi bản: với DQN pixel mỗi
    replay buffer ~2,6 GB. Ghi file tạm rồi `os.replace` để crash giữa lúc ghi
    không làm hỏng checkpoint trước.
    """

    def __init__(self, model_dir: Path, save_freq: int, verbose: int = 0):
        super().__init__(verbose)
        self.model_path = model_dir / "checkpoint.zip"
        self.buffer_path = replay_buffer_path(self.model_path)
        self.save_freq = save_freq
        self._last_save = 0

    def _on_training_start(self) -> None:
        self._last_save = self.num_timesteps

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_save < self.save_freq:
            return True
        self._last_save = self.num_timesteps
        if getattr(self.model, "replay_buffer", None) is not None:
            tmp_buffer = self.buffer_path.with_suffix(".tmp.pkl")
            self.model.save_replay_buffer(tmp_buffer)
            os.replace(tmp_buffer, self.buffer_path)
        tmp_model = self.model_path.with_suffix(".tmp.zip")
        self.model.save(tmp_model)
        os.replace(tmp_model, self.model_path)
        if self.verbose:
            print(f"Checkpoint @ {self.num_timesteps} step -> {self.model_path}")
        return True


def replay_buffer_path(model_path: Path) -> Path:
    return model_path.with_name(f"{model_path.stem}_replay_buffer.pkl")


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
