"""Shared SB3 training callbacks: periodic eval + best-model save, logging
`score`/`pipes_passed` to TensorBoard (SB3 only logs reward/episode length by
default), and checkpointing to resume after a crash.
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
    """Overwrites `checkpoint.zip` every `save_freq` steps instead of SB3's
    `CheckpointCallback`, which keeps every checkpoint (~2.6 GB per replay
    buffer for pixel DQN); writes to a temp file then `os.replace`s it."""

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
