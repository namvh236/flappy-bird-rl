"""Test cho pipeline huấn luyện SB3 và `SB3Agent` (docs/plan/checklist.md P4).

Chỉ chạy số bước rất nhỏ (không train thật) — mục tiêu là xác nhận pipeline
(config → env → model → save/load → predict) không lỗi, không phải đo hiệu
năng agent (việc đó thuộc benchmark G4, chạy riêng bằng `flappy-ai train`).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from flappy_ai.agents.sb3_agent import SB3Agent
from flappy_ai.training.train_sb3 import train


@pytest.fixture
def dqn_config(tmp_path):
    config_path = tmp_path / "dqn.yaml"
    config_path.write_text(
        "total_timesteps: 200\n"
        "eval_freq: 1000\n"
        "n_envs: 1\n"
        "hyperparameters:\n"
        "  learning_rate: 0.0001\n"
        "  buffer_size: 1000\n"
        "  batch_size: 32\n"
        "  learning_starts: 10\n"
        "  policy_kwargs:\n"
        "    net_arch: [8]\n"
    )
    return config_path


def test_train_dqn_produces_loadable_model(tmp_path, dqn_config, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model_dir = train("dqn", dqn_config, seed=0, total_timesteps=None, n_envs=None, device="cpu")
    assert (model_dir / "final_model.zip").exists()
    assert (model_dir.parent.parent / "runs" / model_dir.name / "config.yaml").exists()
    assert (model_dir.parent.parent / "runs" / model_dir.name / "git_commit.txt").exists()

    agent = SB3Agent("dqn", model_dir / "final_model.zip")
    obs = np.zeros(8, dtype=np.float32)
    action = agent.act(obs)
    assert action in (0, 1)
    assert "q_values" in agent.explain()
    assert len(agent.explain()["q_values"]) == 2


def test_sb3_agent_rejects_unknown_algo(tmp_path):
    with pytest.raises(ValueError):
        SB3Agent("unknown", tmp_path / "model.zip")
