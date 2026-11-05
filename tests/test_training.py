"""Smoke test for the SB3 training pipeline and `SB3Agent` (checklist P4);
runs very few steps (no real training) to confirm config -> env -> model ->
save/load -> predict works. Performance is measured separately via
`flappy-ai train` benchmarks."""

from __future__ import annotations

import os
import pickle
import shutil

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


def test_resume_continues_timesteps_and_replay_buffer(tmp_path, dqn_config, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dqn_config.write_text(dqn_config.read_text() + "checkpoint_freq: 100\n")
    first_dir = train("dqn", dqn_config, seed=0, total_timesteps=200, n_envs=None, device="cpu")
    ckpt_dir = tmp_path / "ckpt"
    shutil.copytree(first_dir, ckpt_dir)

    resumed_dir = train(
        "dqn",
        dqn_config,
        seed=0,
        total_timesteps=300,
        n_envs=None,
        device="cpu",
        resume=ckpt_dir / "checkpoint.zip",
    )

    model = SB3Agent("dqn", resumed_dir / "final_model.zip").model
    assert model.num_timesteps == 300
    with (resumed_dir / "checkpoint_replay_buffer.pkl").open("rb") as f:
        # Resume itself only adds <=100 transitions; more means the old buffer loaded.
        assert pickle.load(f).pos > 100
