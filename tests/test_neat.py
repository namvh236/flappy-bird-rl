"""Test smoke cho pipeline huấn luyện NEAT (docs/plan/checklist.md P5).

Chỉ chạy population rất nhỏ (pop 8, 1 generation) — mục tiêu xác nhận pipeline
(config → population → eval → save/load → predict) không lỗi, không phải đo
hiệu năng agent.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from flappy_ai.agents.neat_agent import NeatAgent
from flappy_ai.training.train_neat import DEFAULT_NEAT_CONFIG, train_neat

CONFIG_PATH = Path(__file__).resolve().parent.parent / DEFAULT_NEAT_CONFIG


def test_train_neat_produces_loadable_genome(tmp_path):
    model_dir = train_neat(
        CONFIG_PATH,
        seed=0,
        generations=1,
        episodes_per_genome=2,
        pop_size=8,
        out_root=tmp_path,
    )
    run_dir = tmp_path / "runs" / model_dir.name

    assert (model_dir / "best_genome.pkl").exists()
    assert (model_dir / "config.cfg").exists()
    assert (run_dir / "results.json").exists()

    results = json.loads((run_dir / "results.json").read_text())
    assert len(results) == 1
    assert "best_fitness" in results[0]
    assert "mean_fitness" in results[0]

    agent = NeatAgent(model_dir / "best_genome.pkl", model_dir / "config.cfg")
    rng = np.random.default_rng(0)
    for _ in range(20):
        obs = rng.uniform(-1, 1, size=8).astype(np.float32)
        action = agent.act(obs)
        assert action in (0, 1)
