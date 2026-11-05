"""Factory chung để dựng agent từ tên + tham số (play/evaluate/record dùng
chung, tránh 3 bản `_build_agent` chép tay). Import agent nặng (SB3/torch,
neat-python) đặt trong từng nhánh để `flappy-ai play --agent human` không phải
nạp torch."""

from __future__ import annotations

from pathlib import Path

from .base import Agent

FEATURE_AGENTS = ("random", "heuristic", "dqn", "ppo", "neat")


def build_agent(
    name: str,
    model_path: str | Path | None = None,
    genome_path: str | Path | None = None,
    neat_config: str | Path | None = None,
    seed: int | None = None,
) -> Agent:
    if name == "random":
        from .random_agent import RandomAgent

        return RandomAgent(seed=seed if seed is not None else 0)
    if name == "heuristic":
        from .heuristic import HeuristicAgent

        return HeuristicAgent()
    if name in ("dqn", "ppo"):
        if model_path is None:
            raise ValueError(f"--model bắt buộc với --agent {name}")
        from .sb3_agent import SB3Agent

        return SB3Agent(name, model_path)
    if name == "neat":
        if genome_path is None or neat_config is None:
            raise ValueError("--genome và --neat-config bắt buộc với --agent neat")
        from .neat_agent import NeatAgent

        return NeatAgent(genome_path, neat_config)
    raise ValueError(f"agent không hỗ trợ: {name!r}")
