"""Shared agent factory for play/evaluate/record; heavy imports (SB3/torch,
neat-python) stay inside each branch so `flappy-ai play --agent human` skips
loading torch."""

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
            raise ValueError(f"--model is required for --agent {name}")
        from .sb3_agent import SB3Agent

        return SB3Agent(name, model_path)
    if name == "neat":
        if genome_path is None or neat_config is None:
            raise ValueError("--genome and --neat-config are required for --agent neat")
        from .neat_agent import NeatAgent

        return NeatAgent(genome_path, neat_config)
    raise ValueError(f"unsupported agent: {name!r}")
