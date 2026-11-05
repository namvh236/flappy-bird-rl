"""`evaluate.py`: chạy N episode với 1 agent trên dải seed cố định, xuất
mean/median/max/std điểm số + phân bố `death_cause`, lưu `results.json` +
CSV để so sánh các agent (spec §5.7).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..agents.base import Agent
from ..agents.factory import build_agent
from ..env.flappy_env import DEFAULT_MAX_EPISODE_STEPS, FlappyBirdEnv


@dataclass
class EpisodeResult:
    seed: int
    score: int
    steps: int
    death_cause: str | None


def evaluate_agent(
    agent: Agent, episodes: int, seed_start: int, env: FlappyBirdEnv | None = None
) -> list[EpisodeResult]:
    env = env if env is not None else FlappyBirdEnv()
    results: list[EpisodeResult] = []
    for i in range(episodes):
        seed = seed_start + i
        obs, info = env.reset(seed=seed)
        terminated = truncated = False
        while not (terminated or truncated):
            action = agent.act(obs, info)
            obs, _reward, terminated, truncated, info = env.step(action)
        results.append(
            EpisodeResult(
                seed=seed, score=info["score"], steps=info["frame"], death_cause=info["death_cause"]
            )
        )
    env.close()
    return results


def summarize(results: list[EpisodeResult]) -> dict[str, Any]:
    scores = [r.score for r in results]
    death_causes: dict[str, int] = {}
    for r in results:
        cause = r.death_cause or "none"
        death_causes[cause] = death_causes.get(cause, 0) + 1
    return {
        "episodes": len(results),
        "mean": statistics.mean(scores),
        "median": statistics.median(scores),
        "max": max(scores),
        "std": statistics.pstdev(scores),
        "death_causes": death_causes,
    }


def _write_outputs(
    out_dir: Path, agent_name: str, results: list[EpisodeResult], summary: dict[str, Any]
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{agent_name}_results.json").open("w") as f:
        json.dump({"summary": summary, "episodes": [asdict(r) for r in results]}, f, indent=2)
    with (out_dir / f"{agent_name}_results.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "score", "steps", "death_cause"])
        for r in results:
            writer.writerow([r.seed, r.score, r.steps, r.death_cause])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="flappy-ai eval")
    parser.add_argument(
        "--agent", choices=["random", "heuristic", "dqn", "ppo", "neat"], required=True
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--genome", type=str, default=None)
    parser.add_argument("--neat-config", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-episode-steps", type=int, default=DEFAULT_MAX_EPISODE_STEPS)
    parser.add_argument(
        "--pixel-obs",
        action="store_true",
        help="agent được train trên ảnh: bọc env bằng PixelObs",
    )
    parser.add_argument("--out", type=Path, default=Path("runs/eval"))
    args = parser.parse_args(argv)

    agent_name = args.agent
    env: FlappyBirdEnv = FlappyBirdEnv(
        render_mode="rgb_array" if args.pixel_obs else None,
        max_episode_steps=args.max_episode_steps,
    )
    if args.pixel_obs:
        from ..env.wrappers import PixelObs

        env = PixelObs(env)  # type: ignore[assignment]
        agent_name = f"{args.agent}_pixels"

    agent = build_agent(
        args.agent,
        model_path=args.model,
        genome_path=args.genome,
        neat_config=args.neat_config,
    )
    results = evaluate_agent(agent, args.episodes, args.seed, env=env)
    summary = summarize(results)
    _write_outputs(args.out, agent_name, results, summary)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
