"""Measures headless `FlappyBirdEnv` throughput (G1: >= 5,000 step/s).

Run: `.venv/bin/python scripts/benchmark_headless.py [--steps 20000]`
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from flappy_ai.env.flappy_env import FlappyBirdEnv


def benchmark(steps: int, seed: int = 0) -> float:
    env = FlappyBirdEnv()
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    for i in range(steps):
        _, _, terminated, truncated, _ = env.step(int(rng.integers(0, 2)))
        if terminated or truncated:
            env.reset(seed=seed + i)
    return steps / (time.perf_counter() - start)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rate = benchmark(args.steps, args.seed)
    print(f"{rate:,.0f} step/s ({args.steps} steps, random actions, headless)")
    print("G1 (>= 5.000 step/s):", "PASS" if rate >= 5_000 else "FAIL")


if __name__ == "__main__":
    main()
