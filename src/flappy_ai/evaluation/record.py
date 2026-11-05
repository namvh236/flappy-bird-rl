"""`record.py`: chơi một agent trên `FlappyBirdEnv(render_mode="rgb_array")` và
xuất kết quả ra GIF/MP4 (spec §5.7). Dùng chung `build_agent` với `play.py`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio
import numpy as np

from ..agents.factory import FEATURE_AGENTS, build_agent
from ..env.flappy_env import FlappyBirdEnv


def record(
    agent_name: str,
    model_path: str | Path | None = None,
    genome_path: str | Path | None = None,
    neat_config: str | Path | None = None,
    out: Path = Path("docs/media/demo.gif"),
    seed: int | None = 0,
    max_frames: int | None = 3600,
    overlay: bool = False,
    fps: int = 60,
    scale: float = 1.0,
    pixel_obs: bool = False,
    env_kwargs: dict | None = None,
) -> Path:
    agent = build_agent(
        agent_name,
        model_path=model_path,
        genome_path=genome_path,
        neat_config=neat_config,
        seed=seed,
    )
    env: FlappyBirdEnv = FlappyBirdEnv(
        render_mode="rgb_array", debug_overlay=overlay, **(env_kwargs or {})
    )
    if pixel_obs:
        from ..env.wrappers import PixelObs

        env = PixelObs(env)  # type: ignore[assignment]
    obs, info = env.reset(seed=seed)

    frames = []
    episode = 1
    best_score = 0
    limit = max_frames if max_frames is not None else float("inf")

    while len(frames) < limit:
        action = agent.act(obs, info)
        env.unwrapped.hud = {
            "action": action,
            "explain": agent.explain(),
            "episode": episode,
            "best_score": best_score,
            "fps": float(fps),
        }
        frames.append(env.render())
        obs, _reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            best_score = max(best_score, info["score"])
            episode += 1
            obs, info = env.reset(seed=seed)

    env.close()

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if scale != 1.0:
        from PIL import Image

        frames = [
            np.asarray(
                Image.fromarray(frame).resize(
                    (int(frame.shape[1] * scale), int(frame.shape[0] * scale)),
                    Image.BILINEAR,
                )
            )
            for frame in frames
        ]
    if out.suffix == ".gif":
        imageio.mimsave(out, frames, duration=1000 / fps)
    elif out.suffix == ".mp4":
        with imageio.get_writer(out, fps=fps, codec="libx264") as writer:
            for frame in frames:
                writer.append_data(frame)
    else:
        raise ValueError(f"đuôi file không hỗ trợ: {out.suffix!r}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="flappy-ai record")
    parser.add_argument("--agent", choices=FEATURE_AGENTS, required=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--genome", type=str, default=None)
    parser.add_argument("--neat-config", type=str, default=None)
    parser.add_argument("--out", type=Path, default=Path("docs/media/demo.gif"))
    parser.add_argument("--frames", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--overlay", action="store_true")
    parser.add_argument("--scale", type=float, default=1.0, help="thu nhỏ khung hình, vd 0.5")
    parser.add_argument("--pixel-obs", action="store_true", help="model train trên ảnh")
    args = parser.parse_args(argv)

    out = record(
        args.agent,
        model_path=args.model,
        genome_path=args.genome,
        neat_config=args.neat_config,
        out=args.out,
        seed=args.seed,
        max_frames=args.frames,
        overlay=args.overlay,
        fps=args.fps,
        scale=args.scale,
        pixel_obs=args.pixel_obs,
    )
    print(f"Saved recording to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
