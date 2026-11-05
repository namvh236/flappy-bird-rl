"""CLI entry point cho `flappy-ai` (spec §5.8).

`play --agent human` chơi trực tiếp trên `FlappyGame` + `PygameRenderer`.
`play --agent random|heuristic|dqn|ppo|neat` (kèm `--debug` cho HUD), `eval`,
`train` (SB3 hoặc NEAT), `record` (GIF/MP4) chạy qua `FlappyBirdEnv` + `Agent`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

AGENTS = ["random", "heuristic", "dqn", "ppo", "neat"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="flappy-ai")
    sub = parser.add_subparsers(dest="command", required=True)

    play = sub.add_parser("play", help="Xem một agent chơi")
    play.add_argument("--agent", choices=["human", *AGENTS], default="human")
    play.add_argument("--model", type=str, default=None)
    play.add_argument("--genome", type=str, default=None)
    play.add_argument("--neat-config", type=str, default=None)
    play.add_argument("--fps", type=int, default=60)
    play.add_argument("--seed", type=int, default=None)
    play.add_argument("--debug", action="store_true", help="HUD: cảm biến, action, Q-values")
    play.add_argument("--pixel-obs", action="store_true", help="model train trên ảnh")
    play.add_argument("--population", action="store_true", help="xem cả quần thể NEAT bay cùng lúc")
    play.add_argument("--birds", type=int, default=10, help="số bird khi --population")

    eval_cmd = sub.add_parser("eval", help="Đánh giá agent trên N episode")
    eval_cmd.add_argument("--agent", choices=AGENTS, required=True)
    eval_cmd.add_argument("--model", type=str, default=None)
    eval_cmd.add_argument("--genome", type=str, default=None)
    eval_cmd.add_argument("--neat-config", type=str, default=None)
    eval_cmd.add_argument("--episodes", type=int, default=100)
    eval_cmd.add_argument("--seed", type=int, default=1000)
    eval_cmd.add_argument("--max-episode-steps", type=int, default=None)
    eval_cmd.add_argument("--pixel-obs", action="store_true")

    train_cmd = sub.add_parser("train", help="Huấn luyện: SB3 (DQN/PPO) hoặc NEAT")
    train_cmd.add_argument("--algo", choices=["dqn", "ppo", "neat"], required=True)
    train_cmd.add_argument("--config", type=str, default=None)
    train_cmd.add_argument("--seed", type=int, default=0)
    train_cmd.add_argument("--total-timesteps", type=int, default=None)
    train_cmd.add_argument("--generations", type=int, default=None, help="chỉ dùng cho NEAT")
    train_cmd.add_argument(
        "--episodes-per-genome", type=int, default=None, help="chỉ dùng cho NEAT"
    )
    train_cmd.add_argument("--pop-size", type=int, default=None, help="chỉ dùng cho NEAT")
    train_cmd.add_argument("--n-envs", type=int, default=None)
    train_cmd.add_argument("--device", default="auto")

    record_cmd = sub.add_parser("record", help="Ghi GIF/MP4 một agent chơi")
    record_cmd.add_argument("--agent", choices=AGENTS, default="heuristic")
    record_cmd.add_argument("--model", type=str, default=None)
    record_cmd.add_argument("--genome", type=str, default=None)
    record_cmd.add_argument("--neat-config", type=str, default=None)
    record_cmd.add_argument("--out", type=Path, default=Path("docs/media/demo.gif"))
    record_cmd.add_argument("--frames", type=int, default=3600)
    record_cmd.add_argument("--seed", type=int, default=0)
    record_cmd.add_argument("--fps", type=int, default=60)
    record_cmd.add_argument("--overlay", action="store_true")
    record_cmd.add_argument("--pixel-obs", action="store_true")
    record_cmd.add_argument("--scale", type=float, default=1.0)

    args = parser.parse_args(argv)

    if args.command == "play":
        if args.population:
            from flappy_ai.render.population import run_population_view

            run_population_view(
                genome_path=args.genome,
                config_path=args.neat_config,
                n=args.birds,
                seed=args.seed or 0,
                fps=args.fps,
            )
            return 0
        if args.agent == "human":
            from flappy_ai.game.play_human import run_human_play

            run_human_play(fps=args.fps, seed=args.seed)
        else:
            from flappy_ai.evaluation.play import run_agent_play

            run_agent_play(
                args.agent,
                fps=args.fps,
                seed=args.seed,
                model_path=args.model,
                genome_path=args.genome,
                neat_config=args.neat_config,
                debug=args.debug,
                pixel_obs=args.pixel_obs,
            )
        return 0

    if args.command == "eval":
        from flappy_ai.evaluation.evaluate import main as eval_main

        eval_argv = [
            "--agent",
            args.agent,
            "--episodes",
            str(args.episodes),
            "--seed",
            str(args.seed),
        ]
        if args.model:
            eval_argv += ["--model", args.model]
        if args.genome:
            eval_argv += ["--genome", args.genome]
        if args.neat_config:
            eval_argv += ["--neat-config", args.neat_config]
        if args.max_episode_steps:
            eval_argv += ["--max-episode-steps", str(args.max_episode_steps)]
        if args.pixel_obs:
            eval_argv += ["--pixel-obs"]
        return eval_main(eval_argv)

    if args.command == "train":
        if args.algo == "neat":
            from flappy_ai.training.train_neat import DEFAULT_NEAT_CONFIG, train_neat

            config_path = Path(args.config) if args.config else DEFAULT_NEAT_CONFIG
            model_dir = train_neat(
                config_path,
                seed=args.seed,
                generations=args.generations,
                episodes_per_genome=args.episodes_per_genome,
                pop_size=args.pop_size,
            )
            print(f"Saved best genome to {model_dir}")
            return 0

        from flappy_ai.training.train_sb3 import DEFAULT_CONFIGS, train

        config_path = Path(args.config) if args.config else DEFAULT_CONFIGS[args.algo]
        model_dir = train(
            args.algo, config_path, args.seed, args.total_timesteps, args.n_envs, args.device
        )
        print(f"Saved model to {model_dir}")
        return 0

    if args.command == "record":
        from flappy_ai.evaluation.record import record

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

    parser.error(f"lệnh chưa triển khai: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
