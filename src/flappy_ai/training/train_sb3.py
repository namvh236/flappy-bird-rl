"""`train_sb3.py`: huấn luyện DQN/PPO trên `FlappyBird-v0` qua
`stable-baselines3` (spec §5.7). Mỗi lần chạy tạo `runs/<algo>_<timestamp>/`
chứa `config.yaml` + `git_commit.txt` + log TensorBoard;
`models/<algo>_<timestamp>/best_model.zip` được `EvalCallback` lưu định kỳ.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env

import flappy_ai.env  # noqa: F401 (đăng ký "FlappyBird-v0")

from ..env.wrappers import FrameSkip, PixelObs
from ..game.config import GameConfig
from .callbacks import (
    LatestCheckpointCallback,
    ScoreLoggingCallback,
    build_eval_callback,
    replay_buffer_path,
)

ALGOS = {"dqn": DQN, "ppo": PPO}
DEFAULT_CONFIGS = {"dqn": Path("configs/dqn.yaml"), "ppo": Path("configs/ppo.yaml")}


class BiasedExplorationDQN(DQN):
    """DQN với exploration ngẫu nhiên lệch về phía flap (spec §5.6, §9): với
    action space nhị phân {không làm gì, flap}, epsilon-greedy đều 50/50 mặc
    định của SB3 khiến bird đâm trần gần như ngay lập tức trong toàn bộ pha
    warmup/exploration — không bao giờ sống đủ lâu để thấy pipe (đã quan sát
    thực nghiệm: DQN mặc định plateau ở mean score ~0 sau 1M step, xem
    `docs/plan/experiments.md`). Ghi đè phần lấy hành động ngẫu nhiên trong
    `DQN.predict()` (nguồn duy nhất sinh random action sau warmup) để chỉ
    flap với xác suất `flap_bias` thay vì 50%.
    """

    def __init__(self, *args, flap_bias: float = 0.1, **kwargs):
        self.flap_bias = flap_bias
        super().__init__(*args, **kwargs)

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        if deterministic or np.random.rand() >= self.exploration_rate:
            return super().predict(observation, state, episode_start, deterministic)

        if self.policy.is_vectorized_observation(observation):
            if isinstance(observation, dict):
                n_batch = observation[next(iter(observation.keys()))].shape[0]
            else:
                n_batch = observation.shape[0]
            action = np.array(
                [1 if np.random.rand() < self.flap_bias else 0 for _ in range(n_batch)]
            )
        else:
            action = np.array(1 if np.random.rand() < self.flap_bias else 0)
        return action, state


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _make_env_kwargs(config: dict) -> dict:
    env_kwargs: dict = {}
    if "reward_mode" in config:
        env_kwargs["reward_mode"] = config["reward_mode"]
    if "game_config" in config:
        env_kwargs["config"] = GameConfig(**config["game_config"])
    return env_kwargs


def train(
    algo: str,
    config_path: Path,
    seed: int,
    total_timesteps: int | None,
    n_envs: int | None,
    device: str,
    resume: Path | None = None,
) -> Path:
    algo_cls = ALGOS[algo]
    config = _load_config(config_path)
    flap_bias = config.get("flap_bias")
    if algo == "dqn" and flap_bias is not None:
        algo_cls = BiasedExplorationDQN
    hyperparams = dict(config.get("hyperparameters", {}))
    timesteps = total_timesteps or config.get("total_timesteps", 100_000)
    envs = n_envs or config.get("n_envs", 1)
    policy = config.get("policy", "MlpPolicy")
    pixel_obs = config.get("pixel_obs", False)
    env_kwargs = _make_env_kwargs(config)

    frame_skip = config.get("frame_skip", 1)
    wrapper_kwargs = None
    wrapper_class = None
    if pixel_obs:
        env_kwargs["render_mode"] = "rgb_array"
        wrapper_class = PixelObs
    elif frame_skip > 1:
        wrapper_class = FrameSkip
        wrapper_kwargs = {"skip": frame_skip}

    # Pixel obs bị nghẽn ở khâu render (DummyVecEnv chạy tuần tự ~56 step/s);
    # SubprocVecEnv đẩy phần render sang nhiều process.
    vec_env_cls = None
    vec_env_kwargs = None
    if config.get("vec_env") == "subproc" and envs > 1:
        from stable_baselines3.common.vec_env import SubprocVecEnv

        vec_env_cls = SubprocVecEnv
        vec_env_kwargs = {"start_method": config.get("vec_env_start_method", "forkserver")}

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{algo}_{timestamp}_{os.getpid()}"
    run_dir = Path("runs") / run_name
    model_dir = Path("models") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "config.yaml").open("w") as f:
        yaml.safe_dump({**config, "resumed_from": str(resume)} if resume else config, f)
    (run_dir / "git_commit.txt").write_text(_git_commit_hash() + "\n")

    env = make_vec_env(
        "FlappyBird-v0",
        n_envs=envs,
        seed=seed,
        env_kwargs=env_kwargs,
        wrapper_class=wrapper_class,
        wrapper_kwargs=wrapper_kwargs,
        vec_env_cls=vec_env_cls,
        vec_env_kwargs=vec_env_kwargs,
    )
    eval_env = make_vec_env(
        "FlappyBird-v0",
        n_envs=1,
        seed=seed + 1000,
        env_kwargs=env_kwargs,
        wrapper_class=wrapper_class if pixel_obs else None,
    )

    model_kwargs = dict(hyperparams)
    if algo_cls is BiasedExplorationDQN:
        model_kwargs["flap_bias"] = flap_bias
    if resume:
        # kwargs ghi đè hyperparams đã lưu trước `_setup_model()`, nên thay đổi config
        # (vd. `optimize_memory_usage`) áp dụng được lên model cũ.
        model = algo_cls.load(
            resume, env=env, device=device, tensorboard_log=str(run_dir), **model_kwargs
        )
        buffer_path = replay_buffer_path(resume)
        if hasattr(model, "replay_buffer"):
            if buffer_path.exists():
                model.load_replay_buffer(buffer_path)
            else:
                # Buffer rỗng: thu thập lại `learning_starts` step trước khi train,
                # không thì gradient đầu tiên chỉ lấy mẫu từ vài transition.
                model.learning_starts += model.num_timesteps
    else:
        model = algo_cls(
            policy,
            env,
            seed=seed,
            device=device,
            tensorboard_log=str(run_dir),
            **model_kwargs,
        )

    callbacks = [
        ScoreLoggingCallback(),
        build_eval_callback(eval_env, model_dir, eval_freq=config.get("eval_freq", 25_000)),
    ]
    if "checkpoint_freq" in config:
        callbacks.append(LatestCheckpointCallback(model_dir, config["checkpoint_freq"]))

    model.learn(
        total_timesteps=timesteps - model.num_timesteps,
        tb_log_name="tb",
        callback=callbacks,
        reset_num_timesteps=not resume,
    )
    model.save(model_dir / "final_model.zip")
    env.close()
    eval_env.close()
    return model_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="flappy-ai train")
    parser.add_argument("--algo", choices=list(ALGOS), required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", type=Path, default=None, help="model .zip để train tiếp")
    args = parser.parse_args(argv)

    config_path = args.config or DEFAULT_CONFIGS[args.algo]
    model_dir = train(
        args.algo,
        config_path,
        args.seed,
        args.total_timesteps,
        args.n_envs,
        args.device,
        args.resume,
    )
    print(f"Saved model to {model_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
