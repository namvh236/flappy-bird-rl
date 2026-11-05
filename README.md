# flappy-bird-rl

Flappy Bird (Python + pygame-ce) as a Gymnasium environment for learning reinforcement learning. Includes PPO, DQN and NEAT agents, a heuristic baseline, evaluation tools, GIF/MP4 recording and a debug HUD.

- Headless, deterministic game core with a parity test against the original game.
- `FlappyBird-v0` env with an 8-dim feature observation or 84x84x4 stacked frames.
- Agents: `random`, `heuristic`, `dqn`, `ppo` (stable-baselines3), `neat` (neat-python).

![PPO agent playing Flappy Bird](media/ppo_demo.gif)

## Results

100 episodes per agent, seeds 1000-1099 (not used in training). Episodes are capped at 20,000 frames, so the maximum score is 220 pipes.

| Agent | Mean | Median | Max | Notes |
|-------|------|--------|-----|-------|
| Random | 0 | 0 | 0 | Dies on the ceiling |
| Heuristic | 220 | 220 | 220 | Hits the cap |
| PPO (features, 1M steps) | 220 | 220 | 220 | Solves the game |
| NEAT (120 generations) | 220 | 220 | 220 | Hits the cap from generation 25 |
| DQN (features, 1M steps) | 0.05 | 0 | 5 | Fails despite 4 ablations |
| DQN (pixels, 2M steps) | 0 | 0 | 0 | Dies at the first pipe |

Headless throughput is about 47,000 steps/s (`scripts/benchmark_headless.py`).

## Install

Requires Python 3.12 and [`uv`](https://docs.astral.sh/uv/):

```sh
git clone https://github.com/namvh236/flappy-bird-rl.git
cd flappy-bird-rl
uv sync
uv run pytest -q
```

Without uv: `python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"`.

## Usage

```sh
# Play yourself (click or Space to flap)
.venv/bin/flappy-ai play --agent human

# Watch the pretrained PPO with the debug HUD
.venv/bin/flappy-ai play --agent ppo --model pretrained/ppo_best.zip --debug

# Evaluate
.venv/bin/flappy-ai eval --agent ppo --model pretrained/ppo_best.zip --episodes 100 --seed 1000

# Record a GIF
.venv/bin/flappy-ai record --agent ppo --model pretrained/ppo_best.zip \
  --out demo.gif --frames 900 --fps 50 --overlay --scale 0.5
```

Viewer keys: `+`/`-` speed, `P` pause, `R` restart, `Esc` quit.

## Training

```sh
.venv/bin/flappy-ai train --algo ppo --config configs/ppo.yaml --seed 0
.venv/bin/flappy-ai train --algo dqn --config configs/dqn.yaml --seed 0
.venv/bin/flappy-ai train --algo dqn --config configs/dqn_pixels.yaml --seed 0 --device cuda
.venv/bin/flappy-ai train --algo neat --seed 0 --generations 120 --episodes-per-genome 8
```

Each run writes `runs/<algo>_<timestamp>_<pid>/` (config, git commit, TensorBoard logs) and `models/<algo>_.../` (best model). Both folders are gitignored. SB3 runs can continue with `--resume models/<run>/checkpoint.zip`.

On a machine without a display, set `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy`.

## Layout

```
src/flappy_ai/
  game/        game core (headless) and the original game (legacy.py)
  render/      pygame renderer, debug HUD, population view
  env/         Gymnasium env, observation, reward, wrappers
  agents/      random, heuristic, SB3, NEAT agents
  training/    SB3 and NEAT training
  evaluation/  eval, play, record
  cli.py       flappy-ai play|train|eval|record
configs/       training configs
pretrained/    ppo_best.zip
media/         demo GIF
tests/
```

## Differences from the original game

- Pipes spawn every 90 frames instead of by wall-clock time, so runs are deterministic.
- `reset(seed)` resets all state.
- Score is counted once per pipe pair.
- No "click to start" screen.
