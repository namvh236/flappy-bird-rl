# 🐦 Flappy Bird AI Agent

Bản clone **Flappy Bird** (Python + pygame-ce) đã được refactor thành một **Gymnasium environment** chuẩn để huấn luyện agent AI: PPO, DQN (feature vector + pixel), NEAT, heuristic — kèm công cụ đánh giá, ghi GIF/MP4 và HUD debug.

- Game core headless, tất định theo seed, có test parity với bản gốc.
- Env `FlappyBird-v0` (Gymnasium), observation 8 chiều hoặc ảnh 84×84×4.
- Agent: `random`, `heuristic`, `dqn`, `ppo` (stable-baselines3), `neat` (neat-python).
- **PPO giải hoàn toàn game**: mean/median **220/220** trên 100 episode (chạm trần `max_episode_steps = 20.000` frame).

## 📊 Kết quả benchmark

Mỗi agent chạy 100 episode, dải seed cố định `1000–1099` (không trùng seed train), `max_episode_steps = 20.000` frame ⇒ trần điểm **220 pipe/episode**.

| Agent | Mean | Median | Max | Std | Nhận xét |
|-------|------|--------|-----|-----|----------|
| Random | 0.00 | 0 | 0 | 0.00 | Chết ở trần ngay |
| Heuristic | 220.00 | 220 | 220 | 0.00 | **Chạm trần** mọi episode |
| DQN (features, 1M step) | 0.05 | 0 | 5 | 0.50 | Không học được dù đã thử 4 ablation |
| PPO (features, 1M step, `n_envs=8`) | 220.00 | 220 | 220 | 0.00 | **Giải hoàn toàn game** |
| NEAT (120 generation, 8 ep/genome) | 220.00 | 220 | 220 | 0.00 | Đạt trần từ generation 25 |
| DQN (pixel, CnnPolicy, 2M step) | 0.00 (100 ep) | 0 | 0 | 0.00 | Né được trần/đất nhưng chết ở pipe đầu tiên |
| Headless throughput | **47.000 step/s** (`scripts/benchmark_headless.py`) | | | | G1 (≥ 5.000 step/s) đạt |

Lưu ý về trần điểm: vì heuristic đã chạm trần 220 ở mọi episode, "vượt heuristic" là bất khả thi trong cấu hình mặc định — các agent được so bằng tỉ lệ chết theo pipe và độ ổn định (std) thay vì mean.

## 📥 Cài đặt

Cần Python 3.12 (torch/pygame chưa có wheel ổn định cho 3.13+ trên máy dev này). Dùng [`uv`](https://docs.astral.sh/uv/):

```sh
git clone https://github.com/devk1ngz/flappy-bird.git
cd flappy-bird
uv sync                # tạo .venv + cài đúng version đã khoá trong uv.lock
uv run pytest -q       # 37 test
uv run ruff check .    # lint
```

Hoặc bằng pip thuần:

```sh
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

## 🎮 Chơi tay

```sh
.venv/bin/flappy-ai play --agent human          # game core mới: click/Space để bay
.venv/bin/python -m flappy_ai.game.legacy       # bản gốc, giữ để so sánh/parity test
```

## 👀 Xem agent chơi

```sh
# PPO đã train (model có trong repo? xem mục "Model & phát hành")
.venv/bin/flappy-ai play --agent ppo --model models/ppo_20260927-003700/best_model.zip --debug

# Heuristic (không cần model)
.venv/bin/flappy-ai play --agent heuristic --debug

# NEAT, kèm xem cả quần thể bay cùng lúc
.venv/bin/flappy-ai play --population --birds 12 --genome models/neat_*/best_genome.pkl \
  --neat-config models/neat_*/config.cfg
```

Phím khi xem: `+` / `-` tăng/giảm tốc độ, `P` tạm dừng, `R` chơi lại, `Esc` thoát.
`--debug` bật `DebugOverlay`: đường cảm biến tới khe pipe, action hiện tại, Q-values (DQN) / xác suất flap (PPO) / output (NEAT), episode, best score.

## 🏋️ Huấn luyện

```sh
# PPO — cấu hình tốt nhất (n_envs=8), ~40 phút cho 800k step trên CPU/GPU laptop
.venv/bin/flappy-ai train --algo ppo --config configs/ppo.yaml --seed 0

# DQN baseline (feature vector) — KHÔNG hội tụ dù đã thử 4 ablation (biased exploration, frame_skip, shaped, curriculum)
.venv/bin/flappy-ai train --algo dqn --config configs/dqn.yaml --seed 0

# DQN pixel (CnnPolicy, 84×84×4) — cần GPU, chậm (xem ghi chú bên dưới)
.venv/bin/flappy-ai train --algo dqn --config configs/dqn_pixels.yaml --seed 0 --device cuda

# NEAT (8 input → 1 output, fitness = frames_alive + 100 × pipes)
.venv/bin/flappy-ai train --algo neat --seed 0 --generations 120 --episodes-per-genome 8
```

Mỗi lần train tạo `runs/<algo>_<timestamp>_<pid>/` (config đã dùng + git commit hash + log TensorBoard) và `models/<algo>_.../` (`best_model.zip` / `best_genome.pkl` + `config.cfg`).

## 📏 Đánh giá

```sh
.venv/bin/flappy-ai eval --agent ppo --model models/ppo_20260927-003700/best_model.zip \
  --episodes 100 --seed 1000
```

In ra `mean / median / max / std` + phân bố `death_cause`, đồng thời ghi `runs/eval/<agent>_results.json` + CSV. Thêm `--pixel-obs` nếu model train trên ảnh, `--max-episode-steps N` nếu muốn đổi trần episode.

## 🎥 Ghi GIF/MP4

```sh
.venv/bin/flappy-ai record --agent ppo --model models/ppo_20260927-003700/best_model.zip \
  --out demo.gif --frames 150 --fps 30 --overlay --scale 0.4
```

`--overlay` bật HUD, `--scale` thu nhỏ khung hình (GIF 864×936 rất nặng), đuôi `.mp4` xuất video.

## 🗂 Cấu trúc repo

```
src/flappy_ai/
├── game/        # GameConfig, FlappyGame (headless), entities, legacy.py (bản gốc)
├── render/      # PygameRenderer, DebugOverlay, PopulationRenderer
├── env/         # FlappyBirdEnv (Gymnasium), observation, reward, wrappers (FrameSkip, PixelObs)
├── agents/      # factory + random/heuristic/sb3/neat agent
├── training/    # train_sb3.py, train_neat.py, callbacks
├── evaluation/  # evaluate.py, play.py, record.py
└── cli.py       # flappy-ai play|train|eval|record
configs/         # dqn, ppo, dqn_pixels, neat.cfg + các ablation DQN
scripts/         # benchmark_headless.py
tests/           # core, env, parity, agents, evaluate, training, neat, pixels, record, population
```

## ⚠️ Khác biệt có chủ đích so với bản gốc

1. Sinh pipe theo **số frame** (`pipe_interval_frames = 90`) thay vì `pygame.time.get_ticks()` — tất định, không phụ thuộc tốc độ máy.
2. `reset(seed)` khởi tạo lại **toàn bộ** state (bản gốc quên reset `vel`, `pass_pipe`, ...).
3. Tính điểm theo cờ `passed` của từng cặp pipe (bản gốc có thể tính trùng khi nhiều pipe cùng lúc).
4. Không có pha "click để bắt đầu"; mỗi `step(flap=True)` tương ứng một cú click.
5. Trần episode 20.000 frame ⇒ tối đa 220 pipe/episode.

## 🧪 Test & chất lượng

```sh
uv run pytest -q        # 37 test: physics, collision, scoring, parity, env, agents, training/resume, neat, pixels, record, population
uv run ruff check .     # lint
uv run ruff format .    # format
```

`tests/test_parity.py` replay cùng chuỗi action trên `game/legacy.py` và `FlappyGame`, so quỹ đạo `bird.y` trong 300 frame đầu để đảm bảo refactor không đổi cảm giác chơi.

## 📦 Model

Thư mục `models/` bị `.gitignore`. Muốn có model chạy demo ngay, train lại PPO (~40 phút trên CPU):

```sh
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy .venv/bin/flappy-ai train --algo ppo --config configs/ppo.yaml --seed 1
```

## 📌 Việc còn lại

- **Xếp hạng agent thật sự**: heuristic, PPO và NEAT đều chạm trần 220 pipe/episode nên bằng điểm nhau — muốn tách biệt cần nâng `max_episode_steps` hoặc giảm `pipe_gap`.
- **DQN pixel** (2M step, mean 0) và các ablation DQN không đạt G4; hướng thử tiếp: double DQN, n-step, pre-scale render.

## 📢 Liên hệ

- 📧 Email: vuhoainam2306@gmail.com
- 📂 Github: [devk1ngz](https://github.com/devk1ngz)
- 🔵 Facebook: [Nam Vu](https://facebook.com/imhoainam.yi)
