"""Test cho baseline agents (docs/plan/checklist.md P3)."""

from __future__ import annotations

import numpy as np

from flappy_ai.agents.heuristic import HeuristicAgent
from flappy_ai.agents.random_agent import RandomAgent


def test_random_agent_only_returns_valid_actions():
    agent = RandomAgent(flap_prob=0.5, seed=0)
    actions = {agent.act(np.zeros(8)) for _ in range(200)}
    assert actions <= {0, 1}
    assert len(actions) == 2  # 200 lần với p=0.5 chắc chắn thấy cả hai giá trị


def test_random_agent_flap_prob_zero_never_flaps():
    agent = RandomAgent(flap_prob=0.0, seed=0)
    assert all(agent.act(np.zeros(8)) == 0 for _ in range(50))


def test_heuristic_flaps_when_falling_near_gap_bottom():
    agent = HeuristicAgent(margin=20.0)
    obs = np.zeros(8, dtype=np.float32)
    obs[1] = 1.0  # đang rơi (vel > 0)
    obs[4] = 0.0  # sát mép dưới khe
    assert agent.act(obs) == 1


def test_heuristic_does_not_flap_when_far_from_gap_bottom():
    agent = HeuristicAgent(margin=20.0)
    obs = np.zeros(8, dtype=np.float32)
    obs[1] = 1.0
    obs[4] = 0.5  # còn xa mép dưới
    assert agent.act(obs) == 0


def test_heuristic_does_not_flap_while_rising():
    agent = HeuristicAgent(margin=20.0)
    obs = np.zeros(8, dtype=np.float32)
    obs[1] = -1.0  # vừa flap, đang bay lên
    obs[4] = 0.0
    assert agent.act(obs) == 0
