"""Tests for baseline agents (checklist P3)."""

from __future__ import annotations

import numpy as np

from flappy_ai.agents.heuristic import HeuristicAgent
from flappy_ai.agents.random_agent import RandomAgent


def test_random_agent_only_returns_valid_actions():
    agent = RandomAgent(flap_prob=0.5, seed=0)
    actions = {agent.act(np.zeros(8)) for _ in range(200)}
    assert actions <= {0, 1}
    assert len(actions) == 2  # 200 draws at p=0.5 reliably see both values


def test_random_agent_flap_prob_zero_never_flaps():
    agent = RandomAgent(flap_prob=0.0, seed=0)
    assert all(agent.act(np.zeros(8)) == 0 for _ in range(50))


def test_heuristic_flaps_when_falling_near_gap_bottom():
    agent = HeuristicAgent(margin=20.0)
    obs = np.zeros(8, dtype=np.float32)
    obs[1] = 1.0  # falling (vel > 0)
    obs[4] = 0.0  # near bottom edge of gap
    assert agent.act(obs) == 1


def test_heuristic_does_not_flap_when_far_from_gap_bottom():
    agent = HeuristicAgent(margin=20.0)
    obs = np.zeros(8, dtype=np.float32)
    obs[1] = 1.0
    obs[4] = 0.5  # still far from bottom edge
    assert agent.act(obs) == 0


def test_heuristic_does_not_flap_while_rising():
    agent = HeuristicAgent(margin=20.0)
    obs = np.zeros(8, dtype=np.float32)
    obs[1] = -1.0  # just flapped, rising
    obs[4] = 0.0
    assert agent.act(obs) == 0
