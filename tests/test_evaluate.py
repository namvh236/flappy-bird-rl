"""Tests for `evaluate.py` (checklist P3)."""

from __future__ import annotations

from dataclasses import asdict

from flappy_ai.evaluation.evaluate import EpisodeResult, evaluate_agent, summarize


class _NeverFlapAgent:
    def act(self, obs, info=None):
        return 0

    def explain(self):
        return {}


def test_evaluate_agent_runs_full_episodes_and_reports_death_cause():
    results = evaluate_agent(_NeverFlapAgent(), episodes=3, seed_start=0)
    assert len(results) == 3
    assert all(r.death_cause in ("pipe", "ceiling", "ground") for r in results)
    assert all(r.steps > 0 for r in results)


def test_summarize_computes_mean_median_max_std_and_death_causes():
    results = [
        EpisodeResult(seed=0, score=10, steps=100, death_cause="pipe"),
        EpisodeResult(seed=1, score=20, steps=200, death_cause="ground"),
        EpisodeResult(seed=2, score=30, steps=300, death_cause="pipe"),
    ]
    summary = summarize(results)
    assert summary["episodes"] == 3
    assert summary["mean"] == 20
    assert summary["median"] == 20
    assert summary["max"] == 30
    assert summary["death_causes"] == {"pipe": 2, "ground": 1}


def test_same_seed_and_config_reproduce_eval_results():
    from flappy_ai.agents.heuristic import HeuristicAgent

    first = evaluate_agent(HeuristicAgent(), episodes=5, seed_start=1000)
    second = evaluate_agent(HeuristicAgent(), episodes=5, seed_start=1000)
    assert [asdict(r) for r in first] == [asdict(r) for r in second]
