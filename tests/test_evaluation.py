import numpy as np

from evaluation import _ranking_metrics, _aggregate


def test_ranking_metrics_perfect_ranking():
    result = _ranking_metrics([10, 20, 30], {10, 20}, 3)
    assert result["precision"] == 2 / 3
    assert result["recall"] == 1.0
    assert result["map"] == 1.0
    assert result["ndcg"] == 1.0
    assert result["hit_rate"] == 1.0


def test_ranking_metrics_missed_relevant_item():
    result = _ranking_metrics([10, 20, 30], {99}, 3)
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["map"] == 0.0
    assert result["ndcg"] == 0.0
    assert result["hit_rate"] == 0.0


def test_aggregate_averages_user_metrics():
    result = _aggregate([
        {"precision": 1.0, "recall": 1.0, "map": 1.0, "ndcg": 1.0, "hit_rate": 1.0},
        {"precision": 0.0, "recall": 0.0, "map": 0.0, "ndcg": 0.0, "hit_rate": 0.0},
    ])
    assert result.users_evaluated == 2
    assert result.precision == 0.5
    assert result.ndcg == 0.5
