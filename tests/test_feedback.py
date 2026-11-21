import json

import numpy as np
import pandas as pd
import pytest

from feedback import apply_feedback, get_user_feedback, get_watchlist, record_feedback


def test_feedback_events_keep_latest_state(tmp_path):
    path = tmp_path / "feedback.json"
    record_feedback("jawad", 10, "Film A", "like", path)
    record_feedback("jawad", 10, "Film A", "dislike", path)
    record_feedback("jawad", 11, "Film B", "watchlist", path)

    feedback = get_user_feedback("jawad", path)
    assert {item["movie_id"]: item["action"] for item in feedback} == {
        10: "dislike",
        11: "watchlist",
    }
    assert len(get_watchlist(feedback)) == 1
    assert json.loads(path.read_text(encoding="utf-8"))["jawad"][-1]["action"] == "watchlist"


def test_feedback_isolated_between_profiles(tmp_path):
    path = tmp_path / "feedback.json"
    record_feedback("profile-a", 10, "Film A", "like", path)
    assert get_user_feedback("profile-b", path) == []


def test_invalid_action_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        record_feedback("jawad", 10, "Film A", "skip", tmp_path / "feedback.json")


def test_apply_feedback_hides_disliked_and_watchlist_and_reranks_likes():
    embeddings = np.array([
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
        [-1.0, 0.0],
    ], dtype=np.float32)
    results = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "similarity": [0.95, 0.80, 0.90, 0.85],
    }, index=[0, 1, 2, 3])
    feedback = [
        {"movie_id": 1, "action": "like"},
        {"movie_id": 3, "action": "dislike"},
        {"movie_id": 4, "action": "watchlist"},
    ]

    ranked = apply_feedback(results, embeddings, feedback, top_k=10)
    assert 3 not in ranked["id"].tolist()
    assert 4 not in ranked["id"].tolist()
    assert ranked.iloc[0]["id"] == 2
    assert "personalized_score" in ranked.columns
