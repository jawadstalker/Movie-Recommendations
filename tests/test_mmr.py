import numpy as np
import pandas as pd
import pytest

from mmr import select_diverse_recommendations


def test_mmr_reduces_near_duplicate_recommendations():
    results = pd.DataFrame({"id": [10, 11, 12], "taste_profile_score": [0.96, 0.91, 0.82]}, index=[0, 1, 2])
    embeddings = np.array([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]], dtype=np.float32)
    ranked = select_diverse_recommendations(results, embeddings, top_k=3, diversity_weight=0.35)
    assert ranked.iloc[0]["id"] == 10
    assert ranked.iloc[1]["id"] == 12
    assert "mmr_score" in ranked.columns
    assert "diversity_score" in ranked.columns


def test_zero_diversity_weight_is_relevance_only():
    results = pd.DataFrame({"id": [1, 2, 3], "similarity": [0.2, 0.9, 0.5]}, index=[0, 1, 2])
    ranked = select_diverse_recommendations(results, np.eye(3, dtype=np.float32), top_k=2, diversity_weight=0)
    assert ranked["id"].tolist() == [2, 3]


def test_empty_results_are_supported():
    results = pd.DataFrame(columns=["id", "similarity"])
    ranked = select_diverse_recommendations(results, np.zeros((0, 3)), top_k=5)
    assert ranked.empty and "mmr_score" in ranked.columns


@pytest.mark.parametrize("weight", [-0.1, 1.1])
def test_invalid_diversity_weight_is_rejected(weight):
    results = pd.DataFrame({"similarity": [0.5]}, index=[0])
    with pytest.raises(ValueError):
        select_diverse_recommendations(results, np.ones((1, 2)), diversity_weight=weight)


def test_invalid_embedding_index_is_rejected():
    results = pd.DataFrame({"similarity": [0.5]}, index=[4])
    with pytest.raises(ValueError):
        select_diverse_recommendations(results, np.ones((2, 2)))
