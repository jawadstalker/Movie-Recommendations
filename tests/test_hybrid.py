import numpy as np
import pandas as pd

from hybrid import build_collaborative_model, collaborative_scores, recommend_hybrid


class FakeModel:
    def encode(self, texts, **kwargs):
        return np.array([[1.0, 0.0]], dtype=np.float32)


def test_collaborative_model_scores_unseen_similar_movies():
    movies = pd.DataFrame({"id": [101, 102, 103]})
    ratings = pd.DataFrame({
        "userId": [1, 1, 2, 2, 3, 3],
        "movieId": [11, 12, 11, 12, 11, 13],
        "rating": [5.0, 4.5, 4.0, 4.0, 5.0, 4.5],
    })
    links = pd.DataFrame({"movieId": [11, 12, 13], "tmdbId": [101, 102, 103]})
    model = build_collaborative_model(ratings, links, movies, top_neighbors=2)
    scores = collaborative_scores(1, model)
    assert 103 in scores
    assert 102 not in scores


def test_hybrid_uses_content_fallback_without_collaborative_data():
    movies = pd.DataFrame({
        "id": [101, 102], "title": ["A", "B"], "overview": ["one", "two"],
        "release_date": ["2001-01-01", "2002-01-01"],
        "genres": ['[{"name":"Drama"}]', '[{"name":"Comedy"}]'],
    })
    result = recommend_hybrid(
        "a story", movies, np.array([[1.0, 0.0], [0.0, 1.0]]),
        FakeModel(), user_id=999, collaborative_model=None, top_k=2,
    )
    assert result["recommendation_type"].eq("content-based").all()


def test_hybrid_returns_combined_score_for_known_user():
    movies = pd.DataFrame({
        "id": [101, 102, 103], "title": ["A", "B", "C"],
        "overview": ["one", "two", "three"],
        "release_date": ["2001-01-01", "2002-01-01", "2003-01-01"],
        "genres": ['[{"name":"Drama"}]', '[{"name":"Comedy"}]', '[{"name":"Drama"}]'],
    })
    ratings = pd.DataFrame({
        "userId": [1, 1, 2, 2, 3, 3],
        "movieId": [11, 12, 11, 12, 11, 13],
        "rating": [5.0, 4.5, 4.0, 4.0, 5.0, 4.5],
    })
    links = pd.DataFrame({"movieId": [11, 12, 13], "tmdbId": [101, 102, 103]})
    model = build_collaborative_model(ratings, links, movies, top_neighbors=2)
    result = recommend_hybrid(
        "story", movies, np.array([[1.0, 0.0]] * 3), FakeModel(),
        user_id=1, collaborative_model=model, top_k=3,
    )
    assert "hybrid_score" in result.columns
    assert result["recommendation_type"].eq("hybrid").any()
