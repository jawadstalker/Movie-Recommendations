import numpy as np
import pandas as pd

from explainability import explain_recommendations


def sample_movies():
    return pd.DataFrame([
        {"id": 1, "title": "Space Drama", "genres": '[{"name":"Drama"}]',
         "cast": '[{"name":"Actor A"}]', "keywords": '[{"name":"space travel"}]',
         "crew": '[{"name":"Director A","job":"Director"}]'},
        {"id": 2, "title": "Space Comedy", "genres": '[{"name":"Comedy"}]',
         "cast": '[{"name":"Actor A"}]', "keywords": '[{"name":"space travel"}]',
         "crew": '[{"name":"Director B","job":"Director"}]'},
    ])


def test_explanations_include_real_search_and_profile_evidence():
    movies = sample_movies()
    results = movies.iloc[[1]].copy()
    results["similarity"] = [0.82]
    embeddings = np.array([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32)
    profile = {"preferences": {
        "genres": {"comedy": 1.0},
        "people": {"actor a": 1.0},
        "keywords": {"space travel": 1.0},
    }}
    explained = explain_recommendations(
        results, movies, embeddings, feedback=[], taste_profile=profile
    )
    reasons = explained.iloc[0]["explanations"]
    assert any("Semantic match" in reason and "0.82" in reason for reason in reasons)
    assert any("Preferred genre: Comedy" in reason for reason in reasons)
    assert any("Preferred actor/director: Actor A" in reason for reason in reasons)


def test_explanation_can_reference_previously_liked_movie():
    movies = sample_movies()
    results = movies.iloc[[1]].copy()
    results["similarity"] = [0.75]
    embeddings = np.array([[1.0, 0.0], [0.95, 0.05]], dtype=np.float32)
    explained = explain_recommendations(
        results, movies, embeddings,
        feedback=[{"movie_id": 1, "action": "like"}],
    )
    assert any("Space Drama" in reason for reason in explained.iloc[0]["explanations"])


def test_empty_results_return_explanation_column():
    movies = sample_movies()
    empty = movies.iloc[[]].copy()
    explained = explain_recommendations(empty, movies, np.zeros((2, 2)))
    assert "explanations" in explained.columns
    assert explained.empty
