import numpy as np
import pandas as pd

from recommender import build_movie_texts, parse_tmdb_field, recommend_movies


class FakeModel:
    def encode(self, texts, **kwargs):
        if isinstance(texts, list) and texts[0] == "space":
            return np.array([[1.0, 0.0]], dtype=np.float32)
        return np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)


def sample_movies():
    return pd.DataFrame({
        "title": ["Space Journey", "Quiet Garden"],
        "overview": ["A voyage among stars", "A gardener finds peace"],
        "genres": ['[{"id": 1, "name": "Science Fiction"}]', '[{"id": 2, "name": "Drama"}]'],
        "cast": ['[{"name": "Actor A"}]', '[{"name": "Actor B"}]'],
        "release_date": pd.to_datetime(["2001-01-01", "2010-01-01"]),
    })


def test_parse_tmdb_field_handles_json_and_missing_values():
    assert parse_tmdb_field('[{"name": "Drama"}]') == ["Drama"]
    assert parse_tmdb_field(None) == []
    assert parse_tmdb_field("not-json") == ["not-json"]


def test_build_movie_texts_includes_structured_metadata():
    texts = build_movie_texts(sample_movies())
    assert "Science Fiction" in texts[0]
    assert "Actor A" in texts[0]


def test_recommend_movies_ranks_and_adds_similarity():
    movies = sample_movies()
    result = recommend_movies("space", movies, np.eye(2), FakeModel(), top_k=1)
    assert result.iloc[0]["title"] == "Space Journey"
    assert result.iloc[0]["similarity"] == 1.0


def test_recommend_movies_supports_year_and_genre_filters():
    movies = sample_movies()
    result = recommend_movies(
        "space", movies, np.eye(2), FakeModel(),
        top_k=5, min_year=2005, genre="Drama",
    )
    assert result["title"].tolist() == ["Quiet Garden"]
