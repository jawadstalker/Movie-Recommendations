import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR = Path(".cache")
CACHE_FILE = CACHE_DIR / "movie_embeddings.npz"


def parse_tmdb_field(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        items = value
    else:
        try:
            items = json.loads(value)
        except (TypeError, ValueError):
            try:
                items = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return [str(value)] if str(value).strip() else []
    return [
        str(item.get("name", "")).strip()
        for item in items
        if isinstance(item, dict) and item.get("name")
    ]


def load_movies(data_dir="."):
    data_dir = Path(data_dir)
    movies_path = data_dir / "tmdb_5000_movies.csv"
    credits_path = data_dir / "tmdb_5000_credits.csv"
    missing = [p.name for p in (movies_path, credits_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing dataset file(s): " + ", ".join(missing)
            + ". Download the TMDB 5000 Movies dataset and place both CSV files in the project directory."
        )

    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)
    if "id" not in movies or "movie_id" not in credits:
        raise ValueError("The CSV files do not have the expected TMDB id columns.")

    movies = movies.merge(credits, left_on="id", right_on="movie_id", how="inner")
    if "title" not in movies:
        raise ValueError("The movies CSV must contain a title column.")
    movies["overview"] = movies.get("overview", pd.Series("", index=movies.index)).fillna("")
    movies = movies[movies["title"].fillna("").str.strip().ne("")].reset_index(drop=True)
    if movies.empty:
        raise ValueError("No movies with valid titles were found in the dataset.")
    return movies


def build_movie_texts(movies):
    texts = []
    for _, movie in movies.iterrows():
        genres = parse_tmdb_field(movie.get("genres", ""))
        cast = parse_tmdb_field(movie.get("cast", ""))[:5]
        keywords = parse_tmdb_field(movie.get("keywords", ""))
        parts = [
            str(movie.get("title", "")),
            str(movie.get("tagline", "") or ""),
            str(movie.get("overview", "") or ""),
            "Genres: " + ", ".join(genres),
            "Cast: " + ", ".join(cast),
            "Keywords: " + ", ".join(keywords[:10]),
        ]
        texts.append(" ".join(part for part in parts if part.strip()))
    return texts


def dataset_fingerprint(texts, model_name=MODEL_NAME):
    digest = hashlib.sha256(model_name.encode("utf-8"))
    for text in texts:
        digest.update(text.encode("utf-8", errors="replace"))
        digest.update(b"\0")
    return digest.hexdigest()


def get_movie_embeddings(texts, model, cache_file=CACHE_FILE, model_name=MODEL_NAME):
    cache_file = Path(cache_file)
    fingerprint = dataset_fingerprint(texts, model_name)
    if cache_file.is_file():
        try:
            cached = np.load(cache_file, allow_pickle=False)
            if str(cached["fingerprint"].item()) == fingerprint:
                vectors = cached["embeddings"]
                if vectors.shape[0] == len(texts):
                    return vectors
        except (OSError, ValueError, KeyError):
            pass

    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(texts):
        raise ValueError("The embedding model returned an unexpected array shape.")

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, embeddings=vectors, fingerprint=fingerprint)
    return vectors


def recommend_movies(query, movies, embeddings, model, top_k=5, min_year=None, max_year=None, genre=None):
    query = str(query).strip()
    if not query:
        raise ValueError("Enter a description of the movie you want to watch.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    query_vector = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores = cosine_similarity(query_vector, embeddings)[0]
    candidates = np.arange(len(movies))

    if min_year is not None or max_year is not None:
        years = pd.to_datetime(movies.get("release_date"), errors="coerce").dt.year
        if min_year is not None:
            candidates = candidates[years.iloc[candidates].fillna(0).to_numpy() >= min_year]
        if max_year is not None:
            candidates = candidates[years.iloc[candidates].fillna(9999).to_numpy() <= max_year]

    if genre:
        genre_mask = movies.iloc[candidates]["genres"].apply(
            lambda value: genre.lower() in [name.lower() for name in parse_tmdb_field(value)]
        ).to_numpy()
        candidates = candidates[genre_mask]

    if not len(candidates):
        return movies.iloc[[]].assign(similarity=pd.Series(dtype=float))
    ranked = candidates[np.argsort(scores[candidates])[::-1][:top_k]]
    results = movies.iloc[ranked].copy()
    results["similarity"] = scores[ranked]
    return results
