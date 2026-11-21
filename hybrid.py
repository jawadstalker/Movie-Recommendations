from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


@dataclass
class CollaborativeModel:
    user_item: object
    movie_ids: np.ndarray
    neighbors: dict
    user_to_index: dict
    ratings_min: float
    ratings_max: float


def load_movielens_data(ratings_path="ratings.csv", links_path="links.csv"):
    ratings = pd.read_csv(ratings_path)
    links = pd.read_csv(links_path)
    if not {"userId", "movieId", "rating"}.issubset(ratings.columns):
        raise ValueError("ratings.csv must contain userId, movieId, and rating columns.")
    if not {"movieId", "tmdbId"}.issubset(links.columns):
        raise ValueError("links.csv must contain movieId and tmdbId columns.")
    ratings = ratings.dropna(subset=["userId", "movieId", "rating"]).copy()
    links = links.dropna(subset=["movieId", "tmdbId"]).copy()
    ratings["movieId"] = pd.to_numeric(ratings["movieId"], errors="coerce")
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    links["movieId"] = pd.to_numeric(links["movieId"], errors="coerce")
    links["tmdbId"] = pd.to_numeric(links["tmdbId"], errors="coerce")
    return ratings.dropna(subset=["movieId", "rating"]), links.dropna(subset=["movieId", "tmdbId"])


def build_collaborative_model(ratings, links, movies, top_neighbors=50):
    from scipy.sparse import csr_matrix

    if not {"userId", "movieId", "rating"}.issubset(ratings.columns):
        raise ValueError("Ratings data must contain userId, movieId, and rating columns.")
    if not {"movieId", "tmdbId"}.issubset(links.columns) or "id" not in movies:
        raise ValueError("MovieLens links or TMDB movie data are missing required ID columns.")

    mapping = links[["movieId", "tmdbId"]].dropna().drop_duplicates("movieId").copy()
    mapping["tmdbId"] = pd.to_numeric(mapping["tmdbId"], errors="coerce")
    valid_ids = set(pd.to_numeric(movies["id"], errors="coerce").dropna().astype(int))
    mapping = mapping.dropna(subset=["tmdbId"])
    mapping = mapping[mapping["tmdbId"].astype(int).isin(valid_ids)]
    joined = ratings.merge(mapping, on="movieId", how="inner")
    joined["tmdbId"] = joined["tmdbId"].astype(int)
    joined["rating"] = pd.to_numeric(joined["rating"], errors="coerce")
    joined = joined.dropna(subset=["userId", "rating"])
    joined = joined.drop_duplicates(["userId", "tmdbId"], keep="last")
    if joined.empty:
        raise ValueError("No MovieLens ratings match the TMDB movie catalogue.")

    user_ids = pd.Index(joined["userId"].unique())
    movie_ids = pd.Index(joined["tmdbId"].unique())
    user_indices = user_ids.get_indexer(joined["userId"])
    movie_indices = movie_ids.get_indexer(joined["tmdbId"])
    matrix = csr_matrix(
        (joined["rating"].to_numpy(dtype=np.float32), (user_indices, movie_indices)),
        shape=(len(user_ids), len(movie_ids)),
        dtype=np.float32,
    )

    item_vectors = normalize(matrix.T, norm="l2", axis=1)
    count = min(max(2, top_neighbors + 1), len(movie_ids))
    nearest = NearestNeighbors(n_neighbors=count, metric="cosine", algorithm="brute")
    nearest.fit(item_vectors)
    distances, indices = nearest.kneighbors(item_vectors)
    neighbors = {}
    for index, tmdb_id in enumerate(movie_ids):
        neighbors[int(tmdb_id)] = [
            (int(movie_ids[j]), float(1.0 - distance))
            for j, distance in zip(indices[index], distances[index])
            if j != index and distance < 1.0
        ][:top_neighbors]

    return CollaborativeModel(
        user_item=matrix,
        movie_ids=movie_ids.to_numpy(dtype=int),
        neighbors=neighbors,
        user_to_index={user_id: index for index, user_id in enumerate(user_ids)},
        ratings_min=float(joined["rating"].min()),
        ratings_max=float(joined["rating"].max()),
    )


def collaborative_scores(user_id, collaborative_model):
    user_index = collaborative_model.user_to_index.get(user_id)
    if user_index is None:
        return {}
    row = collaborative_model.user_item.getrow(user_index)
    rated = {
        int(collaborative_model.movie_ids[index]): float(rating)
        for index, rating in zip(row.indices, row.data)
    }
    if not rated:
        return {}

    mean_rating = float(np.mean(list(rated.values())))
    totals = {}
    for seen_id, rating in rated.items():
        for candidate_id, similarity in collaborative_model.neighbors.get(seen_id, []):
            if candidate_id in rated or similarity <= 0:
                continue
            numerator, denominator = totals.get(candidate_id, (0.0, 0.0))
            totals[candidate_id] = (
                numerator + similarity * (rating - mean_rating),
                denominator + abs(similarity),
            )

    low, high = collaborative_model.ratings_min, collaborative_model.ratings_max
    scores = {}
    for movie_id, (numerator, denominator) in totals.items():
        prediction = mean_rating + numerator / denominator if denominator else mean_rating
        prediction = float(np.clip(prediction, low, high))
        scores[movie_id] = (prediction - low) / (high - low) if high > low else 0.5
    return scores


def recommend_hybrid(
    query, movies, embeddings, model, user_id=None, collaborative_model=None,
    top_k=10, content_weight=0.65, min_year=None, max_year=None, genre=None,
):
    from recommender import recommend_movies

    if not 0.0 <= content_weight <= 1.0:
        raise ValueError("content_weight must be between 0 and 1.")
    content = recommend_movies(
        query, movies, embeddings, model, top_k=len(movies),
        min_year=min_year, max_year=max_year, genre=genre,
    )
    if content.empty:
        return content
    if collaborative_model is None or user_id is None:
        content["hybrid_score"] = content["similarity"].clip(0, 1)
        content["recommendation_type"] = "content-based"
        return content.head(top_k)

    cf = collaborative_scores(user_id, collaborative_model)
    if not cf:
        content["hybrid_score"] = content["similarity"].clip(0, 1)
        content["recommendation_type"] = "content-based (cold start)"
        return content.head(top_k)

    content["collaborative_score"] = pd.to_numeric(content["id"], errors="coerce").map(cf)
    has_cf = content["collaborative_score"].notna()
    content["hybrid_score"] = content["similarity"].clip(0, 1)
    content.loc[has_cf, "hybrid_score"] = (
        content.loc[has_cf, "similarity"].clip(0, 1) * content_weight
        + content.loc[has_cf, "collaborative_score"] * (1.0 - content_weight)
    )
    content["recommendation_type"] = np.where(has_cf, "hybrid", "content-based fallback")
    return content.sort_values("hybrid_score", ascending=False).head(top_k)
