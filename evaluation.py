from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from hybrid import build_collaborative_model, collaborative_scores
from recommender import build_movie_texts
from mmr import select_diverse_recommendations


@dataclass
class RankingMetrics:
    precision: float
    recall: float
    map: float
    ndcg: float
    hit_rate: float
    users_evaluated: int


def _ranking_metrics(recommendations, relevant_items, k):
    relevant = set(relevant_items)
    ranked = list(recommendations[:k])
    if not relevant:
        return None

    hits = [1 if item in relevant else 0 for item in ranked]
    precision = sum(hits) / k
    recall = sum(hits) / len(relevant)

    running_hits = 0
    precision_sum = 0.0
    for rank, hit in enumerate(hits, start=1):
        if hit:
            running_hits += 1
            precision_sum += running_hits / rank
    average_precision = precision_sum / min(len(relevant), k)

    discounts = 1.0 / np.log2(np.arange(2, len(hits) + 2))
    dcg = float(np.sum(np.asarray(hits) * discounts))
    ideal_hits = min(len(relevant), k)
    idcg = float(np.sum(1.0 / np.log2(np.arange(2, ideal_hits + 2))))
    ndcg = dcg / idcg if idcg else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "map": average_precision,
        "ndcg": ndcg,
        "hit_rate": float(any(hits)),
    }


def intra_list_diversity(item_indices, embeddings):
    """Mean pairwise cosine distance for a ranked list of embedding row indices."""
    indices = [int(index) for index in item_indices]
    if len(indices) < 2:
        return 0.0
    vectors = np.asarray(embeddings, dtype=np.float32)
    if vectors.ndim != 2 or any(index < 0 or index >= len(vectors) for index in indices):
        raise ValueError("Embedding indices must refer to valid rows in a 2D embedding matrix.")
    similarities = cosine_similarity(vectors[indices])
    upper = similarities[np.triu_indices(len(indices), k=1)]
    return float(np.mean(1.0 - np.clip(upper, -1.0, 1.0)))


def _aggregate(rows):
    if not rows:
        return RankingMetrics(0, 0, 0, 0, 0, 0)
    frame = pd.DataFrame(rows)
    return RankingMetrics(
        precision=float(frame["precision"].mean()),
        recall=float(frame["recall"].mean()),
        map=float(frame["map"].mean()),
        ndcg=float(frame["ndcg"].mean()),
        hit_rate=float(frame["hit_rate"].mean()),
        users_evaluated=len(frame),
    )


def evaluate_recommenders(
    movies,
    embeddings,
    model,
    ratings,
    links,
    k=10,
    relevance_threshold=4.0,
    min_train_ratings=3,
    content_weight=0.65,
    max_users=None,
    mmr_weights=(0.2, 0.35, 0.5),
    mmr_candidate_pool=100,
):
    """
    Per-user temporal holdout evaluation.

    The latest rating of each eligible user is held out. All earlier ratings
    form the training set. Only held-out ratings >= relevance_threshold are
    treated as relevant test items.
    """
    if k < 1:
        raise ValueError("k must be at least 1.")
    if not 0 <= content_weight <= 1:
        raise ValueError("content_weight must be between 0 and 1.")
    mmr_weights = tuple(float(weight) for weight in mmr_weights)
    if not mmr_weights or any(not 0 <= weight <= 1 for weight in mmr_weights):
        raise ValueError("mmr_weights must contain values between 0 and 1.")
    if len(set(mmr_weights)) != len(mmr_weights):
        raise ValueError("mmr_weights must not contain duplicates.")
    if mmr_candidate_pool < k:
        raise ValueError("mmr_candidate_pool must be at least k.")
    if not {"userId", "movieId", "rating", "timestamp"}.issubset(ratings.columns):
        raise ValueError("ratings must contain userId, movieId, rating, and timestamp.")
    if not {"movieId", "tmdbId"}.issubset(links.columns):
        raise ValueError("links must contain movieId and tmdbId.")
    if "id" not in movies:
        raise ValueError("movies must contain TMDB id.")
    if len(movies) != len(embeddings):
        raise ValueError("Movie rows and embedding rows must have the same length.")

    rating_data = ratings.copy()
    link_data = links[["movieId", "tmdbId"]].dropna().drop_duplicates("movieId").copy()
    rating_data["rating"] = pd.to_numeric(rating_data["rating"], errors="coerce")
    rating_data["timestamp"] = pd.to_numeric(rating_data["timestamp"], errors="coerce")
    rating_data["movieId"] = pd.to_numeric(rating_data["movieId"], errors="coerce")
    link_data["movieId"] = pd.to_numeric(link_data["movieId"], errors="coerce")
    link_data["tmdbId"] = pd.to_numeric(link_data["tmdbId"], errors="coerce")
    rating_data = rating_data.dropna(subset=["userId", "movieId", "rating", "timestamp"])
    link_data = link_data.dropna(subset=["movieId", "tmdbId"])
    rating_data = rating_data.merge(link_data, on="movieId", how="inner")
    rating_data["tmdbId"] = rating_data["tmdbId"].astype(int)

    valid_movie_ids = set(pd.to_numeric(movies["id"], errors="coerce").dropna().astype(int))
    rating_data = rating_data[rating_data["tmdbId"].isin(valid_movie_ids)]
    rating_data = rating_data.sort_values(["userId", "timestamp", "movieId"])
    if rating_data.empty:
        raise ValueError("No MovieLens ratings map to movies in the TMDB catalogue.")

    grouped = rating_data.groupby("userId", sort=False)
    eligible = [
        (user_id, group)
        for user_id, group in grouped
        if len(group) >= min_train_ratings + 1
    ]
    if max_users is not None:
        eligible = eligible[:max_users]
    if not eligible:
        raise ValueError("No users have enough ratings for the requested temporal split.")

    heldout_indices = [group.index[-1] for _, group in eligible]
    test_rows = rating_data.loc[heldout_indices].copy()
    train = rating_data.drop(index=heldout_indices).copy()
    relevant_test = test_rows[test_rows["rating"] >= relevance_threshold]
    if relevant_test.empty:
        raise ValueError("No held-out ratings meet the relevance threshold.")

    train_ratings = train[["userId", "movieId", "rating"]].copy()
    train_links = link_data.copy()
    collaborative = build_collaborative_model(train_ratings, train_links, movies)

    movie_ids = pd.to_numeric(movies["id"], errors="coerce").fillna(-1).astype(int).to_numpy()
    movie_texts = build_movie_texts(movies)
    id_to_index = {movie_id: index for index, movie_id in enumerate(movie_ids) if movie_id >= 0}
    embedding_matrix = np.asarray(embeddings, dtype=np.float32)

    content_rows = []
    hybrid_rows = []
    user_details = []
    mmr_rows = {weight: [] for weight in mmr_weights}
    mmr_diversity_rows = {weight: [] for weight in mmr_weights}
    test_by_user = relevant_test.groupby("userId")

    for user_id, group in eligible:
        if user_id not in test_by_user.groups:
            continue
        user_train = group[group.index.isin(train.index)]
        if len(user_train) < min_train_ratings:
            continue

        liked = user_train[user_train["rating"] >= relevance_threshold]
        profile_rows = liked if not liked.empty else user_train
        profile_texts = []
        for tmdb_id in profile_rows["tmdbId"].astype(int):
            index = id_to_index.get(tmdb_id)
            if index is not None:
                profile_texts.append(movie_texts[index])
        if not profile_texts:
            continue

        profile = " ".join(profile_texts[:20])
        query_vector = model.encode(
            [profile], convert_to_numpy=True, normalize_embeddings=True
        )
        content_scores = cosine_similarity(query_vector, embedding_matrix)[0]
        cf_scores = collaborative_scores(user_id, collaborative)

        seen_ids = set(user_train["tmdbId"].astype(int))
        candidate_indices = np.asarray([
            index for index, movie_id in enumerate(movie_ids)
            if movie_id >= 0 and movie_id not in seen_ids
        ], dtype=int)
        if not len(candidate_indices):
            continue

        candidate_content = np.clip(content_scores[candidate_indices], 0, 1)
        candidate_ids = movie_ids[candidate_indices]
        candidate_cf = np.asarray(
            [cf_scores.get(int(movie_id), np.nan) for movie_id in candidate_ids],
            dtype=float,
        )
        has_cf = np.isfinite(candidate_cf)
        candidate_hybrid = candidate_content.copy()
        candidate_hybrid[has_cf] = (
            candidate_content[has_cf] * content_weight
            + candidate_cf[has_cf] * (1.0 - content_weight)
        )

        relevant_ids = set(test_by_user.get_group(user_id)["tmdbId"].astype(int))
        content_order = candidate_ids[np.argsort(candidate_content)[::-1][:k]]
        hybrid_order = candidate_ids[np.argsort(candidate_hybrid)[::-1][:k]]
        content_metric = _ranking_metrics(content_order, relevant_ids, k)
        hybrid_metric = _ranking_metrics(hybrid_order, relevant_ids, k)
        if content_metric is None or hybrid_metric is None:
            continue

        shortlist_size = min(len(candidate_indices), max(k, mmr_candidate_pool))
        shortlist_positions = np.argsort(candidate_hybrid)[::-1][:shortlist_size]
        shortlist_indices = candidate_indices[shortlist_positions]
        shortlist = movies.iloc[shortlist_indices].copy()
        shortlist["evaluation_score"] = candidate_hybrid[shortlist_positions]
        mmr_user_metrics = {}
        for weight in mmr_weights:
            diverse = select_diverse_recommendations(
                shortlist, embedding_matrix, top_k=k,
                diversity_weight=weight, relevance_column="evaluation_score",
            )
            diverse_indices = diverse.index.to_numpy(dtype=int)
            diverse_ids = movie_ids[diverse_indices]
            metric = _ranking_metrics(diverse_ids, relevant_ids, k)
            if metric is not None:
                mmr_rows[weight].append(metric)
                mmr_diversity_rows[weight].append(
                    intra_list_diversity(diverse_indices, embedding_matrix)
                )
                mmr_user_metrics[f"mmr_{weight:.2f}_ndcg"] = metric["ndcg"]
                mmr_user_metrics[f"mmr_{weight:.2f}_ild"] = mmr_diversity_rows[weight][-1]

        content_rows.append(content_metric)
        hybrid_rows.append(hybrid_metric)
        user_details.append({
            "user_id": user_id,
            "relevant_items": len(relevant_ids),
            "content_hit": content_metric["hit_rate"],
            "hybrid_hit": hybrid_metric["hit_rate"],
            "content_ndcg": content_metric["ndcg"],
            "hybrid_ndcg": hybrid_metric["ndcg"],
            **mmr_user_metrics,
        })

    if not content_rows:
        raise ValueError("No eligible users remained after applying the evaluation filters.")

    return {
        "k": k,
        "relevance_threshold": relevance_threshold,
        "content_based": asdict(_aggregate(content_rows)),
        "hybrid": asdict(_aggregate(hybrid_rows)),
        "mmr": {
            f"{weight:.2f}": {
                "ranking": asdict(_aggregate(mmr_rows[weight])),
                "intra_list_diversity": float(np.mean(mmr_diversity_rows[weight]))
                if mmr_diversity_rows[weight] else 0.0,
                "users_evaluated": len(mmr_rows[weight]),
                "candidate_pool": min(mmr_candidate_pool, len(movie_ids)),
            }
            for weight in mmr_weights
        },
        "improvement": {
            metric: (
                _aggregate(hybrid_rows).__dict__[metric]
                - _aggregate(content_rows).__dict__[metric]
            )
            for metric in ("precision", "recall", "map", "ndcg", "hit_rate")
        },
        "per_user": pd.DataFrame(user_details),
        "users_evaluated": len(content_rows),
        "split": "latest interaction per user; earlier interactions used for training",
    }


def save_evaluation_report(results, output_dir="reports"):
    from pathlib import Path
    import json

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = {key: value for key, value in results.items() if key != "per_user"}
    with (output / "evaluation_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    results["per_user"].to_csv(output / "evaluation_per_user.csv", index=False)
    return output
