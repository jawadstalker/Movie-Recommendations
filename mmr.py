"""Diversity-aware reranking with Maximal Marginal Relevance (MMR)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def select_diverse_recommendations(
    results,
    embeddings,
    top_k=10,
    diversity_weight=0.35,
    relevance_column=None,
):
    """Greedily select relevant but mutually diverse recommendations."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if not 0.0 <= diversity_weight <= 1.0:
        raise ValueError("diversity_weight must be between 0 and 1.")
    if results.empty:
        empty = results.copy()
        empty["mmr_score"] = pd.Series(dtype=float)
        empty["diversity_score"] = pd.Series(dtype=float)
        return empty

    vectors = np.asarray(embeddings, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional array.")
    score_column = relevance_column or next(
        (column for column in ("taste_profile_score", "personalized_score", "hybrid_score", "similarity")
         if column in results),
        None,
    )
    if score_column is None:
        raise ValueError("Results must contain a relevance score column.")

    indices = []
    for index in results.index:
        try:
            row_index = int(index)
        except (TypeError, ValueError) as exc:
            raise ValueError("Result indices must be integer positions into embeddings.") from exc
        if row_index < 0 or row_index >= len(vectors):
            raise ValueError(f"Result index {row_index} is outside the embeddings array.")
        indices.append(row_index)

    candidate_vectors = vectors[np.asarray(indices, dtype=int)]
    if not np.isfinite(candidate_vectors).all():
        raise ValueError("Candidate embeddings contain non-finite values.")
    relevance = pd.to_numeric(results[score_column], errors="coerce").fillna(0).to_numpy(dtype=float)
    relevance = np.clip(relevance, 0.0, 1.0)
    pairwise = np.clip(cosine_similarity(candidate_vectors), -1.0, 1.0)

    remaining = list(range(len(results)))
    selected = []
    mmr_scores, diversity_scores = {}, {}
    while remaining and len(selected) < min(top_k, len(results)):
        best_position = None
        best_mmr = -float("inf")
        best_diversity = 1.0
        for position in remaining:
            if selected:
                redundancy = max(0.0, max(float(pairwise[position, chosen]) for chosen in selected))
                diversity = 1.0 - redundancy
            else:
                redundancy, diversity = 0.0, 1.0
            mmr = (1.0 - diversity_weight) * relevance[position] - diversity_weight * redundancy
            if mmr > best_mmr:
                best_position, best_mmr, best_diversity = position, mmr, diversity
        selected.append(best_position)
        remaining.remove(best_position)
        mmr_scores[best_position] = best_mmr
        diversity_scores[best_position] = best_diversity

    ranked = results.iloc[selected].copy()
    ranked["mmr_score"] = [mmr_scores[position] for position in selected]
    ranked["diversity_score"] = [diversity_scores[position] for position in selected]
    return ranked
