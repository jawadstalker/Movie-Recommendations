"""Local, file-backed user feedback for the movie recommender."""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

FEEDBACK_FILE = Path(".data") / "feedback.json"
VALID_ACTIONS = {"like", "dislike", "watchlist", "remove_watchlist"}


def _read_store(path=FEEDBACK_FILE):
    path = Path(path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read feedback store at {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Feedback store must contain a JSON object.")
    return data


def _write_store(data, path=FEEDBACK_FILE):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
        ) as temp:
            json.dump(data, temp, ensure_ascii=False, indent=2)
            temp.write("\n")
            temp_name = temp.name
        os.replace(temp_name, path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def record_feedback(user_id, movie_id, title, action, path=FEEDBACK_FILE):
    """Append a feedback event. The latest event for a movie defines its state."""
    if action not in VALID_ACTIONS:
        raise ValueError(f"action must be one of: {', '.join(sorted(VALID_ACTIONS))}")
    if user_id is None or not str(user_id).strip():
        raise ValueError("A non-empty local profile ID is required.")
    try:
        movie_id = int(movie_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("movie_id must be an integer TMDB ID.") from exc

    data = _read_store(path)
    profile = str(user_id).strip()
    events = data.setdefault(profile, [])
    if not isinstance(events, list):
        raise ValueError(f"Feedback data for profile {profile!r} is invalid.")
    events.append({
        "movie_id": movie_id,
        "title": str(title),
        "action": action,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    _write_store(data, path)
    return events[-1]


def get_user_feedback(user_id, path=FEEDBACK_FILE):
    """Return the latest feedback state for each movie in a profile."""
    if user_id is None or not str(user_id).strip():
        return []
    events = _read_store(path).get(str(user_id).strip(), [])
    if not isinstance(events, list):
        raise ValueError("Feedback events must be stored as a list.")
    latest = {}
    for event in events:
        if isinstance(event, dict) and "movie_id" in event and "action" in event:
            latest[int(event["movie_id"])] = event
    return list(latest.values())


def apply_feedback(results, embeddings, feedback, top_k=10, base_weight=0.8):
    """Rerank candidates using liked/disliked movies and hide saved/rejected items."""
    if results.empty or not feedback:
        return results.head(top_k).copy()
    if not 0.0 <= base_weight <= 1.0:
        raise ValueError("base_weight must be between 0 and 1.")

    state = {int(item["movie_id"]): item["action"] for item in feedback}
    excluded = {
        movie_id for movie_id, action in state.items()
        if action in {"dislike", "watchlist"}
    }
    ranked = results.copy()
    if "id" not in ranked:
        raise ValueError("Recommendation results must contain TMDB movie IDs.")
    ranked_ids = pd.to_numeric(ranked["id"], errors="coerce")
    ranked = ranked.loc[~ranked_ids.isin(excluded)].copy()
    if ranked.empty:
        return ranked

    positive_indices, negative_indices = [], []
    for index, movie_id in zip(ranked.index, ranked_ids.loc[ranked.index]):
        action = state.get(int(movie_id)) if pd.notna(movie_id) else None
        if action == "like" and 0 <= int(index) < len(embeddings):
            positive_indices.append(int(index))
        elif action == "dislike" and 0 <= int(index) < len(embeddings):
            negative_indices.append(int(index))

    if not positive_indices and not negative_indices:
        return ranked.head(top_k)

    profile_parts = []
    if positive_indices:
        profile_parts.append(np.mean(embeddings[positive_indices], axis=0))
    if negative_indices:
        profile_parts.append(-np.mean(embeddings[negative_indices], axis=0))
    profile_vector = np.sum(profile_parts, axis=0, keepdims=True)
    norm = np.linalg.norm(profile_vector)
    if not np.isfinite(norm) or norm == 0:
        return ranked.head(top_k)
    profile_vector = profile_vector / norm

    candidate_vectors = embeddings[ranked.index.to_numpy(dtype=int)]
    affinity = cosine_similarity(candidate_vectors, profile_vector).ravel()
    affinity = np.clip((affinity + 1.0) / 2.0, 0.0, 1.0)
    base_column = "hybrid_score" if "hybrid_score" in ranked else "similarity"
    ranked["feedback_score"] = affinity
    ranked["personalized_score"] = (
        ranked[base_column].fillna(0).clip(0, 1) * base_weight
        + ranked["feedback_score"] * (1.0 - base_weight)
    )
    return ranked.sort_values("personalized_score", ascending=False).head(top_k)


def get_watchlist(feedback):
    """Return movies whose latest profile action is watchlist."""
    return [item for item in feedback if item.get("action") == "watchlist"]
