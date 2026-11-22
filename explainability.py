"""Evidence-based, human-readable explanations for movie recommendations."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from taste_profile import _movie_features


def _safe_float(value):
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _liked_movie_titles(movies, feedback):
    by_id = {}
    for _, movie in movies.iterrows():
        try:
            by_id[int(movie["id"])] = str(movie.get("title", "a movie"))
        except (KeyError, TypeError, ValueError):
            continue
    return {
        int(event["movie_id"]): by_id[int(event["movie_id"])]
        for event in feedback
        if event.get("action") == "like"
        and str(event.get("movie_id", "")).lstrip("-").isdigit()
        and int(event["movie_id"]) in by_id
    }


def explain_recommendations(results, movies, embeddings, feedback=None, taste_profile=None, top_reasons=3):
    """Add a list of grounded explanation strings to each recommendation row.

    DataFrame indices must remain aligned with the movie catalogue/embedding rows,
    as they are in the existing recommender pipeline.
    """
    if results.empty:
        explained = results.copy()
        explained["explanations"] = pd.Series(dtype=object)
        return explained

    feedback = feedback or []
    taste_profile = taste_profile or {"preferences": {}}
    liked_titles = _liked_movie_titles(movies, feedback)
    liked_indices = []
    for movie_id in liked_titles:
        matches = movies.index[pd.to_numeric(movies["id"], errors="coerce") == movie_id].tolist()
        if matches:
            idx = int(matches[0])
            if 0 <= idx < len(embeddings):
                liked_indices.append((idx, liked_titles[movie_id]))

    explanations = []
    for index, movie in results.iterrows():
        reasons = []
        semantic = _safe_float(movie.get("similarity"))
        if semantic is not None:
            reasons.append((1.0, f"Semantic match to your search (score: {semantic:.2f})."))

        features = _movie_features(movie)
        preferences = taste_profile.get("preferences", {})
        for category, label in (
            ("genres", "Preferred genre"),
            ("people", "Preferred actor/director"),
            ("keywords", "Preferred story theme"),
        ):
            category_preferences = preferences.get(category, {})
            matches = []
            for feature in features[category]:
                value = category_preferences.get(str(feature).casefold())
                if value is not None and value > 0.05:
                    matches.append((str(feature), float(value)))
            if matches:
                matches.sort(key=lambda item: item[1], reverse=True)
                names = ", ".join(name for name, _ in matches[:2])
                reasons.append((0.9, f"{label}: {names}."))

        if liked_indices and 0 <= int(index) < len(embeddings):
            similarities = cosine_similarity(
                np.asarray(embeddings[int(index)]).reshape(1, -1),
                np.asarray([embeddings[idx] for idx, _ in liked_indices]),
            ).ravel()
            best_position = int(np.argmax(similarities))
            best_score = float(similarities[best_position])
            if best_score >= 0.35:
                reasons.append((
                    0.8,
                    f"Similar to a movie you liked: {liked_indices[best_position][1]} (similarity: {best_score:.2f}).",
                ))

        collaborative = _safe_float(movie.get("collaborative_score"))
        if collaborative is not None:
            reasons.append((0.7, f"Collaborative filtering signal: {collaborative:.2f}."))

        reasons.sort(key=lambda item: item[0], reverse=True)
        selected = [message for _, message in reasons[:max(1, int(top_reasons))]]
        if not selected:
            selected = ["Recommended based on the available movie metadata and ranking signals."]
        explanations.append(selected)

    explained = results.copy()
    explained["explanations"] = explanations
    return explained
