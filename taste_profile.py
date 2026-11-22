"""Dynamic metadata-based taste profiles built from explicit user feedback."""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timezone
import ast
import json
import math
import pandas as pd
from recommender import parse_tmdb_field

CATEGORY_WEIGHTS = {"genres": 0.45, "people": 0.30, "keywords": 0.25}
HALF_LIFE_DAYS = 120.0

def _director_names(value):
    if not isinstance(value, str):
        return []
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []
    if not isinstance(parsed, list):
        return []
    return [str(person.get("name", "")).strip() for person in parsed
            if isinstance(person, dict) and person.get("job") == "Director" and person.get("name")]

def _movie_features(movie):
    genres = parse_tmdb_field(movie.get("genres", ""))
    cast = parse_tmdb_field(movie.get("cast", ""))[:5]
    keywords = parse_tmdb_field(movie.get("keywords", ""))[:10]
    directors = _director_names(movie.get("crew", ""))
    return {"genres": genres, "people": list(dict.fromkeys(directors + cast)), "keywords": keywords}

def _event_weight(event, now):
    weight = 1.0 if event.get("action") == "like" else -1.0
    timestamp = event.get("timestamp")
    if not timestamp:
        return weight
    try:
        happened = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if happened.tzinfo is None:
            happened = happened.replace(tzinfo=timezone.utc)
        age_days = max(0.0, (now - happened).total_seconds() / 86400)
        return weight * math.exp(-math.log(2) * age_days / HALF_LIFE_DAYS)
    except (TypeError, ValueError):
        return weight

def build_taste_profile(movies, feedback, now=None):
    """Aggregate recency-weighted positive/negative preferences from latest states."""
    now = now or datetime.now(timezone.utc)
    movie_by_id = {}
    for _, movie in movies.iterrows():
        try:
            movie_by_id[int(movie["id"])] = movie
        except (KeyError, TypeError, ValueError):
            continue
    totals = {category: defaultdict(float) for category in CATEGORY_WEIGHTS}
    evidence = {category: defaultdict(float) for category in CATEGORY_WEIGHTS}
    liked = disliked = 0
    for event in feedback:
        action = event.get("action")
        if action not in {"like", "dislike"}:
            continue
        try:
            movie = movie_by_id.get(int(event["movie_id"]))
        except (KeyError, TypeError, ValueError):
            movie = None
        if movie is None:
            continue
        weight = _event_weight(event, now)
        if action == "like":
            liked += 1
        else:
            disliked += 1
        for category, features in _movie_features(movie).items():
            for feature in set(item.casefold() for item in features if item):
                totals[category][feature] += weight
                evidence[category][feature] += abs(weight)
    preferences = {
        category: {feature: score / evidence[category][feature]
                   for feature, score in totals[category].items() if evidence[category][feature] > 0}
        for category in CATEGORY_WEIGHTS
    }
    return {"preferences": preferences, "liked_count": liked, "disliked_count": disliked,
            "feedback_count": liked + disliked}

def score_taste_profile(results, profile):
    """Return metadata affinity scores in [0, 1] for each candidate row."""
    preferences = profile.get("preferences", {})
    if results.empty or not any(preferences.get(category) for category in CATEGORY_WEIGHTS):
        return pd.Series(0.5, index=results.index, dtype=float)
    scores = []
    for _, movie in results.iterrows():
        features = _movie_features(movie)
        category_scores, category_weights = [], []
        for category, category_weight in CATEGORY_WEIGHTS.items():
            values = preferences.get(category, {})
            matched = [values.get(feature.casefold()) for feature in features[category]]
            matched = [value for value in matched if value is not None]
            if matched:
                category_scores.append((sum(matched) / len(matched) + 1.0) / 2.0)
                category_weights.append(category_weight)
        scores.append(sum(s*w for s,w in zip(category_scores, category_weights)) / sum(category_weights)
                      if category_scores else 0.5)
    return pd.Series(scores, index=results.index, dtype=float)

def apply_taste_profile(results, profile, top_k=10, profile_weight=0.25):
    """Blend explicit taste affinity with the existing recommendation score."""
    if results.empty:
        return results.copy()
    if not 0.0 <= profile_weight <= 1.0:
        raise ValueError("profile_weight must be between 0 and 1.")
    ranked = results.copy()
    ranked["taste_score"] = score_taste_profile(ranked, profile)
    base_column = next((c for c in ("personalized_score", "hybrid_score", "similarity") if c in ranked), None)
    if base_column is None:
        raise ValueError("Results must contain a base recommendation score.")
    ranked["taste_profile_score"] = (ranked[base_column].fillna(0).clip(0, 1) * (1-profile_weight)
                                     + ranked["taste_score"] * profile_weight)
    return ranked.sort_values("taste_profile_score", ascending=False).head(top_k)

def describe_taste_profile(profile, limit=5):
    """Create concise human-readable preference summaries for the UI."""
    descriptions = []
    labels = {"genres": "Genres", "people": "Actors / directors", "keywords": "Story themes"}
    for category in CATEGORY_WEIGHTS:
        prefs = profile.get("preferences", {}).get(category, {})
        positive = sorted(((n,s) for n,s in prefs.items() if s > 0.05), key=lambda x:x[1], reverse=True)[:limit]
        negative = sorted(((n,s) for n,s in prefs.items() if s < -0.05), key=lambda x:x[1])[:limit]
        descriptions.append((labels[category], positive, negative))
    return descriptions
