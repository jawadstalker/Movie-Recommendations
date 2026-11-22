"""Lightweight local A/B testing for recommendation ranking strategies."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

DEFAULT_EVENT_FILE = Path(".data") / "experiment_events.jsonl"
DEFAULT_EXPERIMENT = "ranking_mmr_v1"
VARIANTS = ("A", "B")


def _profile_hash(profile_id):
    if profile_id is None or not str(profile_id).strip():
        raise ValueError("A non-empty local profile ID is required.")
    return hashlib.sha256(str(profile_id).strip().encode("utf-8")).hexdigest()[:16]


def assign_variant(profile_id, experiment=DEFAULT_EXPERIMENT):
    """Assign a profile consistently to A/B without storing its raw local ID."""
    if not experiment or not str(experiment).strip():
        raise ValueError("experiment must be a non-empty string.")
    digest = hashlib.sha256(f"{experiment}:{_profile_hash(profile_id)}".encode("utf-8")).digest()
    return "A" if digest[0] % 2 == 0 else "B"


def _append_event(event, path=DEFAULT_EVENT_FILE):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")


def record_exposure(profile_id, variant, movie_ids, experiment=DEFAULT_EXPERIMENT, path=DEFAULT_EVENT_FILE):
    """Record one displayed recommendation list and return its impression ID."""
    if variant not in VARIANTS:
        raise ValueError("variant must be A or B.")
    normalized_ids = []
    for movie_id in movie_ids:
        try:
            value = int(movie_id)
        except (TypeError, ValueError) as exc:
            raise ValueError("movie_ids must contain integer TMDB IDs.") from exc
        if value not in normalized_ids:
            normalized_ids.append(value)
    if not normalized_ids:
        raise ValueError("An exposure must contain at least one recommended movie.")
    event = {
        "event_type": "exposure",
        "experiment": str(experiment),
        "variant": variant,
        "impression_id": uuid.uuid4().hex,
        "profile_hash": _profile_hash(profile_id),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "movie_ids": normalized_ids,
    }
    _append_event(event, path)
    return event


def record_outcome(profile_id, impression_id, movie_id, action, experiment=DEFAULT_EXPERIMENT, path=DEFAULT_EVENT_FILE):
    """Record explicit feedback on a movie from a previously displayed impression."""
    if action not in {"like", "dislike", "watchlist"}:
        raise ValueError("action must be like, dislike, or watchlist.")
    if not impression_id:
        raise ValueError("impression_id is required to attribute an outcome.")
    event = {
        "event_type": "outcome",
        "experiment": str(experiment),
        "impression_id": str(impression_id),
        "profile_hash": _profile_hash(profile_id),
        "movie_id": int(movie_id),
        "action": action,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _append_event(event, path)
    return event


def summarize_experiment(experiment=DEFAULT_EXPERIMENT, path=DEFAULT_EVENT_FILE):
    """Summarize item-level explicit-feedback rates for each randomized variant."""
    path = Path(path)
    if not path.exists():
        return {
            "experiment": experiment,
            "variants": {variant: _empty_summary() for variant in VARIANTS},
        }

    events = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid experiment event at line {line_number}: {exc}") from exc

    exposures = {
        event["impression_id"]: event
        for event in events
        if event.get("event_type") == "exposure" and event.get("experiment") == experiment
    }
    outcomes = {}
    for event in events:
        if event.get("event_type") != "outcome" or event.get("experiment") != experiment:
            continue
        exposure = exposures.get(event.get("impression_id"))
        if not exposure or event.get("profile_hash") != exposure.get("profile_hash"):
            continue
        try:
            movie_id = int(event["movie_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if movie_id not in exposure.get("movie_ids", []):
            continue
        outcomes[(event["impression_id"], movie_id)] = event

    summaries = {}
    for variant in VARIANTS:
        variant_exposures = [event for event in exposures.values() if event.get("variant") == variant]
        item_count = sum(len(event.get("movie_ids", [])) for event in variant_exposures)
        profile_count = len({event.get("profile_hash") for event in variant_exposures})
        variant_outcomes = [
            outcome for (impression_id, _), outcome in outcomes.items()
            if exposures[impression_id].get("variant") == variant
        ]
        counts = {action: sum(item.get("action") == action for item in variant_outcomes)
                  for action in ("like", "dislike", "watchlist")}
        interacted = len(variant_outcomes)
        denominator = item_count or 0
        summaries[variant] = {
            "impressions": len(variant_exposures),
            "profiles": profile_count,
            "recommended_items": item_count,
            "interacted_items": interacted,
            "likes": counts["like"],
            "dislikes": counts["dislike"],
            "watchlist_adds": counts["watchlist"],
            "engagement_rate": interacted / denominator if denominator else 0.0,
            "like_rate": counts["like"] / denominator if denominator else 0.0,
            "dislike_rate": counts["dislike"] / denominator if denominator else 0.0,
            "watchlist_rate": counts["watchlist"] / denominator if denominator else 0.0,
        }

    return {"experiment": experiment, "variants": summaries}


def _empty_summary():
    return {
        "impressions": 0, "profiles": 0, "recommended_items": 0,
        "interacted_items": 0, "likes": 0, "dislikes": 0, "watchlist_adds": 0,
        "engagement_rate": 0.0, "like_rate": 0.0,
        "dislike_rate": 0.0, "watchlist_rate": 0.0,
    }
