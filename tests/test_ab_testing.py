import json

import pytest

from ab_testing import (
    assign_variant,
    record_exposure,
    record_outcome,
    summarize_experiment,
)


def test_variant_assignment_is_stable_and_balanced_over_many_profiles():
    assert assign_variant("profile-1") == assign_variant("profile-1")
    variants = {assign_variant(f"profile-{index}") for index in range(100)}
    assert variants == {"A", "B"}


def test_exposure_and_outcomes_are_attributed_to_correct_variant(tmp_path):
    path = tmp_path / "events.jsonl"
    variant = assign_variant("jawad")
    exposure = record_exposure("jawad", variant, [10, 11, 12], path=path)
    record_outcome("jawad", exposure["impression_id"], 10, "like", path=path)
    record_outcome("jawad", exposure["impression_id"], 11, "watchlist", path=path)

    report = summarize_experiment(path=path)
    result = report["variants"][variant]
    assert result["impressions"] == 1
    assert result["recommended_items"] == 3
    assert result["likes"] == 1
    assert result["watchlist_adds"] == 1
    assert result["engagement_rate"] == pytest.approx(2 / 3)


def test_outcomes_for_unshown_movies_are_ignored(tmp_path):
    path = tmp_path / "events.jsonl"
    exposure = record_exposure("profile-a", "A", [1, 2], path=path)
    record_outcome("profile-a", exposure["impression_id"], 99, "like", path=path)
    assert summarize_experiment(path=path)["variants"]["A"]["likes"] == 0


def test_outcome_from_another_profile_is_ignored(tmp_path):
    path = tmp_path / "events.jsonl"
    exposure = record_exposure("profile-a", "B", [1], path=path)
    record_outcome("profile-b", exposure["impression_id"], 1, "like", path=path)
    assert summarize_experiment(path=path)["variants"]["B"]["likes"] == 0


def test_empty_report_has_both_variants(tmp_path):
    report = summarize_experiment(path=tmp_path / "missing.jsonl")
    assert set(report["variants"]) == {"A", "B"}
    assert report["variants"]["A"]["impressions"] == 0


def test_invalid_event_line_is_reported(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text("{broken\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        summarize_experiment(path=path)
