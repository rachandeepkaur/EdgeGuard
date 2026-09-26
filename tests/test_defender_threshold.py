"""
Tests for defender/threshold.py.

Run from the EdgeGuard folder with:
    python -m pytest tests/test_defender_threshold.py -v

All scores here are MOCK numbers made up for testing.
"""

import pytest

from defender.threshold import (
    ACCEPT,
    ATTACK,
    choose_threshold,
    load_threshold,
    make_decision,
    save_threshold,
)
from shared.schemas import DefenderOutput

MOCK_NORMAL_SCORES = [0.05, 0.10, 0.10, 0.20, 0.30, 0.35, 0.40, 0.50, 0.60, 0.90]


def false_alarm_rate(scores, threshold):
    alarms = sum(1 for s in scores if make_decision(s, threshold) == ATTACK)
    return alarms / len(scores)


# ---------- Correct ATTACK / ACCEPT decisions -------------------------
def test_score_above_threshold_is_attack():
    assert make_decision(0.83, 0.70) == ATTACK


def test_score_equal_to_threshold_is_attack():
    assert make_decision(0.70, 0.70) == ATTACK


def test_score_below_threshold_is_accept():
    assert make_decision(0.69, 0.70) == ACCEPT


def test_rule_matches_shared_schema():
    """make_decision and DefenderOutput must never disagree."""
    for score in [0.0, 0.3, 0.6999, 0.7, 0.71, 1.0]:
        decision = make_decision(score, 0.70)
        DefenderOutput(window_id="cap07_w0001", attack_score=score, threshold=0.70,
                       decision=decision, evidence="MOCK", model_version="v1")


# ---------- Valid threshold selection ---------------------------------
def test_ten_percent_rate_is_respected():
    t = choose_threshold(MOCK_NORMAL_SCORES, 0.10)
    assert false_alarm_rate(MOCK_NORMAL_SCORES, t) <= 0.10


def test_threshold_is_the_lowest_possible():
    """Any lower threshold would break the target."""
    t = choose_threshold(MOCK_NORMAL_SCORES, 0.10)
    lower = t - 1e-9
    assert false_alarm_rate(MOCK_NORMAL_SCORES, lower) > 0.10


def test_ties_do_not_add_false_alarms():
    scores = [0.5, 0.5, 0.5, 0.1]
    t = choose_threshold(scores, 0.25)
    assert false_alarm_rate(scores, t) <= 0.25


def test_rate_of_one_allows_threshold_zero():
    assert choose_threshold(MOCK_NORMAL_SCORES, 1.0) == 0.0


# ---------- Zero false-alarm rate --------------------------------------
def test_zero_rate_is_strictly_above_highest_normal_score():
    t = choose_threshold(MOCK_NORMAL_SCORES, 0.0)
    assert t > max(MOCK_NORMAL_SCORES)
    assert false_alarm_rate(MOCK_NORMAL_SCORES, t) == 0.0


def test_zero_rate_impossible_when_normal_score_is_one():
    with pytest.raises(ValueError, match="Cannot meet the target"):
        choose_threshold([1.0, 0.2], 0.0)


# ---------- Empty and invalid inputs ----------------------------------
def test_empty_score_list_rejected():
    with pytest.raises(ValueError, match="empty"):
        choose_threshold([], 0.01)


@pytest.mark.parametrize("bad", [-0.1, 1.5, float("nan")])
def test_invalid_score_in_list_rejected(bad):
    with pytest.raises(ValueError, match=r"normal_scores\[1\]"):
        choose_threshold([0.1, bad], 0.0)


def test_text_score_rejected():
    with pytest.raises(TypeError):
        make_decision("high", 0.70)


@pytest.mark.parametrize("bad_rate", [-0.01, 1.01, float("nan")])
def test_invalid_false_alarm_rate_rejected(bad_rate):
    with pytest.raises(ValueError, match="max_false_alarm_rate"):
        choose_threshold(MOCK_NORMAL_SCORES, bad_rate)


def test_missing_false_alarm_rate_has_no_default():
    with pytest.raises(TypeError):
        choose_threshold(MOCK_NORMAL_SCORES)          # rate must be supplied


@pytest.mark.parametrize("bad_threshold", [-0.1, 1.1])
def test_invalid_threshold_rejected(bad_threshold, tmp_path):
    with pytest.raises(ValueError, match="threshold"):
        make_decision(0.5, bad_threshold)
    with pytest.raises(ValueError, match="threshold"):
        save_threshold(bad_threshold, "v1", tmp_path / "t.json")


# ---------- Save / load with model version ----------------------------
def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "threshold_v1.json"
    save_threshold(0.61, "v1", path)
    assert load_threshold(path, "v1") == 0.61


def test_model_version_mismatch_refused(tmp_path):
    path = tmp_path / "threshold_v1.json"
    save_threshold(0.61, "v1", path)
    with pytest.raises(ValueError, match="Model version mismatch"):
        load_threshold(path, "v2")


def test_missing_threshold_file_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="No threshold file"):
        load_threshold(tmp_path / "missing.json", "v1")