"""Tests for part1/escalation_policy.py.

Run from EdgeGuard: python -m pytest tests/test_escalation_policy.py -v
"""

import pytest

from part1.escalation_policy import choose_band_half_width, should_escalate
from shared.schemas import DefenderOutput


def output(score, threshold=0.5, decision="ATTACK"):
    return DefenderOutput(window_id="cap07_w0001", attack_score=score,
                          threshold=threshold, decision=decision,
                          evidence="x", model_version="v1")


# ---------- should_escalate ---------------------------------------------
def test_score_inside_band_escalates():
    result = should_escalate(output(0.52), band_half_width=0.05)
    assert result.escalate
    assert (result.band_low, result.band_high) == (pytest.approx(0.45), pytest.approx(0.55))


def test_score_confidently_above_band_does_not_escalate():
    assert not should_escalate(output(0.95), band_half_width=0.05).escalate


def test_score_confidently_below_band_does_not_escalate():
    assert not should_escalate(output(0.05, decision="ACCEPT"), band_half_width=0.05).escalate


def test_score_on_band_edge_escalates():
    """Inclusive boundary: a score exactly at threshold + band still escalates."""
    assert should_escalate(output(0.55), band_half_width=0.05).escalate


def test_band_clips_at_zero_and_one():
    """Threshold near an edge must not produce an out-of-range band."""
    result = should_escalate(output(0.02, threshold=0.05, decision="ACCEPT"), band_half_width=0.2)
    assert result.band_low == 0.0   # would be -0.15 unclipped


def test_zero_width_band_only_matches_exact_threshold():
    assert should_escalate(output(0.5), band_half_width=0.0).escalate
    assert not should_escalate(output(0.500001), band_half_width=0.0).escalate


def test_rejects_out_of_range_band_width():
    with pytest.raises(ValueError):
        should_escalate(output(0.5), band_half_width=0.6)


def test_rejects_negative_band_width():
    with pytest.raises(ValueError):
        should_escalate(output(0.5), band_half_width=-0.1)


# ---------- choose_band_half_width --------------------------------------
def test_wider_budget_gives_wider_band():
    scores = [0.1, 0.3, 0.45, 0.5, 0.55, 0.7, 0.9]
    narrow = choose_band_half_width(scores, threshold=0.5, max_escalation_rate=1/7)
    wide = choose_band_half_width(scores, threshold=0.5, max_escalation_rate=3/7)
    assert wide > narrow


def test_full_budget_returns_widest_band():
    scores = [0.1, 0.3, 0.7, 0.9]
    assert choose_band_half_width(scores, threshold=0.5, max_escalation_rate=1.0) == 0.5


def test_chosen_band_never_exceeds_the_budget():
    """The actual escalation count at the returned band must never exceed
    what max_escalation_rate allows, checked directly, not assumed."""
    scores = [0.1, 0.3, 0.45, 0.5, 0.55, 0.7, 0.9]
    for rate in (1/7, 2/7, 3/7, 4/7, 1.0):
        band = choose_band_half_width(scores, threshold=0.5, max_escalation_rate=rate)
        count = sum(1 for s in scores if 0.5 - band <= s <= 0.5 + band)
        allowed = int(rate * len(scores))
        assert count <= allowed, f"rate={rate}: count {count} exceeds allowed {allowed}"


def test_raises_when_a_score_sits_exactly_on_threshold_and_budget_is_zero():
    """Zero escalations is impossible if a normal score is exactly at the
    threshold -- even the tightest band (width 0) still catches it. This
    must be a clear error, not a silently-wrong 0.0."""
    with pytest.raises(ValueError, match="cannot achieve"):
        choose_band_half_width([0.3, 0.5, 0.7], threshold=0.5, max_escalation_rate=0.0)


def test_rejects_empty_scores():
    with pytest.raises(ValueError, match="empty"):
        choose_band_half_width([], threshold=0.5, max_escalation_rate=0.1)


def test_rejects_out_of_range_score_in_list():
    with pytest.raises(ValueError):
        choose_band_half_width([0.5, 1.5], threshold=0.5, max_escalation_rate=0.1)


def test_returned_band_is_always_a_valid_should_escalate_width_even_off_center():
    """Regression: with a threshold away from 0.5 (the normal case -- a
    real Defender threshold is rarely exactly 0.5), |score - threshold|
    can exceed 0.5, and an unclipped candidate could leak that value out
    as band_half_width -- which should_escalate() then rejects outright.
    Every score here is far below a high threshold, so the naive
    (unclipped) candidate set would have picked something well past 0.5."""
    scores = [0.0, 0.1, 0.2, 0.3]
    band = choose_band_half_width(scores, threshold=0.9, max_escalation_rate=1.0)
    assert 0.0 <= band <= 0.5
    should_escalate(output(0.1, threshold=0.9, decision="ACCEPT"), band_half_width=band)
