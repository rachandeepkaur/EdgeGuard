"""Tests for part3/per_capture_metrics.py: metrics broken out per capture,
not blended into one aggregate that can hide where errors concentrate.

Run from EdgeGuard: python -m pytest tests/test_part3_per_capture_metrics.py -v
"""

import pytest

from part3.per_capture_metrics import compute_per_capture_metrics
from shared.schemas import DefenderOutput, GroundTruthLabel

THRESHOLD = 0.5


def out(window_id, score):
    decision = "ATTACK" if score >= THRESHOLD else "ACCEPT"
    return DefenderOutput(window_id=window_id, attack_score=score, threshold=THRESHOLD,
                          decision=decision, evidence="test", model_version="vtest",
                          latency_ms=1.0)


def normal(window_id):
    return GroundTruthLabel(window_id=window_id, is_attack=False)


def attack(window_id):
    return GroundTruthLabel(window_id=window_id, is_attack=True, family="speed", target="0D0")


def test_false_alarms_are_broken_out_by_capture_not_blended():
    """A single blended rate can hide that one capture (here, 'highway')
    accounts for every false alarm while another ('city') has none."""
    outputs = [out("highway_w0", 0.9), out("highway_w1", 0.1),
              out("city_w0", 0.1), out("city_w1", 0.1)]
    labels = [normal("highway_w0"), normal("highway_w1"), normal("city_w0"), normal("city_w1")]
    mapping = {"highway_w0": "highway", "highway_w1": "highway",
              "city_w0": "city", "city_w1": "city"}

    per_capture = compute_per_capture_metrics(outputs, labels, mapping)
    by_id = {c.capture_id: c.metrics for c in per_capture}

    assert by_id["highway"].fp == 1
    assert by_id["highway"].tn == 1
    assert by_id["city"].fp == 0
    assert by_id["city"].tn == 2
    # The blended aggregate (1 fp / 3 tn) would have hidden that city has zero.
    assert by_id["highway"].false_alarm_rate == pytest.approx(0.5)
    assert by_id["city"].false_alarm_rate == pytest.approx(0.0)


def test_results_are_sorted_by_capture_id():
    outputs = [out("b_w0", 0.1), out("a_w0", 0.1)]
    labels = [normal("b_w0"), normal("a_w0")]
    mapping = {"b_w0": "b", "a_w0": "a"}
    per_capture = compute_per_capture_metrics(outputs, labels, mapping)
    assert [c.capture_id for c in per_capture] == ["a", "b"]


def test_attacks_and_normals_in_the_same_capture_both_count():
    outputs = [out("cap01_w0", 0.9), out("cap01_w1", 0.1)]
    labels = [attack("cap01_w0"), normal("cap01_w1")]
    mapping = {"cap01_w0": "cap01", "cap01_w1": "cap01"}
    [capture] = compute_per_capture_metrics(outputs, labels, mapping)
    assert (capture.metrics.tp, capture.metrics.tn) == (1, 1)


def test_missing_capture_mapping_raises_with_the_window_id():
    outputs = [out("cap01_w0", 0.1)]
    labels = [normal("cap01_w0")]
    with pytest.raises(ValueError, match="cap01_w0"):
        compute_per_capture_metrics(outputs, labels, {})


def test_join_errors_from_evaluate_batch_still_propagate():
    """Duplicate window IDs must still be caught (via evaluate_batch), not
    silently swallowed by the per-capture grouping."""
    outputs = [out("cap01_w0", 0.1), out("cap01_w0", 0.1)]
    labels = [normal("cap01_w0")]
    with pytest.raises(ValueError):
        compute_per_capture_metrics(outputs, labels, {"cap01_w0": "cap01"})
