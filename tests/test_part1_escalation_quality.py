"""Tests for part1/escalation_quality.py: does escalation target real mistakes?

Hand-built outputs and labels only -- no dataset, no model, no cloud.

Run from EdgeGuard: python -m pytest tests/test_part1_escalation_quality.py -v
"""

import pytest

from part1.escalation_quality import compute_escalation_quality
from shared.schemas import DefenderOutput, GroundTruthLabel

THRESHOLD = 0.5


def out(window_id, score, threshold=THRESHOLD):
    return DefenderOutput(
        window_id=window_id, attack_score=score, threshold=threshold,
        decision="ATTACK" if score >= threshold else "ACCEPT",
        evidence="test", model_version="vtest", latency_ms=1.0,
    )


def label(window_id, is_attack):
    if is_attack:
        return GroundTruthLabel(window_id=window_id, is_attack=True,
                                family="offset", target="0D0")
    return GroundTruthLabel(window_id=window_id, is_attack=False)


# band 0.1 around 0.5 -> [0.4, 0.6]
CASES = [
    # id,  score, is_attack      local outcome   in band?
    ("a", 0.90, True),         # TP              no
    ("b", 0.10, False),        # TN              no
    ("c", 0.55, False),        # FP              yes
    ("d", 0.45, True),         # FN              yes
    ("e", 0.58, True),         # TP              yes
    ("f", 0.20, True),         # FN              no  <- a confident miss
]


def build(cases):
    return ([out(i, s) for i, s, _ in cases], [label(i, a) for i, _, a in cases])


def test_counts_precision_recall_and_error_rates():
    outputs, labels = build(CASES)
    q = compute_escalation_quality(outputs, labels, 0.1)

    assert q.windows == 6
    assert q.escalations == 3
    assert q.escalation_rate == pytest.approx(0.5)
    assert (q.local_errors, q.local_false_positives, q.local_false_negatives) == (3, 1, 2)
    assert (q.escalated_errors, q.escalated_false_positives,
            q.escalated_false_negatives) == (2, 1, 1)
    assert q.escalation_precision == pytest.approx(2 / 3)
    assert q.escalation_recall == pytest.approx(2 / 3)
    assert q.error_rate_outside_band == pytest.approx(1 / 3)   # only "f"
    assert q.local_accuracy == pytest.approx(3 / 6)
    # A perfect second opinion fixes c and d; the confident miss f stays wrong.
    assert q.local_plus_perfect_cloud_accuracy == pytest.approx(5 / 6)


def test_bundle_bytes_are_measured_on_escalated_windows_only():
    outputs, labels = build(CASES)
    q = compute_escalation_quality(outputs, labels, 0.1)
    assert q.bundle_bytes_total > 0
    assert q.bundle_bytes_mean == pytest.approx(q.bundle_bytes_total / 3)
    assert q.bundle_bytes_max >= q.bundle_bytes_mean


def test_bundle_fields_are_exactly_the_sanitizer_allow_list():
    """If anyone ever adds a raw-data field to SanitizedBundle, this fails:
    the report's 'zero raw CAN bytes' claim rests on this exact list."""
    outputs, labels = build(CASES)
    q = compute_escalation_quality(outputs, labels, 0.1)
    assert q.bundle_fields == ["attack_score", "decision", "evidence",
                               "model_version", "threshold", "window_id"]


def test_zero_band_escalates_nothing_and_reports_none_not_zero():
    outputs, labels = build(CASES)
    q = compute_escalation_quality(outputs, labels, 0.0)
    assert q.escalations == 0
    assert q.escalation_precision is None       # undefined, not 0
    assert q.escalation_recall == pytest.approx(0.0)
    assert q.bundle_bytes_mean is None and q.bundle_bytes_max is None
    assert q.bundle_bytes_total == 0
    assert q.local_plus_perfect_cloud_accuracy == pytest.approx(q.local_accuracy)


def test_no_local_errors_makes_recall_undefined():
    cases = [("a", 0.9, True), ("b", 0.1, False), ("c", 0.55, True)]
    outputs, labels = build(cases)
    q = compute_escalation_quality(outputs, labels, 0.1)
    assert q.local_errors == 0
    assert q.escalation_recall is None
    assert q.escalation_precision == pytest.approx(0.0)


def test_off_center_threshold_band_is_clipped_at_one():
    """Real models sit near threshold 1.0 (v2 is ~0.999): the band's top is
    clipped, so every ATTACK decision falls inside it."""
    outputs = [out("x", 0.999, threshold=0.95), out("y", 0.5, threshold=0.95)]
    labels = [label("x", False), label("y", False)]
    q = compute_escalation_quality(outputs, labels, 0.1)
    assert q.escalations == 1
    assert q.escalated_false_positives == 1


def test_rejects_empty_missing_and_duplicate_records():
    outputs, labels = build(CASES)
    with pytest.raises(ValueError):
        compute_escalation_quality([], [], 0.1)
    with pytest.raises(ValueError):
        compute_escalation_quality(outputs, labels[:-1], 0.1)          # missing label
    with pytest.raises(ValueError):
        compute_escalation_quality(outputs + outputs[:1], labels, 0.1)  # duplicate output


def test_rejects_invalid_band():
    outputs, labels = build(CASES)
    with pytest.raises(ValueError):
        compute_escalation_quality(outputs, labels, 0.6)
