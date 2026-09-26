"""Tests for part1/egress_metrics.py.

Run from EdgeGuard: python -m pytest tests/test_egress_metrics.py -v
"""

import pytest

from part1.egress_metrics import EscalationOutcome, compute_egress_report


def test_escalation_rate_and_latency():
    outcomes = [
        EscalationOutcome("w1", escalated=False),
        EscalationOutcome("w2", escalated=False),
        EscalationOutcome("w3", escalated=True, added_latency_ms=20.0, corrected=False),
        EscalationOutcome("w4", escalated=True, added_latency_ms=30.0, corrected=True),
    ]
    report = compute_egress_report(outcomes)
    assert report.windows == 4
    assert report.escalations == 2
    assert report.escalation_rate == pytest.approx(0.5)
    assert report.mean_added_latency_ms == pytest.approx(25.0)


def test_failed_calls_excluded_from_correction_rate():
    """A failed call must not be silently treated as 'confirmed' -- it
    should count toward failed_calls, not toward correction_rate."""
    outcomes = [
        EscalationOutcome("w1", escalated=True, added_latency_ms=20.0, corrected=False),
        EscalationOutcome("w2", escalated=True, added_latency_ms=30.0, corrected=True),
        EscalationOutcome("w3", escalated=True, added_latency_ms=1000.0, corrected=None),
    ]
    report = compute_egress_report(outcomes)
    assert report.failed_calls == 1
    assert report.corrections == 1
    assert report.correction_rate == pytest.approx(0.5)   # 1 of 2 ANSWERED calls, not 3


def test_no_escalations_gives_none_metrics_not_zero():
    outcomes = [EscalationOutcome("w1", escalated=False), EscalationOutcome("w2", escalated=False)]
    report = compute_egress_report(outcomes)
    assert report.escalation_rate == 0.0   # a real, meaningful zero
    assert report.mean_added_latency_ms is None   # not measurable, not zero
    assert report.correction_rate is None


def test_empty_input():
    report = compute_egress_report([])
    assert report.windows == 0
    assert report.escalation_rate == 0.0


# ---------- EscalationOutcome consistency guards --------------------------
def test_rejects_latency_on_non_escalated_window():
    with pytest.raises(ValueError, match="must not carry"):
        EscalationOutcome("w1", escalated=False, added_latency_ms=10.0)


def test_rejects_corrected_flag_on_non_escalated_window():
    with pytest.raises(ValueError, match="must not carry"):
        EscalationOutcome("w1", escalated=False, corrected=True)


def test_escalated_window_requires_latency():
    with pytest.raises(ValueError, match="must record added_latency_ms"):
        EscalationOutcome("w1", escalated=True, corrected=True)


def test_rejects_negative_latency():
    with pytest.raises(ValueError):
        EscalationOutcome("w1", escalated=True, added_latency_ms=-5.0, corrected=True)
