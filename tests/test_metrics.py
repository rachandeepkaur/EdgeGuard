"""Tests for window-level metrics using made-up evaluation results.

Run from EdgeGuard: python -m pytest tests/test_metrics.py -v
"""

from typing import Optional

import pytest

from part3.evaluator import EvaluationResult
from part3.metrics import compute_metrics


def result(number: int, outcome: str, latency_ms: Optional[float] = None):
    return EvaluationResult(
        window_id=f"cap07_w{number:04d}",
        outcome=outcome,
        attack_score=0.9 if outcome in ("TP", "FP") else 0.1,
        model_version="v1",
        family="freeze" if outcome in ("TP", "FN") else None,
        latency_ms=latency_ms,
    )


def test_counts_and_rates_on_mixed_results():
    report = compute_metrics(
        [
            result(1, "TP", 2.0),
            result(2, "TP", 4.0),
            result(3, "FN", None),
            result(4, "FP", 6.0),
            result(5, "TN", None),
            result(6, "TN", None),
            result(7, "TN", None),
        ]
    )
    assert (report.windows, report.tp, report.fp, report.tn, report.fn) == (7, 2, 1, 3, 1)
    assert report.recall == pytest.approx(2 / 3)
    assert report.false_alarm_rate == pytest.approx(1 / 4)
    assert report.precision == pytest.approx(2 / 3)
    assert report.f1 == pytest.approx(2 / 3)
    assert report.mean_inference_ms == pytest.approx(4.0)


def test_empty_results_have_undefined_rates():
    report = compute_metrics([])
    assert report.windows == 0
    assert report.recall is None
    assert report.false_alarm_rate is None
    assert report.precision is None
    assert report.f1 is None
    assert report.mean_inference_ms is None


def test_normal_only_has_no_recall():
    report = compute_metrics([result(1, "FP"), result(2, "TN")])
    assert report.recall is None
    assert report.false_alarm_rate == 0.5


def test_attack_only_has_no_false_alarm_rate():
    report = compute_metrics([result(1, "TP"), result(2, "FN")])
    assert report.recall == 0.5
    assert report.false_alarm_rate is None


def test_all_attacks_missed_has_zero_recall_but_undefined_precision():
    report = compute_metrics([result(1, "FN"), result(2, "FN")])
    assert report.recall == 0.0
    assert report.precision is None
    assert report.f1 == 0.0


def test_rejects_duplicate_window_results():
    with pytest.raises(ValueError, match="duplicate evaluation result"):
        compute_metrics([result(1, "TP"), result(1, "FN")])
