"""Tests for paired final-test comparison; all examples are MOCK data.

Run from EdgeGuard: python -m pytest tests/test_evidence_gate.py -v
"""

import pytest

from part3.evidence_gate import compare_final_models
from shared.schemas import DefenderOutput, GroundTruthLabel


IDS = ["cap90_w0001", "cap91_w0001", "cap92_w0001", "cap93_w0001"]


def answer(window_id, *, version, attack):
    return DefenderOutput(
        window_id=window_id, attack_score=0.9 if attack else 0.1,
        threshold=0.5, decision="ATTACK" if attack else "ACCEPT",
        evidence="mock", model_version=version, latency_ms=2.0,
    )


def labels():
    return [
        GroundTruthLabel(window_id=window_id, is_attack=is_attack,
                         family="freeze" if is_attack else None)
        for window_id, is_attack in zip(IDS, [True, True, False, False])
    ]


def inputs():
    baseline = [answer(window_id, version="v1", attack=positive)
                for window_id, positive in zip(IDS, [True, False, True, False])]
    updated = [answer(window_id, version="v2", attack=positive)
               for window_id, positive in zip(IDS, [True, True, False, False])]
    return baseline, updated


def compare(baseline=None, updated=None, hidden_labels=None,
            final_ids=None, training_ids=()):
    default_baseline, default_updated = inputs()
    return compare_final_models(
        baseline if baseline is not None else default_baseline,
        updated if updated is not None else default_updated,
        hidden_labels if hidden_labels is not None else labels(),
        final_test_window_ids=final_ids if final_ids is not None else IDS,
        training_window_ids=training_ids,
    )


def test_reports_paired_baseline_and_updated_metrics():
    report = compare()
    assert report.source_split == "final_test"
    assert (report.baseline_version, report.updated_version) == ("v1", "v2")
    assert report.baseline.recall == 0.5
    assert report.baseline.false_alarm_rate == 0.5
    assert report.updated.recall == 1.0
    assert report.updated.false_alarm_rate == 0.0


def test_rejects_training_overlap():
    with pytest.raises(ValueError, match="overlap"):
        compare(training_ids=[IDS[0]])


def test_rejects_missing_updated_output():
    baseline, updated = inputs()
    with pytest.raises(ValueError, match="do not match"):
        compare(baseline=baseline, updated=updated[:-1])


def test_rejects_missing_hidden_label():
    with pytest.raises(ValueError, match="labels must match"):
        compare(hidden_labels=labels()[:-1])


def test_rejects_reusing_baseline_version():
    baseline, _ = inputs()
    with pytest.raises(ValueError, match="versions must differ"):
        compare(baseline=baseline, updated=baseline)


def test_rejects_duplicate_final_test_ids():
    with pytest.raises(ValueError, match="nonempty and unique"):
        compare(final_ids=IDS + [IDS[0]])
