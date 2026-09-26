"""Tests for Part 3 evaluator using mock labels and Defender outputs.

Run from EdgeGuard: python -m pytest tests/test_evaluator.py -v
"""

import pytest

from part3.evaluator import evaluate_batch, evaluate_one
from shared.schemas import DefenderOutput, GroundTruthLabel


def output(window_id="cap07_w0001", *, decision="ATTACK", score=None):
    if score is None:
        score = 0.9 if decision == "ATTACK" else 0.1
    return DefenderOutput(
        window_id=window_id,
        attack_score=score,
        threshold=0.5,
        decision=decision,
        evidence="mock evidence",
        model_version="v1",
        latency_ms=3.0,
    )


def label(window_id="cap07_w0001", *, is_attack=True):
    return GroundTruthLabel(
        window_id=window_id,
        is_attack=is_attack,
        family="freeze" if is_attack else None,
        injection_interval=[1.0, 2.0] if is_attack else None,
    )


@pytest.mark.parametrize(
    "is_attack,decision,expected",
    [
        (True, "ATTACK", "TP"),
        (True, "ACCEPT", "FN"),
        (False, "ATTACK", "FP"),
        (False, "ACCEPT", "TN"),
    ],
)
def test_four_outcomes(is_attack, decision, expected):
    result = evaluate_one(output(decision=decision), label(is_attack=is_attack))
    assert result.outcome == expected
    assert result.window_id == "cap07_w0001"
    assert result.latency_ms == 3.0


def test_joins_out_of_order_records_by_id():
    outputs = [output("cap07_w0001"), output("cap07_w0002", decision="ACCEPT")]
    labels = [label("cap07_w0002", is_attack=False), label("cap07_w0001")]
    assert [result.outcome for result in evaluate_batch(outputs, labels)] == ["TP", "TN"]


def test_rejects_mismatched_single_pair():
    with pytest.raises(ValueError, match="different window_id"):
        evaluate_one(output("cap07_w0001"), label("cap07_w0002"))


def test_rejects_duplicate_outputs():
    with pytest.raises(ValueError, match="duplicate Defender output"):
        evaluate_batch([output(), output()], [label()])


def test_rejects_duplicate_labels():
    with pytest.raises(ValueError, match="duplicate ground-truth label"):
        evaluate_batch([output()], [label(), label()])


def test_rejects_missing_label():
    with pytest.raises(ValueError, match="outputs without labels"):
        evaluate_batch([output()], [])


def test_rejects_missing_output():
    with pytest.raises(ValueError, match="labels without outputs"):
        evaluate_batch([], [label()])


def test_accepts_empty_batch():
    assert evaluate_batch([], []) == []
