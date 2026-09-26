"""Tests for recording confirmed development-set misses.

Run from EdgeGuard: python -m pytest tests/test_evasion_log.py -v
"""

import json

import pytest

from part3.attack_spec import AttackSpec
from part3.evaluator import EvaluationResult
from part3.evasion_log import append_evasion, make_evasion_entry


def spec():
    return AttackSpec(
        attack_id="atk_001", family="freeze", target_can_id="0f4",
        start_offset_ms=100, duration_ms=500,
    )


def result(outcome="FN", family="freeze"):
    return EvaluationResult(
        window_id="cap07_w0153_v01", outcome=outcome,
        attack_score=0.2, model_version="v1", family=family, latency_ms=3.0,
    )


def test_writes_confirmed_miss_without_raw_traffic(tmp_path):
    entry = make_evasion_entry(spec(), result(), split="development")
    path = tmp_path / "logs" / "evasions.jsonl"
    append_evasion(path, entry)
    saved = json.loads(path.read_text(encoding="utf-8").strip())
    assert saved["window_id"] == "cap07_w0153_v01"
    assert saved["target_can_id"] == "0F4"
    assert saved["attack_score"] == 0.2
    assert "frames" not in saved and "payload" not in saved
    assert "reasoning" not in saved


def test_rejects_final_test_miss():
    with pytest.raises(ValueError, match="development-set"):
        make_evasion_entry(spec(), result(), split="final_test")


def test_rejects_detected_attack():
    with pytest.raises(ValueError, match="false negative"):
        make_evasion_entry(spec(), result(outcome="TP"), split="development")


def test_rejects_mismatched_attack_family():
    with pytest.raises(ValueError, match="does not match"):
        make_evasion_entry(spec(), result(family="offset"), split="development")
