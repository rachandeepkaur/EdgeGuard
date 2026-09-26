"""Tests for selecting development misses for later Defender retraining.

Run from EdgeGuard: python -m pytest tests/test_hardening_set.py -v
"""

from dataclasses import replace

import pytest

from part3.evasion_log import EvasionEntry
from part3.hardening_set import build_hardening_set


def evasion(window_id="cap07_w0001_v01"):
    return EvasionEntry(
        attack_id="atk_001", window_id=window_id, family="freeze",
        target_can_id="0F4", start_offset_ms=0.0,
        duration_ms=1000.0, attack_score=0.2,
        model_version="v1", split="development",
    )


def test_builds_deduplicated_manifest_without_raw_frames():
    report = build_hardening_set(
        [evasion("cap07_w0002_v01"), evasion(), evasion()],
        ["cap07_w0001_v01", "cap07_w0002_v01"],
        baseline_model_version="v1",
    )
    assert report.window_ids == ("cap07_w0001_v01", "cap07_w0002_v01")
    assert report.source_split == "development"
    assert not hasattr(report, "frames")


def test_refuses_final_test_evasion():
    with pytest.raises(ValueError, match="not from development"):
        build_hardening_set(
            [replace(evasion(), split="final_test")],
            ["cap07_w0001_v01"], baseline_model_version="v1",
        )


def test_refuses_window_outside_development_manifest():
    with pytest.raises(ValueError, match="not in the development manifest"):
        build_hardening_set(
            [evasion()], ["cap99_w0001_v01"], baseline_model_version="v1"
        )


def test_refuses_mixed_baseline_versions():
    with pytest.raises(ValueError, match="different model version"):
        build_hardening_set(
            [replace(evasion(), model_version="v2")],
            ["cap07_w0001_v01"], baseline_model_version="v1",
        )


def test_empty_misses_give_empty_hardening_set():
    report = build_hardening_set([], [], baseline_model_version="v1")
    assert report.window_ids == ()
