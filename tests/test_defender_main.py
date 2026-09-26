"""
Tests for defender/defender.py (the score_window interface).

Run from the EdgeGuard folder with:
    python -m pytest tests/test_defender_main.py -v

ALL windows here are MOCK traffic made up for testing.
"""

import pytest
from pydantic import ValidationError

import defender.defender as defender_module
from defender.defender import (
    Defender,
    score_window,
    set_active_defender,
    stage1_path,
    threshold_path,
)
from defender.stage1 import Stage1Model
from defender.threshold import choose_threshold, save_threshold
from shared.schemas import DefenderOutput, TrafficWindow

MOCK_PERIODS = {"0D0": 0.010, "0F4": 0.020, "1A0": 0.050}


def make_normal_window(index: int) -> TrafficWindow:
    """MOCK 1-second window with small timing jitter."""
    start = 1000.0 + index
    frames = []
    for can_id, period in MOCK_PERIODS.items():
        for k in range(int(round(1.0 / period))):
            jitter = (((index * 7 + k * 3) % 5) - 2) * 0.0002
            frames.append({"timestamp": round(start + 0.001 + k * period + jitter, 6),
                           "can_id": can_id, "payload": "0011223344556677"})
    frames.sort(key=lambda f: f["timestamp"])
    return TrafficWindow(window_id=f"cap01_w{index:04d}", capture_id="cap01",
                         window_start=start, window_end=start + 1.0, frames=frames)


def make_fuzzing_window() -> TrafficWindow:
    """MOCK attack: copy of a normal window plus frames from unknown ID 7FF."""
    data = make_normal_window(200).model_dump()
    data["frames"] += [{"timestamp": 1200.1 + k * 0.05, "can_id": "7FF",
                        "payload": "FFFFFFFFFFFFFFFF"} for k in range(10)]
    data["frames"].sort(key=lambda f: f["timestamp"])
    return TrafficWindow(**data)


@pytest.fixture(scope="module")
def trained():
    """MOCK training: fit on 'train' windows, threshold on 'validation' windows."""
    stage1 = Stage1Model().fit([make_normal_window(i) for i in range(40)])
    validation = [stage1.score(make_normal_window(i)).score for i in range(100, 130)]
    threshold = choose_threshold(validation, 0.0)
    return stage1, threshold


@pytest.fixture
def defender(trained):
    stage1, threshold = trained
    return Defender(stage1, threshold, "v1")


# ---------- Output format ---------------------------------------------
def test_returns_valid_defender_output(defender):
    out = defender.score_window(make_normal_window(120))
    assert isinstance(out, DefenderOutput)
    assert out.window_id == "cap01_w0120"
    assert out.model_version == "v1"


def test_threshold_in_output_is_the_loaded_one(defender, trained):
    assert defender.score_window(make_normal_window(121)).threshold == trained[1]


def test_latency_is_measured(defender):
    out = defender.score_window(make_normal_window(122))
    assert out.latency_ms is not None and out.latency_ms >= 0.0


# ---------- Decisions --------------------------------------------------
def test_normal_window_is_accepted(defender):
    out = defender.score_window(make_normal_window(123))
    assert out.decision == "ACCEPT"


def test_fuzzing_window_is_attack(defender):
    out = defender.score_window(make_fuzzing_window())
    assert out.decision == "ATTACK"
    assert "7FF" in out.evidence


# ---------- Input handling --------------------------------------------
def test_accepts_plain_dict(defender):
    out = defender.score_window(make_normal_window(124).model_dump())
    assert out.window_id == "cap01_w0124"


def test_dict_with_hidden_label_is_rejected(defender):
    data = make_normal_window(125).model_dump()
    data["is_attack"] = True
    with pytest.raises(ValidationError):
        defender.score_window(data)


def test_rejects_other_types(defender):
    with pytest.raises(TypeError, match="TrafficWindow or dict"):
        defender.score_window("cap01_w0001")


def test_does_not_modify_window(defender):
    w = make_fuzzing_window()
    before = w.model_dump()
    defender.score_window(w)
    assert w.model_dump() == before


# ---------- Construction checks ---------------------------------------
def test_rejects_untrained_stage1():
    with pytest.raises(ValueError, match="trained Stage1Model"):
        Defender(Stage1Model(), 0.5, "v1")


def test_rejects_threshold_outside_zero_to_one(trained):
    with pytest.raises(ValueError):
        Defender(trained[0], 1.5, "v1")


def test_rejects_empty_model_version(trained):
    with pytest.raises(ValueError, match="model_version"):
        Defender(trained[0], trained[1], "")


# ---------- Loading from files ----------------------------------------
def test_load_from_files_gives_same_results(trained, tmp_path):
    stage1, threshold = trained
    stage1.save(stage1_path(tmp_path, "v1"), "v1")
    save_threshold(threshold, "v1", threshold_path(tmp_path, "v1"))
    loaded = Defender.load(tmp_path, "v1")
    direct = Defender(stage1, threshold, "v1")
    for w in [make_normal_window(126), make_fuzzing_window()]:
        a, b = loaded.score_window(w), direct.score_window(w)
        assert (a.attack_score, a.decision, a.threshold) == (b.attack_score, b.decision, b.threshold)


def test_load_refuses_threshold_from_other_version(trained, tmp_path):
    stage1, threshold = trained
    stage1.save(stage1_path(tmp_path, "v1"), "v1")
    save_threshold(threshold, "v2", threshold_path(tmp_path, "v1"))   # wrong version inside
    with pytest.raises(ValueError, match="Model version mismatch"):
        Defender.load(tmp_path, "v1")


def test_load_missing_files_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        Defender.load(tmp_path, "v1")


# ---------- Module-level score_window() for Part 3 --------------------
def test_score_window_without_active_defender_gives_clear_error(monkeypatch):
    monkeypatch.setattr(defender_module, "_active_defender", None)
    with pytest.raises(RuntimeError, match="No Defender loaded"):
        score_window(make_normal_window(127))


def test_score_window_uses_active_defender(defender, monkeypatch):
    monkeypatch.setattr(defender_module, "_active_defender", None)
    set_active_defender(defender)
    assert score_window(make_fuzzing_window()).decision == "ATTACK"


def test_set_active_defender_rejects_wrong_type():
    with pytest.raises(TypeError):
        set_active_defender("not a defender")

# ---------- Window length guard ---------------------------------------
def test_refuses_windows_cut_differently_from_training(trained):
    stage1, threshold = trained
    guarded = Defender(stage1, threshold, "v1", window_s=1.0)
    assert guarded.score_window(make_normal_window(120)).window_id == "cap01_w0120"
    half = make_normal_window(121).model_dump()
    half["window_end"] = half["window_start"] + 0.5
    half["frames"] = [f for f in half["frames"] if f["timestamp"] <= half["window_end"]]
    with pytest.raises(ValueError, match="trained on 1.0 s windows"):
        guarded.score_window(half)


def test_load_reads_window_length_from_train_info(trained, tmp_path):
    import json
    stage1, threshold = trained
    stage1.save(tmp_path / "stage1_v1.json", "v1")
    save_threshold(threshold, "v1", tmp_path / "threshold_v1.json")
    assert Defender.load(tmp_path, "v1").window_s is None          # no train_info: no guard
    (tmp_path / "train_info_v1.json").write_text(json.dumps({"window_s": 1.0}))
    assert Defender.load(tmp_path, "v1").window_s == 1.0
