"""
Tests for defender/stage1.py.

Run from the EdgeGuard folder with:
    python -m pytest tests/test_defender_stage1.py -v

ALL windows here are MOCK traffic made up for testing. They imitate the
shape of ROAD data (periodic IDs, ~1 s windows) but are not real data,
so these tests prove the logic works, not how well it does on ROAD.
"""

import pytest

from defender.stage1 import Stage1Model
from defender.threshold import ATTACK, ACCEPT, choose_threshold, make_decision
from shared.schemas import TrafficWindow

# MOCK normal traffic: ID -> period in seconds
MOCK_PERIODS = {"0D0": 0.010, "0F4": 0.020, "1A0": 0.050}


def make_normal_window(index: int) -> TrafficWindow:
    """MOCK 1-second window with small, varying timing jitter."""
    start = 1000.0 + index
    frames = []
    for can_id, period in MOCK_PERIODS.items():
        count = int(round(1.0 / period))
        for k in range(count):
            jitter = (((index * 7 + k * 3) % 5) - 2) * 0.0002   # -0.4 .. +0.4 ms
            t = start + 0.001 + k * period + jitter
            frames.append({"timestamp": round(t, 6), "can_id": can_id,
                           "payload": "0011223344556677"})
    frames.sort(key=lambda f: f["timestamp"])
    return TrafficWindow(window_id=f"cap01_w{index:04d}", capture_id="cap01",
                         window_start=start, window_end=start + 1.0, frames=frames)


def with_extra_frames(window: TrafficWindow, extra: list) -> TrafficWindow:
    """MOCK attack: return a COPY with extra frames added."""
    data = window.model_dump()
    data["frames"] = sorted(data["frames"] + extra, key=lambda f: f["timestamp"])
    return TrafficWindow(**data)


@pytest.fixture(scope="module")
def model():
    return Stage1Model().fit([make_normal_window(i) for i in range(40)])


@pytest.fixture(scope="module")
def normal_validation_scores(model):
    return [model.score(make_normal_window(i)).score for i in range(100, 130)]


# ---------- MOCK attacks ----------------------------------------------
def fuzzing_window():
    w = make_normal_window(200)
    extra = [{"timestamp": w.window_start + 0.1 + k * 0.05, "can_id": "7FF",
              "payload": "FFFFFFFFFFFFFFFF"} for k in range(10)]
    return with_extra_frames(w, extra)


def fabrication_window():
    """Extra 0D0 frames halfway between the real ones (double rate)."""
    w = make_normal_window(201)
    extra = [{"timestamp": round(w.window_start + 0.006 + k * 0.010, 6), "can_id": "0D0",
              "payload": "DEADBEEF00000000"} for k in range(99)]
    return with_extra_frames(w, extra)


def single_injection_window():
    """One extra 1A0 frame 0.1 ms after a real one: tiny rate change, very short gap."""
    w = make_normal_window(202)
    first_1a0 = next(f for f in w.frames if f.can_id == "1A0")
    extra = [{"timestamp": round(first_1a0.timestamp + 0.0001, 6), "can_id": "1A0",
              "payload": "DEADBEEF00000000"}]
    return with_extra_frames(w, extra)


def masquerade_window():
    """Same timing, different payloads: Stage 1 is NOT expected to see this."""
    w = make_normal_window(203)
    data = w.model_dump()
    for frame in data["frames"]:
        if frame["can_id"] == "0D0":
            frame["payload"] = "FFFF000000000000"
    return TrafficWindow(**data)


# ---------- Normal behaviour ------------------------------------------
def test_scores_are_between_zero_and_one(model):
    for w in [make_normal_window(300), fuzzing_window(), fabrication_window()]:
        result = model.score(w)
        assert 0.0 <= result.score < 1.0
        assert set(result.sub_scores) == {"unknown_ids", "rate_excess", "short_gaps"}


def test_normal_window_has_no_timing_evidence(model):
    result = model.score(make_normal_window(5))       # a training window
    assert "No timing anomaly" in result.evidence


def test_scoring_is_deterministic(model):
    w = fabrication_window()
    assert model.score(w).score == model.score(w).score


# ---------- Attacks score above all normal windows --------------------
def test_fuzzing_detected(model, normal_validation_scores):
    result = model.score(fuzzing_window())
    assert result.score > max(normal_validation_scores)
    assert "7FF" in result.evidence
    assert "unknown_ids" in result.evidence


def test_fabrication_detected(model, normal_validation_scores):
    result = model.score(fabrication_window())
    assert result.score > max(normal_validation_scores)
    assert "0D0" in result.evidence


def test_single_injection_caught_by_short_gap(model, normal_validation_scores):
    result = model.score(single_injection_window())
    assert result.score > max(normal_validation_scores)
    assert "short_gaps" in result.evidence
    assert "1A0" in result.evidence


def test_masquerade_is_a_known_limit(model):
    """Payload-only change keeps timing identical -> same Stage 1 score."""
    original = make_normal_window(203)
    assert model.score(masquerade_window()).score == model.score(original).score


# ---------- Works with threshold.py -----------------------------------
def test_end_to_end_with_threshold(model, normal_validation_scores):
    threshold = choose_threshold(normal_validation_scores, 0.0)
    for s in normal_validation_scores:
        assert make_decision(s, threshold) == ACCEPT
    for w in [fuzzing_window(), fabrication_window(), single_injection_window()]:
        assert make_decision(model.score(w).score, threshold) == ATTACK


# ---------- Safety -----------------------------------------------------
def test_scoring_does_not_modify_window(model):
    w = fuzzing_window()
    before = w.model_dump()
    model.score(w)
    assert w.model_dump() == before


def test_lowercase_ids_match_uppercase(model):
    w = make_normal_window(301)
    data = w.model_dump()
    for frame in data["frames"]:
        frame["can_id"] = frame["can_id"].lower()
    lower = TrafficWindow(**data)
    assert model.score(lower).score == model.score(w).score


# ---------- Invalid use ------------------------------------------------
def test_score_before_fit_gives_clear_error():
    with pytest.raises(RuntimeError, match="not trained"):
        Stage1Model().score(make_normal_window(0))


def test_fit_rejects_empty_list():
    with pytest.raises(ValueError, match="at least 2"):
        Stage1Model().fit([])


def test_fit_rejects_non_window():
    with pytest.raises(TypeError, match="expected TrafficWindow"):
        Stage1Model().fit([make_normal_window(0), {"frames": []}])


def test_score_rejects_dictionary(model):
    with pytest.raises(TypeError, match="needs a TrafficWindow"):
        model.score({"frames": []})


def test_zero_length_window_rejected(model):
    w = TrafficWindow(window_id="cap01_w9999", capture_id="cap01",
                      window_start=5.0, window_end=5.0, frames=[])
    with pytest.raises(ValueError, match="duration"):
        model.score(w)


# ---------- Save / load -----------------------------------------------
def test_save_and_load_give_identical_scores(model, tmp_path):
    path = tmp_path / "stage1_v1.json"
    model.save(path, "v1")
    loaded = Stage1Model.load(path, "v1")
    for w in [make_normal_window(400), fuzzing_window(), fabrication_window()]:
        assert loaded.score(w).score == model.score(w).score


def test_load_refuses_other_model_version(model, tmp_path):
    path = tmp_path / "stage1_v1.json"
    model.save(path, "v1")
    with pytest.raises(ValueError, match="Model version mismatch"):
        Stage1Model.load(path, "v2")


def test_cannot_save_untrained_model(tmp_path):
    with pytest.raises(RuntimeError, match="untrained"):
        Stage1Model().save(tmp_path / "x.json", "v1")