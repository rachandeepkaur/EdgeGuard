"""
Tests for defender/defender.py v2 support (Stage 1 + Stage 2 fusion).

Run from the EdgeGuard folder with:
    python -m pytest tests/test_defender_main_v2.py -v

ALL windows here are MOCK traffic made up for testing.
"""

import pytest

from defender.defender import Defender, stage1_path, stage2_path, threshold_path
from defender.stage1 import Stage1Model
from defender.stage2 import Stage2Model
from defender.threshold import choose_threshold, save_threshold
from shared.schemas import TrafficWindow

TIMING_PERIODS = {"0D0": 0.010, "0F4": 0.020, "1A0": 0.050}


def make_normal_window(index: int) -> TrafficWindow:
    """MOCK 1-second window: normal timing AND normal, varying payloads."""
    start = 1000.0 + index
    frames = []
    for can_id, period in TIMING_PERIODS.items():
        for k in range(int(round(1.0 / period))):
            jitter = (((index * 7 + k * 3) % 5) - 2) * 0.0002
            byte0 = k % 200          # varies; independent of index, so train and
                                     # validation windows wrap the same way
            frames.append({"timestamp": round(start + 0.001 + k * period + jitter, 6),
                           "can_id": can_id, "payload": f"{byte0:02X}" + "00" * 7})
    frames.sort(key=lambda f: f["timestamp"])
    return TrafficWindow(window_id=f"cap01_w{index:04d}", capture_id="cap01",
                         window_start=start, window_end=start + 1.0, frames=frames)


def with_timing_attack(window: TrafficWindow) -> TrafficWindow:
    """MOCK: adds unknown-ID frames (fuzzing-style). Stage 1 catches this,
    Stage 2 does not, since no existing payload field is touched."""
    data = window.model_dump()
    start = data["window_start"]
    data["frames"] += [{"timestamp": start + 0.1 + k * 0.05, "can_id": "7FF",
                        "payload": "FFFFFFFFFFFFFFFF"} for k in range(10)]
    data["frames"].sort(key=lambda f: f["timestamp"])
    return TrafficWindow(**data)


def with_payload_attack(window: TrafficWindow) -> TrafficWindow:
    """MOCK: forces 0D0's byte0 out of its trained range. Stage 2 catches this,
    Stage 1 does not, since timing is untouched."""
    data = window.model_dump()
    for frame in data["frames"]:
        if frame["can_id"] == "0D0":
            frame["payload"] = "FF" + "00" * 7
    return TrafficWindow(**data)


@pytest.fixture(scope="module")
def stage1():
    return Stage1Model().fit([make_normal_window(i) for i in range(40)])


@pytest.fixture(scope="module")
def stage2():
    return Stage2Model().fit([make_normal_window(i) for i in range(40)])


@pytest.fixture(scope="module")
def fused_threshold(stage1, stage2):
    from defender.fusion import fuse
    scores = [fuse(stage1.score(make_normal_window(i)), stage2.score(make_normal_window(i))).score
              for i in range(100, 130)]
    return choose_threshold(scores, 0.0)


@pytest.fixture
def v1_defender(stage1):
    threshold = choose_threshold([stage1.score(make_normal_window(i)).score
                                  for i in range(100, 130)], 0.0)
    return Defender(stage1, threshold, "v1")


@pytest.fixture
def v2_defender(stage1, stage2, fused_threshold):
    return Defender(stage1, fused_threshold, "v2", stage2=stage2)


# ---------- v1 behaviour is unchanged by adding v2 support --------------
def test_v1_defender_has_no_stage2(v1_defender):
    assert v1_defender.stage2 is None


def test_v1_still_misses_payload_only_attacks(v1_defender):
    w = with_payload_attack(make_normal_window(200))
    assert v1_defender.score_window(w).decision == "ACCEPT"


def test_v1_evidence_is_labelled_stage1(v1_defender):
    out = v1_defender.score_window(with_timing_attack(make_normal_window(201)))
    assert out.evidence.startswith("[Stage 1]")


# ---------- v2 catches what v1 alone misses ------------------------------
def test_v2_catches_timing_attack(v2_defender):
    out = v2_defender.score_window(with_timing_attack(make_normal_window(202)))
    assert out.decision == "ATTACK" and out.evidence.startswith("[Stage 1]")


def test_v2_catches_payload_attack_v1_would_miss(v2_defender):
    out = v2_defender.score_window(with_payload_attack(make_normal_window(203)))
    assert out.decision == "ATTACK" and out.evidence.startswith("[Stage 2]")


def test_v2_still_accepts_normal_traffic(v2_defender):
    assert v2_defender.score_window(make_normal_window(204)).decision == "ACCEPT"


def test_v2_output_is_valid_defender_output(v2_defender):
    out = v2_defender.score_window(make_normal_window(205))
    assert out.model_version == "v2" and out.latency_ms is not None


# ---------- Construction checks ------------------------------------------
def test_rejects_untrained_stage2(stage1):
    with pytest.raises(ValueError, match="trained Stage2Model"):
        Defender(stage1, 0.5, "v2", stage2=Stage2Model())


def test_rejects_wrong_type_for_stage2(stage1):
    with pytest.raises(ValueError, match="trained Stage2Model"):
        Defender(stage1, 0.5, "v2", stage2="not a model")


# ---------- Loading: v1 files load as v1, v2 files load as v2 -----------
def test_load_v1_files_gives_no_stage2(stage1, tmp_path):
    threshold = choose_threshold([stage1.score(make_normal_window(i)).score
                                  for i in range(100, 130)], 0.0)
    stage1.save(stage1_path(tmp_path, "v1"), "v1")
    save_threshold(threshold, "v1", threshold_path(tmp_path, "v1"))
    loaded = Defender.load(tmp_path, "v1")
    assert loaded.stage2 is None


def test_load_v2_files_gives_fusion(stage1, stage2, fused_threshold, tmp_path):
    stage1.save(stage1_path(tmp_path, "v2"), "v2")
    stage2.save(stage2_path(tmp_path, "v2"), "v2")
    save_threshold(fused_threshold, "v2", threshold_path(tmp_path, "v2"))
    loaded = Defender.load(tmp_path, "v2")
    assert loaded.stage2 is not None
    out = loaded.score_window(with_payload_attack(make_normal_window(206)))
    assert out.decision == "ATTACK" and out.evidence.startswith("[Stage 2]")


def test_does_not_modify_window_in_v2(v2_defender):
    w = with_payload_attack(make_normal_window(207))
    before = w.model_dump()
    v2_defender.score_window(w)
    assert w.model_dump() == before