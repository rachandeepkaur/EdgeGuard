"""
Tests for defender/fused_defender.py. Needs PyTorch -- run from the machine
that has it:
    .venv/bin/python3 -m pytest tests/test_fused_defender.py -v

ALL windows here are MOCK traffic, same style as tests/test_ml_baseline.py
and tests/test_timing_cnn.py.
"""

import pytest

torch = pytest.importorskip("torch")

from defender.defender import Defender
from defender.fused_defender import FusedV2TimingCNNDefender
from defender.stage1 import Stage1Model
from defender.stage2 import Stage2Model
from defender.threshold import ACCEPT, ATTACK, choose_threshold
from defender.fusion import fuse
from defender.timing_cnn import TimingCNNModel, bin_timing_features
from shared.schemas import TrafficWindow

MOCK_PERIODS = {"0D0": 0.010, "0F4": 0.020}


def make_normal_window(index: int) -> TrafficWindow:
    start = 1000.0 + index
    frames = []
    for can_id, period in MOCK_PERIODS.items():
        count = int(round(1.0 / period))
        for k in range(count):
            t = round(start + 0.001 + k * period, 6)
            frames.append({"timestamp": t, "can_id": can_id,
                           "payload": "0011223344556677"})
    frames.sort(key=lambda f: f["timestamp"])
    return TrafficWindow(window_id=f"cap01_w{index:04d}", capture_id="cap01",
                         window_start=start, window_end=start + 1.0, frames=frames)


def make_payload_attack_window(index: int) -> TrafficWindow:
    """MOCK masquerade: 0D0 frozen at an out-of-range value -- v2 (Stage 2)
    should catch this; TimingCNN, timing-only, should NOT (same frame count
    and timing as normal)."""
    window = make_normal_window(index)
    data = window.model_dump()
    for frame in data["frames"]:
        if frame["can_id"] == "0D0":
            frame["payload"] = "FF11223344556677"
    return TrafficWindow(**data)


def make_burst_window(index: int) -> TrafficWindow:
    """MOCK fabrication-style timing attack: 0D0 traffic crammed into the
    window's first 100ms -- TimingCNN should catch this; v2 (Stage 1/2 on
    whole-window aggregates with unchanged, in-range payloads) should not."""
    window = make_normal_window(index)
    data = window.model_dump()
    start = data["window_start"]
    frames = [f for f in data["frames"] if f["can_id"] != "0D0"]
    for k in range(100):
        frames.append({"timestamp": round(start + 0.001 + k * 0.0009, 6),
                       "can_id": "0D0", "payload": "0011223344556677"})
    frames.sort(key=lambda f: f["timestamp"])
    data["frames"] = frames
    return TrafficWindow(**data)


@pytest.fixture(scope="module")
def fitted_v2():
    normal = [make_normal_window(i) for i in range(60)]
    stage1 = Stage1Model().fit(normal)
    stage2 = Stage2Model().fit(normal)
    scores = [fuse(stage1.score(w), stage2.score(w)).score for w in normal]
    threshold = choose_threshold(scores, 0.05)
    return Defender(stage1=stage1, threshold=threshold, model_version="v2", stage2=stage2)


@pytest.fixture(scope="module")
def fitted_cnn():
    normal = [bin_timing_features(make_normal_window(i)) for i in range(30)]
    bursts = [bin_timing_features(make_burst_window(i)) for i in range(30)]
    model = TimingCNNModel().fit(normal + bursts, [False] * 30 + [True] * 30,
                                 epochs=40, seed=0)
    model.calibrate_threshold(normal, max_false_alarm_rate=0.05)
    return model


def test_rejects_a_non_defender_v2(fitted_cnn):
    with pytest.raises(ValueError, match="Defender for v2"):
        FusedV2TimingCNNDefender(object(), fitted_cnn)


def test_rejects_an_unfitted_cnn(fitted_v2):
    with pytest.raises(ValueError, match="fitted TimingCNNModel"):
        FusedV2TimingCNNDefender(fitted_v2, TimingCNNModel())


def test_normal_window_is_accepted(fitted_v2, fitted_cnn):
    fused = FusedV2TimingCNNDefender(fitted_v2, fitted_cnn)
    output = fused.score_window(make_normal_window(900))
    assert output.decision == ACCEPT
    assert output.model_version == "fused_v2_timing_cnn_v1"
    assert 0.0 <= output.attack_score < output.threshold == 1.0


def test_payload_attack_is_caught_by_v2_component(fitted_v2, fitted_cnn):
    fused = FusedV2TimingCNNDefender(fitted_v2, fitted_cnn)
    output = fused.score_window(make_payload_attack_window(901))
    assert output.decision == ATTACK
    assert "v2 (" in output.evidence


def test_timing_burst_is_caught_by_cnn_component(fitted_v2, fitted_cnn):
    fused = FusedV2TimingCNNDefender(fitted_v2, fitted_cnn)
    output = fused.score_window(make_burst_window(902))
    assert output.decision == ATTACK
    assert "TimingCNN" in output.evidence


def test_output_schema_is_self_consistent_either_way(fitted_v2, fitted_cnn):
    fused = FusedV2TimingCNNDefender(fitted_v2, fitted_cnn)
    for window in (make_normal_window(903), make_payload_attack_window(903),
                  make_burst_window(903)):
        output = fused.score_window(window)
        expected = ATTACK if output.attack_score >= output.threshold else ACCEPT
        assert output.decision == expected
        assert output.latency_ms is not None and output.latency_ms >= 0.0
