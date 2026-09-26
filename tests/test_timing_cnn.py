"""
Tests for defender/timing_cnn.py (comparison-only TimingCNN).

Needs PyTorch. This session's sandboxed bridge and the real laptop terminal
resolve `.venv/bin/python3` to two DIFFERENT system interpreters (a real
venv was never created here -- `.venv/bin/python3` is a bare symlink to
`/usr/bin/python3`, so it silently points at whichever machine runs it).
Torch was installed on the real laptop terminal, so run this file FROM
THERE:
    .venv/bin/python3 -m pytest tests/test_timing_cnn.py -v

ALL windows here are MOCK traffic, same style as tests/test_ml_baseline.py:
these prove the wiring and the math are correct, not how well the model
does on real ROAD data.
"""

import pytest

torch = pytest.importorskip("torch")

from defender.threshold import ACCEPT, ATTACK
from defender.timing_cnn import (
    N_BINS,
    N_CHANNELS,
    TimingCNNDefender,
    TimingCNNModel,
    bin_timing_features,
)
from shared.schemas import TrafficWindow

MOCK_PERIODS = {"0D0": 0.010, "0F4": 0.020}


def make_normal_window(index: int) -> TrafficWindow:
    """MOCK 1-second window: steady periodic traffic, spread evenly across
    all 10 bins -- normal timing shape."""
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


def make_burst_attack_window(index: int) -> TrafficWindow:
    """MOCK attack: all of 0D0's traffic crammed into the first 100ms bin
    instead of spread across the window -- a timing-SHAPE anomaly a
    single-window aggregate count would miss but per-bin features catch."""
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


def test_bin_timing_features_shape():
    features = bin_timing_features(make_normal_window(0))
    assert len(features) == N_CHANNELS
    assert all(len(channel) == N_BINS for channel in features)


def test_burst_window_concentrates_counts_in_early_bins():
    normal = bin_timing_features(make_normal_window(1))
    burst = bin_timing_features(make_burst_attack_window(1))
    total_channel = 0
    # Normal traffic's total-count channel should be flatter (lower max bin)
    # relative to its own average than the burst's, which crams frames early.
    def peak_to_mean(channel):
        return max(channel) / (sum(channel) / len(channel) + 1e-9)
    assert peak_to_mean(burst[total_channel]) > peak_to_mean(normal[total_channel])


def test_fit_requires_both_classes():
    model = TimingCNNModel()
    zeros = [[0.0] * N_BINS] * N_CHANNELS
    with pytest.raises(ValueError, match="one normal and one attack"):
        model.fit([zeros, zeros], [False, False])


def test_fit_requires_matching_lengths():
    model = TimingCNNModel()
    zeros = [[0.0] * N_BINS] * N_CHANNELS
    with pytest.raises(ValueError, match="same length"):
        model.fit([zeros], [False, True])


def test_training_loss_decreases_and_scores_are_valid_probabilities():
    normal = [bin_timing_features(make_normal_window(i)) for i in range(15)]
    attacks = [bin_timing_features(make_burst_attack_window(i)) for i in range(15)]
    model = TimingCNNModel()
    model.fit(normal + attacks, [False] * 15 + [True] * 15, epochs=40, seed=0)
    assert model.fitted
    assert model.final_train_loss is not None and model.final_train_loss < 1.0

    scores = model.predict_proba_attack(normal + attacks)
    assert all(0.0 <= s <= 1.0 for s in scores)
    # A model trained on a clearly-separable burst-vs-steady split should
    # score the bursts higher on average -- otherwise the wiring is broken.
    assert sum(scores[15:]) / 15 > sum(scores[:15]) / 15


def test_calibrate_threshold_limits_false_alarms_on_the_calibration_set():
    normal = [bin_timing_features(make_normal_window(i)) for i in range(20)]
    attacks = [bin_timing_features(make_burst_attack_window(i)) for i in range(20)]
    model = TimingCNNModel().fit(normal + attacks, [False] * 20 + [True] * 20,
                                 epochs=40, seed=1)
    model.calibrate_threshold(normal, max_false_alarm_rate=0.1)
    scores = model.predict_proba_attack(normal)
    false_alarms = sum(1 for s in scores if s >= model.threshold)
    assert false_alarms / len(normal) <= 0.1 + 1e-9


def test_save_load_round_trip_preserves_predictions(tmp_path):
    normal = [bin_timing_features(make_normal_window(i)) for i in range(12)]
    attacks = [bin_timing_features(make_burst_attack_window(i)) for i in range(12)]
    model = TimingCNNModel().fit(normal + attacks, [False] * 12 + [True] * 12,
                                 epochs=30, seed=2)
    model.calibrate_threshold(normal, max_false_alarm_rate=0.1)

    directory = tmp_path / "timing_cnn_v1"
    model.save(directory)
    reloaded = TimingCNNModel.load(directory)

    before = model.predict_proba_attack(normal[:3] + attacks[:3])
    after = reloaded.predict_proba_attack(normal[:3] + attacks[:3])
    assert before == pytest.approx(after, abs=1e-6)
    assert reloaded.threshold == model.threshold


def test_load_missing_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        TimingCNNModel.load(tmp_path / "does_not_exist")


def test_timing_cnn_defender_score_window_matches_schema_and_decision():
    normal = [bin_timing_features(make_normal_window(i)) for i in range(15)]
    attacks = [bin_timing_features(make_burst_attack_window(i)) for i in range(15)]
    model = TimingCNNModel().fit(normal + attacks, [False] * 15 + [True] * 15,
                                 epochs=40, seed=3)
    model.calibrate_threshold(normal, max_false_alarm_rate=0.1)

    defender = TimingCNNDefender(model)
    normal_output = defender.score_window(make_normal_window(500))
    attack_output = defender.score_window(make_burst_attack_window(500))

    for output in (normal_output, attack_output):
        assert output.model_version == "timing_cnn_v1"
        assert 0.0 <= output.attack_score <= 1.0
        assert output.threshold == model.threshold
        assert output.latency_ms is not None and output.latency_ms >= 0.0
        expected = ATTACK if output.attack_score >= output.threshold else ACCEPT
        assert output.decision == expected


def test_timing_cnn_defender_rejects_an_unfitted_model():
    with pytest.raises(ValueError, match="fitted TimingCNNModel"):
        TimingCNNDefender(TimingCNNModel())
