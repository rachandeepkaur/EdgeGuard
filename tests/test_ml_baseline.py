"""
Tests for defender/ml_baseline.py (comparison-only ML baseline).

ALL windows here are MOCK traffic, same style as tests/test_defender_stage2.py:
these prove the wiring and the math are correct, not how well the baseline
does on real ROAD data (see results/crossval_ml_baseline.json and
results/final_evaluation.json for the real numbers).
"""

import math

import pytest

from defender.ml_baseline import (
    FEATURES,
    MLBaselineDefender,
    MLBaselineModel,
    extract_features,
)
from defender.stage1 import Stage1Model
from defender.stage2 import Stage2Model
from defender.threshold import ACCEPT, ATTACK
from shared.schemas import TrafficWindow

MOCK_PERIODS = {"0D0": 0.010, "0F4": 0.020}


def mock_payload(k: int) -> str:
    b0 = k % 200
    return f"{b0:02X}00000000000000"


def make_normal_window(index: int) -> TrafficWindow:
    start = 1000.0 + index
    frames = []
    for can_id, period in MOCK_PERIODS.items():
        count = int(round(1.0 / period))
        for k in range(count):
            t = round(start + 0.001 + k * period, 6)
            frames.append({"timestamp": t, "can_id": can_id,
                           "payload": mock_payload(index * 100 + k)})
    frames.sort(key=lambda f: f["timestamp"])
    return TrafficWindow(window_id=f"cap01_w{index:04d}", capture_id="cap01",
                         window_start=start, window_end=start + 1.0, frames=frames)


def make_frozen_attack_window(index: int) -> TrafficWindow:
    """A window where 0D0's byte0 never changes -- a frozen_break attack."""
    window = make_normal_window(index)
    data = window.model_dump()
    for frame in data["frames"]:
        if frame["can_id"] == "0D0":
            frame["payload"] = "FF00000000000000"
    return TrafficWindow(**data)


@pytest.fixture(scope="module")
def fitted_stages():
    normal = [make_normal_window(i) for i in range(60)]
    stage1 = Stage1Model().fit(normal)
    stage2 = Stage2Model().fit(normal)
    return stage1, stage2


def test_extract_features_returns_six_values_in_feature_order(fitted_stages):
    stage1, stage2 = fitted_stages
    features = extract_features(stage1, stage2, make_normal_window(999))
    assert len(features) == len(FEATURES) == 6
    assert all(isinstance(f, float) for f in features)


def test_frozen_attack_produces_a_higher_frozen_break_feature(fitted_stages):
    stage1, stage2 = fitted_stages
    normal_features = extract_features(stage1, stage2, make_normal_window(998))
    attack_features = extract_features(stage1, stage2, make_frozen_attack_window(998))
    frozen_idx = FEATURES.index("frozen_break")
    assert attack_features[frozen_idx] > normal_features[frozen_idx]


def test_fit_requires_both_classes():
    model = MLBaselineModel()
    with pytest.raises(ValueError, match="one normal and one attack"):
        model.fit([[0.0] * 6, [0.0] * 6], [False, False])


def test_fit_requires_matching_lengths():
    model = MLBaselineModel()
    with pytest.raises(ValueError, match="same length"):
        model.fit([[0.0] * 6], [False, True])


def test_predict_proba_is_between_0_and_1(fitted_stages):
    stage1, stage2 = fitted_stages
    normal = [extract_features(stage1, stage2, make_normal_window(i)) for i in range(20)]
    attacks = [extract_features(stage1, stage2, make_frozen_attack_window(i)) for i in range(20)]
    model = MLBaselineModel().fit(normal + attacks, [False] * 20 + [True] * 20)
    scores = model.predict_proba_attack(normal + attacks)
    assert all(0.0 <= s <= 1.0 for s in scores)
    # A model fit on a clearly-separable frozen-vs-normal split should score
    # attacks higher than normals on average -- otherwise the wiring is broken.
    assert sum(scores[20:]) / 20 > sum(scores[:20]) / 20


def test_calibrate_threshold_limits_false_alarms_on_the_calibration_set(fitted_stages):
    stage1, stage2 = fitted_stages
    normal = [extract_features(stage1, stage2, make_normal_window(i)) for i in range(50)]
    attacks = [extract_features(stage1, stage2, make_frozen_attack_window(i)) for i in range(50)]
    model = MLBaselineModel().fit(normal + attacks, [False] * 50 + [True] * 50)
    model.calibrate_threshold(normal, max_false_alarm_rate=0.1)
    scores = model.predict_proba_attack(normal)
    false_alarms = sum(1 for s in scores if s >= model.threshold)
    assert false_alarms / len(normal) <= 0.1 + 1e-9


def test_save_load_round_trip_preserves_predictions(fitted_stages, tmp_path):
    stage1, stage2 = fitted_stages
    normal = [extract_features(stage1, stage2, make_normal_window(i)) for i in range(30)]
    attacks = [extract_features(stage1, stage2, make_frozen_attack_window(i)) for i in range(30)]
    model = MLBaselineModel().fit(normal + attacks, [False] * 30 + [True] * 30)
    model.calibrate_threshold(normal, max_false_alarm_rate=0.05)

    path = tmp_path / "ml_baseline_v1.json"
    model.save(path)
    reloaded = MLBaselineModel.load(path)

    before = model.predict_proba_attack(normal[:5] + attacks[:5])
    after = reloaded.predict_proba_attack(normal[:5] + attacks[:5])
    assert before == pytest.approx(after)
    assert reloaded.threshold == model.threshold


def test_load_rejects_a_file_with_a_different_feature_order(tmp_path):
    import json
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({
        "features": ["only_one_feature"], "coef": [0.0], "intercept": 0.0,
        "threshold": 0.5, "training_windows": 1, "training_attack_windows": 0,
    }))
    with pytest.raises(ValueError, match="does not match"):
        MLBaselineModel.load(path)


def test_ml_baseline_defender_score_window_matches_schema_and_decision(fitted_stages):
    stage1, stage2 = fitted_stages
    normal = [extract_features(stage1, stage2, make_normal_window(i)) for i in range(40)]
    attacks = [extract_features(stage1, stage2, make_frozen_attack_window(i)) for i in range(40)]
    model = MLBaselineModel().fit(normal + attacks, [False] * 40 + [True] * 40)
    model.calibrate_threshold(normal, max_false_alarm_rate=0.05)

    defender = MLBaselineDefender(stage1, stage2, model)
    normal_output = defender.score_window(make_normal_window(500))
    attack_output = defender.score_window(make_frozen_attack_window(500))

    for output in (normal_output, attack_output):
        assert output.model_version == "ml_baseline_v1"
        assert 0.0 <= output.attack_score <= 1.0
        assert output.threshold == model.threshold
        assert output.latency_ms is not None and output.latency_ms >= 0.0
        expected = ATTACK if output.attack_score >= output.threshold else ACCEPT
        assert output.decision == expected


def test_ml_baseline_defender_rejects_an_unfitted_model(fitted_stages):
    stage1, stage2 = fitted_stages
    with pytest.raises(ValueError, match="fitted MLBaselineModel"):
        MLBaselineDefender(stage1, stage2, MLBaselineModel())


def test_ml_baseline_defender_rejects_an_unfitted_stage1():
    unfitted = Stage1Model()
    fitted_stage2 = Stage2Model().fit([make_normal_window(i) for i in range(10)])
    model = MLBaselineModel().fit([[0.0] * 6, [1.0] * 6], [False, True])
    with pytest.raises(ValueError, match="fitted Stage1Model"):
        MLBaselineDefender(unfitted, fitted_stage2, model)
