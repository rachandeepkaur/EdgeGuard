"""Tests for shared/schemas.py using mock data.

Run from the EdgeGuard folder:
    python -m pytest tests/test_shared_schemas.py -v
"""

import pytest
from pydantic import ValidationError

from shared.schemas import DefenderOutput, Frame, GroundTruthLabel, TrafficWindow


def mock_window():
    """Matches the team's agreed window format."""
    return {
        "window_id": "cap07_w0153",
        "capture_id": "cap07",
        "window_start": 1532.0,
        "window_end": 1533.0,
        "frames": [
            {"timestamp": 1532.001, "can_id": "0F4", "payload": "960C010204B10240"},
            {"timestamp": 1532.050, "can_id": "1A0", "payload": "00FF"},
        ],
        "features": {},
        "decoded_signals": {},
    }


def mock_defender_output():
    return {
        "window_id": "cap07_w0153",
        "attack_score": 0.83,
        "threshold": 0.70,
        "decision": "ATTACK",
        "evidence": "Cross-signal mismatch",
        "model_version": "v1",
        "latency_ms": 4.2,
    }


def test_team_example_window_is_valid():
    window = TrafficWindow(**mock_window())
    assert window.frames[0].payload == "960C010204B10240"


def test_window_after_preprocessing_is_valid():
    data = mock_window()
    data["features"] = {"frame_count": 2.0}
    data["decoded_signals"] = {"MOCK_speed": [50.0, 50.2]}
    assert TrafficWindow(**data).features["frame_count"] == 2.0


def test_features_and_signals_are_optional():
    data = mock_window()
    del data["features"]
    del data["decoded_signals"]
    assert TrafficWindow(**data).decoded_signals == {}


def test_empty_payload_is_allowed():
    """A CAN frame may carry zero data bytes."""
    assert Frame(timestamp=1.0, can_id="0F4", payload="").payload == ""


def test_lowercase_hex_is_allowed():
    assert Frame(timestamp=1.0, can_id="0f4", payload="960c").can_id == "0f4"


def test_valid_defender_attack():
    assert DefenderOutput(**mock_defender_output()).decision == "ATTACK"


def test_valid_defender_accept():
    data = mock_defender_output()
    data["attack_score"] = 0.20
    data["decision"] = "ACCEPT"
    assert DefenderOutput(**data).decision == "ACCEPT"


def test_score_equal_to_threshold_is_attack():
    data = mock_defender_output()
    data["attack_score"] = 0.70
    assert DefenderOutput(**data).decision == "ATTACK"


def test_latency_is_optional():
    data = mock_defender_output()
    del data["latency_ms"]
    assert DefenderOutput(**data).latency_ms is None


def test_valid_attack_label():
    label = GroundTruthLabel(
        window_id="cap07_w0153",
        is_attack=True,
        family="offset",
        target="MOCK_speed",
        injection_interval=[1532.4, 1537.4],
    )
    assert label.injection_interval == [1532.4, 1537.4]


def test_attack_label_without_interval_is_allowed():
    """Some ROAD attacks began before recording, so no interval is given."""
    label = GroundTruthLabel(
        window_id="cap09_w0001", is_attack=True, family="accelerator"
    )
    assert label.injection_interval is None


def test_valid_normal_label():
    assert not GroundTruthLabel(window_id="cap07_w0001", is_attack=False).is_attack


def test_rejects_old_parallel_array_format():
    data = mock_window()
    del data["frames"]
    data["timestamps"] = [1532.001]
    data["can_ids"] = ["0F4"]
    data["payloads"] = ["960C010204B10240"]
    with pytest.raises(ValidationError):
        TrafficWindow(**data)


def test_rejects_non_hex_payload():
    with pytest.raises(ValidationError, match="non-hex"):
        Frame(timestamp=1.0, can_id="0F4", payload="96ZZ")


def test_rejects_odd_length_payload():
    with pytest.raises(ValidationError, match="odd number"):
        Frame(timestamp=1.0, can_id="0F4", payload="960")


def test_rejects_payload_longer_than_8_bytes():
    with pytest.raises(ValidationError, match="at most 8 bytes"):
        Frame(timestamp=1.0, can_id="0F4", payload="00" * 9)


def test_rejects_non_hex_can_id():
    with pytest.raises(ValidationError, match="not hexadecimal"):
        Frame(timestamp=1.0, can_id="speed", payload="00")


def test_rejects_frame_outside_window():
    data = mock_window()
    data["frames"][0]["timestamp"] = 1600.0
    with pytest.raises(ValidationError, match="outside the window"):
        TrafficWindow(**data)


def test_rejects_window_end_before_start():
    data = mock_window()
    data["window_end"] = 1000.0
    with pytest.raises(ValidationError, match="before window_start"):
        TrafficWindow(**data)


def test_defender_input_rejects_is_attack():
    data = mock_window()
    data["is_attack"] = True
    with pytest.raises(ValidationError):
        TrafficWindow(**data)


def test_defender_input_rejects_family():
    data = mock_window()
    data["family"] = "offset"
    with pytest.raises(ValidationError):
        TrafficWindow(**data)


def test_defender_input_rejects_injection_interval():
    data = mock_window()
    data["injection_interval"] = [1532.4, 1532.9]
    with pytest.raises(ValidationError):
        TrafficWindow(**data)


def test_frame_rejects_hidden_label():
    data = mock_window()
    data["frames"][0]["is_attack"] = True
    with pytest.raises(ValidationError):
        TrafficWindow(**data)


def test_rejects_capture_id_revealing_attack():
    data = mock_window()
    data["capture_id"] = "fuzzing_attack_1"
    with pytest.raises(ValidationError, match="neutral"):
        TrafficWindow(**data)


def test_rejects_window_id_revealing_ambient():
    data = mock_window()
    data["window_id"] = "Ambient_dyno_w0001"
    with pytest.raises(ValidationError, match="neutral"):
        TrafficWindow(**data)


def test_rejects_decision_that_contradicts_score():
    data = mock_defender_output()
    data["decision"] = "ACCEPT"
    with pytest.raises(ValidationError, match="should be ATTACK"):
        DefenderOutput(**data)


def test_rejects_unknown_decision():
    data = mock_defender_output()
    data["decision"] = "MAYBE"
    with pytest.raises(ValidationError):
        DefenderOutput(**data)


def test_rejects_score_below_zero():
    data = mock_defender_output()
    data["attack_score"] = -0.1
    data["decision"] = "ACCEPT"
    with pytest.raises(ValidationError):
        DefenderOutput(**data)


@pytest.mark.parametrize("bad_threshold", [-0.1, 1.1])
def test_rejects_threshold_outside_zero_to_one(bad_threshold):
    data = mock_defender_output()
    data["threshold"] = bad_threshold
    with pytest.raises(ValidationError):
        DefenderOutput(**data)


def test_rejects_score_above_one():
    data = mock_defender_output()
    data["attack_score"] = 1.5
    with pytest.raises(ValidationError):
        DefenderOutput(**data)


def test_rejects_negative_latency():
    data = mock_defender_output()
    data["latency_ms"] = -1.0
    with pytest.raises(ValidationError):
        DefenderOutput(**data)


def test_defender_output_rejects_metrics():
    """The Defender does not know labels, so it cannot report recall."""
    data = mock_defender_output()
    data["recall"] = 0.9
    with pytest.raises(ValidationError):
        DefenderOutput(**data)


def test_rejects_attack_label_without_family():
    with pytest.raises(ValidationError, match="family is required"):
        GroundTruthLabel(window_id="cap07_w0153", is_attack=True)


def test_rejects_normal_label_with_family():
    with pytest.raises(ValidationError, match="must not have"):
        GroundTruthLabel(window_id="cap07_w0001", is_attack=False, family="offset")


def test_rejects_normal_label_with_interval():
    with pytest.raises(ValidationError, match="must not have"):
        GroundTruthLabel(
            window_id="cap07_w0001", is_attack=False, injection_interval=[1.0, 2.0]
        )


def test_old_onset_field_is_rejected():
    with pytest.raises(ValidationError):
        GroundTruthLabel(
            window_id="cap07_w0153", is_attack=True, family="offset", onset=1532.4
        )


@pytest.mark.parametrize(
    "bad_interval, message",
    [
        ([1532.4], "exactly 2"),
        ([1.0, 2.0, 3.0], "exactly 2"),
        ([1537.4, 1532.4], "end must be > start"),
        ([10.0, 10.0], "end must be > start"),
        ([-5.0, 2.0], "negative"),
        ([float("nan"), 2.0], "finite numbers"),
        ([1.0, float("inf")], "finite numbers"),
    ],
)
def test_rejects_invalid_interval(bad_interval, message):
    with pytest.raises(ValidationError, match=message):
        GroundTruthLabel(
            window_id="cap07_w0153",
            is_attack=True,
            family="offset",
            injection_interval=bad_interval,
        )

# ---------- vehicle_id and label details (Part 1) ---------------------
def test_window_vehicle_id_is_optional_and_must_be_neutral():
    assert TrafficWindow(**mock_window()).vehicle_id is None
    assert TrafficWindow(**{**mock_window(), "vehicle_id": "veh01"}).vehicle_id == "veh01"
    with pytest.raises(ValidationError, match="neutral"):
        TrafficWindow(**{**mock_window(), "vehicle_id": "ambient_car"})


def test_attack_label_carries_injected_frames_and_overlap():
    label = GroundTruthLabel(window_id="cap07_w0012", is_attack=True, family="speed",
                             injection_interval=[8.0, 12.0], injected_frames=3,
                             interval_overlap_s=0.4)
    assert label.injected_frames == 3 and label.interval_overlap_s == 0.4


def test_normal_label_cannot_have_injected_frames():
    with pytest.raises(ValidationError, match="must not have"):
        GroundTruthLabel(window_id="cap07_w0001", is_attack=False, injected_frames=1)
    with pytest.raises(ValidationError, match="must not have"):
        GroundTruthLabel(window_id="cap07_w0001", is_attack=False, interval_overlap_s=0.2)
