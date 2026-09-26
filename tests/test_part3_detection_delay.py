"""Tests for part3/detection_delay.py: time to first alert after attack onset.

Hand-built windows/outputs/labels for the logic tests (full control over edge
cases); one integration test against fake_road's real interval attack,
trained with real models, to prove the wiring works end to end.

Run from EdgeGuard: python -m pytest tests/test_part3_detection_delay.py -v
"""

import pytest

from defender.defender import Defender
from defender.run_training import main as train_main
from part1.pipeline import RoadData, preprocess
from part1.split_manifest import load_manifest
from part3.detection_delay import Scored, compute_detection_delay
from shared.schemas import DefenderOutput, GroundTruthLabel, TrafficWindow

THRESHOLD = 0.5
WINDOW_S = 1.0


def window(capture_id, index, start):
    return TrafficWindow(
        window_id=f"{capture_id}_w{index:04d}", capture_id=capture_id,
        window_start=start, window_end=start + WINDOW_S, frames=[],
    )


def output(capture_id, index, decision):
    score = 0.9 if decision == "ATTACK" else 0.1
    return DefenderOutput(
        window_id=f"{capture_id}_w{index:04d}", attack_score=score, threshold=THRESHOLD,
        decision=decision, evidence="test", model_version="vtest", latency_ms=1.0,
    )


def attack_label(capture_id, index, interval, family="speed", target="0D0"):
    return GroundTruthLabel(window_id=f"{capture_id}_w{index:04d}", is_attack=True,
                            family=family, target=target, injection_interval=list(interval))


def normal_label(capture_id, index):
    return GroundTruthLabel(window_id=f"{capture_id}_w{index:04d}", is_attack=False)


def scored_row(capture_id, index, start, decision, label):
    return Scored(window(capture_id, index, start), output(capture_id, index, decision), label)


# ---------------------------------------------------------------------
# Core logic, hand-built
# ---------------------------------------------------------------------

def test_delay_is_measured_from_interval_start_to_the_first_attack_window_end():
    """Interval starts at 8.0s. Windows 0..7 are pre-attack normal (ACCEPT,
    no label), window 8 covers [8, 9) and IS flagged. Delay = window_end (9.0)
    - interval_start (8.0) = 1.0s -- one window's worth of collection time,
    the minimum possible delay, not 0."""
    interval = (8.0, 12.0)
    rows = [scored_row("cap01", i, float(i), "ACCEPT", normal_label("cap01", i))
           for i in range(8)]
    rows.append(scored_row("cap01", 8, 8.0, "ATTACK", attack_label("cap01", 8, interval)))
    rows.append(scored_row("cap01", 9, 9.0, "ATTACK", attack_label("cap01", 9, interval)))

    report = compute_detection_delay(rows)
    assert report.captures_with_an_interval == 1
    assert report.captures_detected == 1
    [capture] = report.per_capture
    assert capture.detected is True
    assert capture.delay_s == pytest.approx(1.0)
    assert report.mean_delay_s == pytest.approx(1.0)


def test_a_late_alert_inside_the_interval_gives_a_larger_delay():
    """The Defender misses windows 8-9 and only catches window 10: delay =
    11.0 - 8.0 = 3.0s, a real (if unwelcome) number, not clamped to the
    first window."""
    interval = (8.0, 12.0)
    rows = [
        scored_row("cap01", 8, 8.0, "ACCEPT", attack_label("cap01", 8, interval)),
        scored_row("cap01", 9, 9.0, "ACCEPT", attack_label("cap01", 9, interval)),
        scored_row("cap01", 10, 10.0, "ATTACK", attack_label("cap01", 10, interval)),
    ]
    [capture] = compute_detection_delay(rows).per_capture
    assert capture.delay_s == pytest.approx(3.0)


def test_a_capture_never_flagged_is_a_miss_not_a_zero():
    interval = (8.0, 12.0)
    rows = [scored_row("cap01", i, float(i), "ACCEPT", attack_label("cap01", i, interval))
           for i in range(8, 12)]
    report = compute_detection_delay(rows)
    [capture] = report.per_capture
    assert capture.detected is False
    assert capture.delay_s is None
    # A miss must not silently become a 0 or be dropped from the denominator.
    assert report.captures_with_an_interval == 1
    assert report.captures_detected == 0
    assert report.mean_delay_s is None
    assert report.median_delay_s is None


def test_captures_with_no_interval_are_skipped_not_scored_as_zero():
    """Ambient/fuzzing windows carry no injection_interval -- they are not
    measurable and must not appear in per_capture at all."""
    rows = [scored_row("cap02", i, float(i), "ACCEPT", normal_label("cap02", i))
           for i in range(5)]
    interval = (8.0, 12.0)
    rows.append(scored_row("cap01", 8, 8.0, "ATTACK", attack_label("cap01", 8, interval)))
    report = compute_detection_delay(rows)
    assert report.captures_with_an_interval == 1
    assert {c.capture_id for c in report.per_capture} == {"cap01"}


def test_all_captures_with_no_interval_raises():
    rows = [scored_row("cap02", i, float(i), "ACCEPT", normal_label("cap02", i))
           for i in range(3)]
    with pytest.raises(ValueError, match="injection_interval"):
        compute_detection_delay(rows)


def test_empty_input_raises():
    with pytest.raises(ValueError, match="empty"):
        compute_detection_delay([])


def test_disagreeing_interval_starts_within_one_capture_raises():
    rows = [
        scored_row("cap01", 8, 8.0, "ATTACK", attack_label("cap01", 8, (8.0, 12.0))),
        scored_row("cap01", 9, 9.0, "ATTACK", attack_label("cap01", 9, (8.5, 12.0))),
    ]
    with pytest.raises(ValueError, match="disagree"):
        compute_detection_delay(rows)


def test_window_order_in_the_input_does_not_matter():
    """Windows are sorted internally by window_start; passing them in
    scrambled order must give the same answer as passing them in order."""
    interval = (8.0, 12.0)
    in_order = [
        scored_row("cap01", 8, 8.0, "ACCEPT", attack_label("cap01", 8, interval)),
        scored_row("cap01", 9, 9.0, "ATTACK", attack_label("cap01", 9, interval)),
    ]
    scrambled = [in_order[1], in_order[0]]
    assert (compute_detection_delay(in_order).per_capture[0].delay_s
            == compute_detection_delay(scrambled).per_capture[0].delay_s
            == pytest.approx(2.0))


def test_aggregates_across_multiple_captures_and_excludes_misses_from_percentiles():
    interval = (8.0, 12.0)
    rows = [
        scored_row("cap01", 8, 8.0, "ATTACK", attack_label("cap01", 8, interval)),   # delay 1.0
        scored_row("cap02", 8, 8.0, "ACCEPT", attack_label("cap02", 8, interval)),
        scored_row("cap02", 9, 9.0, "ATTACK", attack_label("cap02", 9, interval)),   # delay 2.0
        scored_row("cap03", 8, 8.0, "ACCEPT", attack_label("cap03", 8, interval)),   # miss
    ]
    report = compute_detection_delay(rows)
    assert report.captures_with_an_interval == 3
    assert report.captures_detected == 2
    assert report.mean_delay_s == pytest.approx(1.5)
    assert report.max_delay_s == pytest.approx(2.0)


# ---------------------------------------------------------------------
# Integration: real fake_road capture, real trained models
# ---------------------------------------------------------------------

def train(fake):
    model_dir = fake.data_dir.parent / "models"
    assert train_main(["--max-false-alarm-rate", "0.0",
                       "--data-dir", str(fake.data_dir),
                       "--manifest", str(fake.manifest_path),
                       "--model-dir", str(model_dir),
                       "--max-windows-per-capture", "10"]) == 0
    return model_dir


def test_wires_into_a_real_final_test_capture(fake_road):
    """speed_attack_3 (fake_road's real final_test attack, interval [8.0,
    11.99]) scored with the real v2 model: v2 catches the speed family, so
    this must report a real, positive, finite delay -- not raise, not None."""
    model_dir = train(fake_road)
    v2 = Defender.load(model_dir, "v2")
    manifest = load_manifest(fake_road.manifest_path)
    road = RoadData(fake_road.data_dir, manifest=manifest, final_evaluation=True)

    rows = []
    for window, label in road.labelled_windows("speed_attack_3"):
        rows.append(Scored(window, v2.score_window(preprocess(window)), label))

    report = compute_detection_delay(rows)
    assert report.captures_with_an_interval == 1
    [capture] = report.per_capture
    assert capture.capture_id == "cap12"
    assert capture.family == "speed"
    assert capture.interval_start_s == pytest.approx(8.0)
    if capture.detected:
        assert capture.delay_s > 0
    else:
        # Honest either way: v2 is expected to catch this on real fake_road
        # data (test_run_final_evaluation.py's own headline finding), but if
        # something upstream ever changes this must fail loudly, not silently.
        pytest.fail("v2 was expected to detect speed_attack_3; check the fixture/model")
