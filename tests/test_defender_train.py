"""
Tests for defender/train.py.

Run from the EdgeGuard folder with:
    python -m pytest tests/test_defender_train.py -v

ALL captures and windows here are MOCK. mock_make_windows() imitates the
agreed Part 1 signature:
    make_windows(capture_path, capture_id, window_s, stride_s) -> Iterator[TrafficWindow]
It does not read any file; it generates periodic traffic.
"""

import json

import pytest

from defender.defender import Defender
from defender.train import train_defender, train_from_captures, train_info_path
from shared.schemas import TrafficWindow

MOCK_PERIODS = {"0D0": 0.010, "0F4": 0.020, "1A0": 0.050}
MOCK_WINDOWS_PER_CAPTURE = 8


def mock_make_windows(capture_path, capture_id, window_s, stride_s):
    """MOCK stand-in for Part 1's make_windows(). Ignores capture_path."""
    seed = int(capture_id.replace("cap", ""))
    for w in range(MOCK_WINDOWS_PER_CAPTURE):
        start = 1080000000.0 + w * stride_s
        frames = []
        for can_id, period in MOCK_PERIODS.items():
            for k in range(int(round(window_s / period))):
                jitter = (((seed * 11 + w * 7 + k * 3) % 5) - 2) * 0.0002
                t = round(start + 0.001 + k * period + jitter, 6)
                if start <= t <= start + window_s:
                    frames.append({"timestamp": t, "can_id": can_id,
                                   "payload": "0011223344556677"})
        frames.sort(key=lambda f: f["timestamp"])
        yield TrafficWindow(window_id=f"{capture_id}_w{w:04d}", capture_id=capture_id,
                            window_start=start, window_end=start + window_s, frames=frames)


def windows_for(capture_ids, window_s=1.0, stride_s=1.0):
    out = []
    for cid in capture_ids:
        out.extend(mock_make_windows("MOCK", cid, window_s, stride_s))
    return out


# MOCK split: 12 ambient captures exist in ROAD; here 6 train, 2 validation.
TRAIN_IDS = [f"cap{i:02d}" for i in range(1, 7)]
VALIDATION_IDS = ["cap07", "cap08"]


def run(tmp_path, **overrides):
    args = dict(
        train_windows=windows_for(TRAIN_IDS),
        validation_windows=windows_for(VALIDATION_IDS),
        max_false_alarm_rate=0.0,
        model_version="v1",
        model_dir=tmp_path,
        window_s=1.0,
        stride_s=1.0,
    )
    args.update(overrides)
    return train_defender(**args)


# ---------- Normal training -------------------------------------------
def test_training_saves_loadable_defender(tmp_path):
    run(tmp_path)
    defender = Defender.load(tmp_path, "v1")
    out = defender.score_window(windows_for(["cap09"])[0])
    assert out.model_version == "v1"


def test_report_records_what_was_used(tmp_path):
    report = run(tmp_path)
    assert report.train_capture_ids == TRAIN_IDS
    assert report.validation_capture_ids == VALIDATION_IDS
    assert report.train_windows == 6 * MOCK_WINDOWS_PER_CAPTURE
    assert report.validation_windows == 2 * MOCK_WINDOWS_PER_CAPTURE
    assert report.window_s == 1.0 and report.stride_s == 1.0


def test_validation_false_alarm_rate_meets_target(tmp_path):
    report = run(tmp_path, max_false_alarm_rate=0.0)
    assert report.validation_false_alarm_rate == 0.0


def test_train_info_file_matches_report(tmp_path):
    report = run(tmp_path)
    saved = json.loads(train_info_path(tmp_path, "v1").read_text(encoding="utf-8"))
    assert saved == report.model_dump()


def test_threshold_in_report_is_the_saved_one(tmp_path):
    report = run(tmp_path)
    assert Defender.load(tmp_path, "v1").threshold == report.threshold


def test_train_from_captures_uses_make_windows(tmp_path):
    report = train_from_captures(
        train_captures=[("MOCK_PATH", cid) for cid in TRAIN_IDS],
        validation_captures=[("MOCK_PATH", cid) for cid in VALIDATION_IDS],
        make_windows=mock_make_windows,
        window_s=1.0, stride_s=1.0, max_false_alarm_rate=0.0,
        model_version="v1", model_dir=tmp_path,
    )
    assert report.train_windows == 6 * MOCK_WINDOWS_PER_CAPTURE


def test_accepts_iterators_like_make_windows(tmp_path):
    report = run(tmp_path, train_windows=mock_make_windows("MOCK", "cap01", 1.0, 1.0))
    assert report.train_windows == MOCK_WINDOWS_PER_CAPTURE


# ---------- Protections -----------------------------------------------
def test_rejects_capture_in_both_splits(tmp_path):
    with pytest.raises(ValueError, match="Data leak"):
        run(tmp_path, validation_windows=windows_for(["cap01", "cap07"]))


def test_refuses_to_overwrite_existing_model(tmp_path):
    run(tmp_path)
    with pytest.raises(FileExistsError, match="already exists"):
        run(tmp_path)


def test_overwrite_true_is_allowed(tmp_path):
    run(tmp_path)
    run(tmp_path, overwrite=True)


def test_new_version_does_not_touch_old_one(tmp_path):
    run(tmp_path, model_version="v1")
    before = (tmp_path / "threshold_v1.json").read_text(encoding="utf-8")
    run(tmp_path, model_version="v2")
    assert (tmp_path / "threshold_v1.json").read_text(encoding="utf-8") == before


def test_false_alarm_rate_has_no_default(tmp_path):
    with pytest.raises(TypeError):
        train_defender(train_windows=windows_for(TRAIN_IDS),
                       validation_windows=windows_for(VALIDATION_IDS),
                       model_version="v1", model_dir=tmp_path,
                       window_s=1.0, stride_s=1.0)


# ---------- Invalid inputs --------------------------------------------
def test_rejects_empty_validation(tmp_path):
    with pytest.raises(ValueError, match="validation_windows is empty"):
        run(tmp_path, validation_windows=[])


def test_rejects_too_few_train_windows(tmp_path):
    with pytest.raises(ValueError, match="at least 2 train windows"):
        run(tmp_path, train_windows=windows_for(["cap01"])[:1])


def test_rejects_non_window_items(tmp_path):
    with pytest.raises(TypeError, match="expected TrafficWindow"):
        run(tmp_path, train_windows=[{"frames": []}, {"frames": []}])


def test_rejects_empty_model_version(tmp_path):
    with pytest.raises(ValueError, match="model_version"):
        run(tmp_path, model_version="")


@pytest.mark.parametrize("window_s, stride_s", [(0.0, 1.0), (1.0, -0.5)])
def test_rejects_non_positive_window_settings(tmp_path, window_s, stride_s):
    with pytest.raises(ValueError, match="positive"):
        run(tmp_path, window_s=window_s, stride_s=stride_s)