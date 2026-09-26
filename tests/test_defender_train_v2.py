"""
Tests for defender/train.py train_v2 (Stage 1 reused, new Stage 2, fusion).

Run from the EdgeGuard folder with:
    python -m pytest tests/test_defender_train_v2.py -v

ALL captures and windows here are MOCK.
"""

import json

import pytest

from defender.defender import Defender
from defender.train import train_defender, train_info_path, train_v2
from shared.schemas import TrafficWindow

TRAIN_IDS = [f"cap{i:02d}" for i in range(1, 7)]
VALIDATION_IDS = ["cap07", "cap08"]
WINDOWS_PER_CAPTURE = 8


def make_windows(capture_ids, attack_byte0=None):
    """MOCK windows: normal timing + varying byte0 payload on 0D0.
    If attack_byte0 is given, every 0D0 frame in the LAST window of the
    LAST capture is forced to that value (a MOCK payload-only attack)."""
    out = []
    for cid in capture_ids:
        seed = int(cid.replace("cap", ""))
        for w in range(WINDOWS_PER_CAPTURE):
            start = 1080000000.0 + seed * 100 + w
            frames = []
            for k in range(50):
                jitter = (((seed * 11 + w * 7 + k * 3) % 5) - 2) * 0.0002
                frames.append({"timestamp": round(start + 0.001 + k * 0.02 + jitter, 6),
                               "can_id": "0D0", "payload": f"{k % 200:02X}" + "00" * 7})
            frames.sort(key=lambda f: f["timestamp"])
            window = TrafficWindow(window_id=f"{cid}_w{w:04d}", capture_id=cid,
                                   window_start=start, window_end=start + 1.0, frames=frames)
            out.append(window)
    if attack_byte0 is not None:
        data = out[-1].model_dump()
        for frame in data["frames"]:
            frame["payload"] = f"{attack_byte0:02X}" + "00" * 7
        out[-1] = TrafficWindow(**data)
    return out


def train_v1(tmp_path, version="v1"):
    return train_defender(
        train_windows=make_windows(TRAIN_IDS),
        validation_windows=make_windows(VALIDATION_IDS),
        max_false_alarm_rate=0.0,
        model_version=version,
        model_dir=tmp_path,
        window_s=1.0,
        stride_s=1.0,
    )


def run_v2(tmp_path, v1_version="v1", **overrides):
    args = dict(
        v1_model_dir=tmp_path,
        v1_version=v1_version,
        train_windows=make_windows(TRAIN_IDS),
        validation_windows=make_windows(VALIDATION_IDS),
        model_version="v2",
        model_dir=tmp_path,
        window_s=1.0,
        stride_s=1.0,
    )
    args.update(overrides)
    return train_v2(**args)


# ---------- Normal v2 training ------------------------------------------
def test_v2_reuses_v1_false_alarm_rate_by_default(tmp_path):
    train_v1(tmp_path)
    report = run_v2(tmp_path)
    assert report.max_false_alarm_rate == 0.0
    assert report.base_stage1_version == "v1"


def test_v2_can_override_false_alarm_rate(tmp_path):
    train_v1(tmp_path)
    report = run_v2(tmp_path, max_false_alarm_rate=0.05)
    assert report.max_false_alarm_rate == 0.05


def test_v2_defender_loads_with_stage2(tmp_path):
    train_v1(tmp_path)
    run_v2(tmp_path)
    defender = Defender.load(tmp_path, "v2")
    assert defender.stage2 is not None
    assert defender.model_version == "v2"


def test_v1_model_files_untouched_by_v2_training(tmp_path):
    train_v1(tmp_path)
    before = (tmp_path / "stage1_v1.json").read_text(encoding="utf-8")
    run_v2(tmp_path)
    assert (tmp_path / "stage1_v1.json").read_text(encoding="utf-8") == before


def test_v2_stage1_content_matches_v1_stage1(tmp_path):
    """Stage 1 is REUSED, not retrained: the saved files' Stage 1 content
    is identical (only the model_version label inside differs)."""
    train_v1(tmp_path)
    run_v2(tmp_path)
    v1_data = json.loads((tmp_path / "stage1_v1.json").read_text(encoding="utf-8"))
    v2_data = json.loads((tmp_path / "stage1_v2.json").read_text(encoding="utf-8"))
    v1_data.pop("model_version")
    v2_data.pop("model_version")
    assert v1_data == v2_data


def test_v2_catches_a_payload_attack_v1_misses(tmp_path):
    """A MOCK masquerade-style attack (0D0 held at an unseen value): v1
    (Stage 1 only) misses it, v2 (Stage 1 + Stage 2) catches it."""
    train_v1(tmp_path)
    run_v2(tmp_path)
    v1_defender = Defender.load(tmp_path, "v1")
    v2_defender = Defender.load(tmp_path, "v2")

    attack_window = make_windows(["cap09"], attack_byte0=250)[-1]
    assert v1_defender.score_window(attack_window).decision == "ACCEPT"
    assert v2_defender.score_window(attack_window).decision == "ATTACK"


# ---------- Enforces the "only Stage 2 differs" invariant ----------------
def test_rejects_different_train_captures_than_v1(tmp_path):
    train_v1(tmp_path)
    with pytest.raises(ValueError, match="SAME train captures"):
        run_v2(tmp_path, train_windows=make_windows(["cap01", "cap02"]))


def test_rejects_different_validation_captures_than_v1(tmp_path):
    train_v1(tmp_path)
    with pytest.raises(ValueError, match="SAME validation captures"):
        run_v2(tmp_path, validation_windows=make_windows(["cap07"]))


def test_rejects_capture_in_both_splits(tmp_path):
    train_v1(tmp_path)
    with pytest.raises(ValueError, match="Data leak"):
        run_v2(tmp_path, validation_windows=make_windows(TRAIN_IDS[:1] + VALIDATION_IDS))


# ---------- Missing v1 / bad inputs ---------------------------------------
def test_missing_v1_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_v2(tmp_path)


def test_refuses_to_overwrite_existing_v2(tmp_path):
    train_v1(tmp_path)
    run_v2(tmp_path)
    with pytest.raises(FileExistsError, match="already exists"):
        run_v2(tmp_path)


def test_overwrite_true_is_allowed(tmp_path):
    train_v1(tmp_path)
    run_v2(tmp_path)
    run_v2(tmp_path, overwrite=True)


def test_rejects_empty_model_version(tmp_path):
    train_v1(tmp_path)
    with pytest.raises(ValueError, match="model_version"):
        run_v2(tmp_path, model_version="")


def test_report_saved_to_disk_matches_returned_report(tmp_path):
    train_v1(tmp_path)
    report = run_v2(tmp_path)
    saved = json.loads(train_info_path(tmp_path, "v2").read_text(encoding="utf-8"))
    assert saved == report.model_dump()


def test_report_records_full_capture_ranges(tmp_path):
    train_v1(tmp_path)
    frames = [[(f.timestamp, f.can_id, f.payload) for w in make_windows([cid]) for f in w.frames]
              for cid in TRAIN_IDS]
    report = run_v2(tmp_path, stage2_range_captures=frames)
    assert report.stage2_ranges_from_full_captures is True


def test_report_default_is_windows_only(tmp_path):
    train_v1(tmp_path)
    assert run_v2(tmp_path).stage2_ranges_from_full_captures is False

def test_v2_bound_percentile_defaults_to_none_and_can_be_set(tmp_path):
    train_v1(tmp_path)
    report_default = run_v2(tmp_path)
    assert report_default.stage2_bound_percentile is None

    report_p = run_v2(tmp_path, model_version="v2p", stage2_bound_percentile=10.0)
    assert report_p.stage2_bound_percentile == 10.0
    defender = Defender.load(tmp_path, "v2p")
    assert defender.stage2.bound_percentile == 10.0
