"""
Tests for defender/crossval_timing_cnn.py. Needs PyTorch -- run from the
machine that has it:
    .venv/bin/python3 -m pytest tests/test_crossval_timing_cnn.py -v

Uses the FAKE ROAD folder and manifest from conftest.py (the same fixture
tests/test_defender_crossval.py and tests/test_crossval_ml_baseline.py use).
Never touches the real dataset.
"""

import json

import pytest

torch = pytest.importorskip("torch")

from defender.crossval_timing_cnn import _examples, cross_validate, main
from part1.pipeline import RoadData

TRAIN = ["ambient_a", "ambient_b", "ambient_c"]
VALIDATION = ["ambient_d", "ambient_e"]
WATCH_IDS = ("0D0",)


def quiet(*_):
    pass


def road_for(fake):
    return RoadData(fake.data_dir, manifest=fake.manifest)


def test_examples_skips_masquerade_normal_windows(fake_road):
    road = road_for(fake_road)
    features, labels, meta = _examples(
        road, WATCH_IDS, ["speed_attack_1", "speed_attack_1_masquerade"])
    assert all(y for m, y in zip(meta, labels) if m["kind"] == "masquerade")
    assert len(features) == len(labels) == len(meta)
    assert all(len(f) == 3 and all(len(ch) == 10 for ch in f) for f in features)


def test_cross_validate_only_reads_train_validation_development(fake_road, monkeypatch):
    road = road_for(fake_road)
    read = []
    real_path = road.path
    monkeypatch.setattr(road, "path", lambda name: read.append(name) or real_path(name))

    report = cross_validate(road, WATCH_IDS, max_false_alarm_rate=0.2, epochs=10, log=quiet)

    off_limits = set(fake_road.manifest.names("final_test", "separate"))
    assert read and not (set(read) & off_limits)
    rows = {r["capture_id"]: r for r in report["per_capture"]}
    assert sorted(rows) == ["cap07", "cap08", "cap09", "cap10", "cap11"]   # development only
    assert "speed" in report["macro_recall_by_family"]
    assert 0.0 <= report["macro_recall_by_family"]["speed"] <= 1.0


def test_cross_validate_needs_at_least_two_folds(fake_road):
    road = road_for(fake_road)

    class OneFoldManifest:
        def __getattr__(self, item):
            return getattr(fake_road.manifest, item)

        def development_folds(self):
            return {1: fake_road.manifest.names("development")}

    road.manifest = OneFoldManifest()
    with pytest.raises(ValueError, match="at least 2"):
        cross_validate(road, WATCH_IDS, 0.1, epochs=5, log=quiet)


def test_main_writes_a_report(fake_road, tmp_path):
    out = tmp_path / "out" / "crossval_timing_cnn.json"
    rc = main(["--data-dir", str(fake_road.data_dir),
              "--manifest", str(fake_road.manifest_path),
              "--max-false-alarm-rate", "0.2",
              "--watch-ids", "0D0",
              "--epochs", "10",
              "--out", str(out)])
    assert rc == 0
    report = json.loads(out.read_text())
    assert report["max_false_alarm_rate"] == 0.2
    assert len(report["per_capture"]) == 5
