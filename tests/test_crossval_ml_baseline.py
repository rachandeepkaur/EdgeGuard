"""
Tests for defender/crossval_ml_baseline.py.

Uses the FAKE ROAD folder and manifest from conftest.py (the same fixture
tests/test_defender_crossval.py uses for attack_cv), so these numbers are
directly comparable in spirit. Never touches the real dataset.
"""

import json

import pytest

from defender.crossval_ml_baseline import _examples, cross_validate, main
from defender.defender import Defender
from defender.stage1 import Stage1Model
from defender.stage2 import Stage2Model
from part1.pipeline import RoadData

TRAIN = ["ambient_a", "ambient_b", "ambient_c"]
VALIDATION = ["ambient_d", "ambient_e"]


def quiet(*_):
    pass


def road_for(fake):
    return RoadData(fake.data_dir, manifest=fake.manifest)


def fake_v2(road):
    """A real, fitted stage1/stage2 (watching 0D0, where the fake attacks
    live -- see conftest.py) wrapped as a Defender, purely so
    crossval_ml_baseline can borrow it as a feature extractor exactly the
    way it borrows the real models/stage{1,2}_v2.json in production."""
    windows = [w for n in TRAIN + VALIDATION for w in road.windows(n)]
    stage1 = Stage1Model().fit(windows)
    stage2 = Stage2Model(watch_ids=["0D0"]).fit(windows)
    return Defender(stage1=stage1, threshold=0.99, model_version="v2", stage2=stage2)


def test_examples_skips_masquerade_normal_windows(fake_road):
    road = road_for(fake_road)
    v2 = fake_v2(road)
    features, labels, meta = _examples(
        road, v2, ["speed_attack_1", "speed_attack_1_masquerade"])
    masquerade_rows = [m for m, y in zip(meta, labels) if m["kind"] == "masquerade"]
    assert masquerade_rows and all(
        y for m, y in zip(meta, labels) if m["kind"] == "masquerade"
    )
    assert len(features) == len(labels) == len(meta)


def test_cross_validate_only_reads_train_validation_development(fake_road, monkeypatch):
    road = road_for(fake_road)
    v2 = fake_v2(road)
    read = []
    real_path = road.path
    monkeypatch.setattr(road, "path", lambda name: read.append(name) or real_path(name))

    report = cross_validate(road, v2, max_false_alarm_rate=0.2, log=quiet)

    off_limits = set(fake_road.manifest.names("final_test", "separate"))
    assert read and not (set(read) & off_limits)
    assert report["features"] == ["unknown_ids", "rate_excess", "short_gaps",
                                  "out_of_range", "frozen_break", "large_jump"]
    rows = {r["capture_id"]: r for r in report["per_capture"]}
    assert sorted(rows) == ["cap07", "cap08", "cap09", "cap10", "cap11"]   # development only
    for r in rows.values():
        assert r["attacked"] > 0 or r["family"] == "fuzzing"
    assert "speed" in report["macro_recall_by_family"]
    assert 0.0 <= report["macro_recall_by_family"]["speed"] <= 1.0


def test_cross_validate_needs_at_least_two_folds(fake_road):
    road = road_for(fake_road)
    v2 = fake_v2(road)

    class OneFoldManifest:
        def __getattr__(self, item):
            return getattr(fake_road.manifest, item)

        def development_folds(self):
            return {1: fake_road.manifest.names("development")}

    road.manifest = OneFoldManifest()
    with pytest.raises(ValueError, match="at least 2"):
        cross_validate(road, v2, 0.1, log=quiet)


def test_main_writes_a_report(fake_road, tmp_path):
    import defender.crossval_ml_baseline as mod
    import defender.defender as defender_mod

    road = road_for(fake_road)
    v2 = fake_v2(road)
    (tmp_path / "models").mkdir()
    monkey_defender = staticmethod(lambda model_dir, model_version: v2)
    orig_load = defender_mod.Defender.load
    defender_mod.Defender.load = monkey_defender
    try:
        out = tmp_path / "out" / "crossval_ml_baseline.json"
        rc = mod.main(["--data-dir", str(fake_road.data_dir),
                      "--manifest", str(fake_road.manifest_path),
                      "--max-false-alarm-rate", "0.2",
                      "--v2-model-dir", str(tmp_path / "models"),
                      "--out", str(out)])
        assert rc == 0
        report = json.loads(out.read_text())
        assert report["max_false_alarm_rate"] == 0.2
        assert len(report["per_capture"]) == 5
    finally:
        defender_mod.Defender.load = orig_load
