"""
Tests for defender/crossval_fused_timing_cnn.py. Needs PyTorch -- run from
the machine that has it:
    .venv/bin/python3 -m pytest tests/test_crossval_fused_timing_cnn.py -v

Uses the FAKE ROAD folder and manifest from conftest.py, same as the other
crossval test files. Never touches the real dataset.
"""

import json

import pytest

torch = pytest.importorskip("torch")

from defender.crossval_fused_timing_cnn import cross_validate, main
from defender.defender import Defender
from defender.stage1 import Stage1Model
from defender.stage2 import Stage2Model
from part1.pipeline import RoadData

TRAIN = ["ambient_a", "ambient_b", "ambient_c"]
VALIDATION = ["ambient_d", "ambient_e"]
WATCH_IDS = ("0D0",)


def quiet(*_):
    pass


def road_for(fake):
    return RoadData(fake.data_dir, manifest=fake.manifest)


def fake_v2(road):
    windows = [w for n in TRAIN + VALIDATION for w in road.windows(n)]
    stage1 = Stage1Model().fit(windows)
    stage2 = Stage2Model(watch_ids=["0D0"]).fit(windows)
    from defender.threshold import choose_threshold
    from defender.fusion import fuse
    scores = [fuse(stage1.score(w), stage2.score(w)).score for w in windows]
    threshold = choose_threshold(scores, 0.2)
    return Defender(stage1=stage1, threshold=threshold, model_version="v2", stage2=stage2)


def test_cross_validate_only_reads_train_validation_development(fake_road, monkeypatch):
    road = road_for(fake_road)
    v2 = fake_v2(road)
    read = []
    real_path = road.path
    monkeypatch.setattr(road, "path", lambda name: read.append(name) or real_path(name))

    report = cross_validate(road, v2, WATCH_IDS, max_false_alarm_rate=0.2, epochs=10, log=quiet)

    off_limits = set(fake_road.manifest.names("final_test", "separate"))
    assert read and not (set(read) & off_limits)
    rows = {r["capture_id"]: r for r in report["per_capture"]}
    assert sorted(rows) == ["cap07", "cap08", "cap09", "cap10", "cap11"]
    for r in rows.values():
        # fused can never detect FEWER attacks than either component alone
        assert r["fused_detected"] >= r["v2_detected"]
        assert r["fused_detected"] >= r["cnn_detected"]
        assert r["fused_false"] >= r["v2_false"]
        assert r["fused_false"] >= r["cnn_false"]
    for name in ("v2", "cnn", "fused"):
        assert 0.0 <= report["summary"][name]["overall_recall"] <= 1.0


def test_fused_never_detects_less_than_the_max_of_its_components(fake_road):
    road = road_for(fake_road)
    v2 = fake_v2(road)
    report = cross_validate(road, v2, WATCH_IDS, max_false_alarm_rate=0.2, epochs=10, log=quiet)
    for name in ("v2", "cnn"):
        assert (report["summary"]["fused"]["overall_recall"]
               >= report["summary"][name]["overall_recall"] - 1e-9)


def test_main_writes_a_report(fake_road, tmp_path, monkeypatch):
    import defender.crossval_fused_timing_cnn as mod
    import defender.defender as defender_mod

    road = road_for(fake_road)
    v2 = fake_v2(road)
    (tmp_path / "models").mkdir()
    monkeypatch.setattr(defender_mod.Defender, "load", staticmethod(lambda d, v: v2))

    out = tmp_path / "out" / "fused.json"
    rc = mod.main(["--data-dir", str(fake_road.data_dir),
                  "--manifest", str(fake_road.manifest_path),
                  "--max-false-alarm-rate", "0.2",
                  "--watch-ids", "0D0",
                  "--epochs", "10",
                  "--v2-model-dir", str(tmp_path / "models"),
                  "--out", str(out)])
    assert rc == 0
    report = json.loads(out.read_text())
    assert set(report["summary"]) == {"v2", "cnn", "fused"}
    assert len(report["per_capture"]) == 5
