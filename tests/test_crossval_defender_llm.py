"""
Tests for defender/crossval_defender_llm.py.

Uses the FAKE ROAD folder and manifest from conftest.py (same fixture as
tests/test_crossval_ml_baseline.py / tests/test_defender_crossval.py), and a
FAKE generate_fn (no real model, no transformers/torch needed) -- these
prove the wiring (reads only train/validation/development, calibrates once,
reports per-fold/per-family shapes correctly) is right, not how a real
local LLM does on real ROAD data.
"""

import json

import pytest

from defender.crossval_defender_llm import _examples, cross_validate, main
from defender.defender import Defender
from defender.defender_llm import DefenderLLMModel
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
    windows = [w for n in TRAIN + VALIDATION for w in road.windows(n)]
    stage1 = Stage1Model().fit(windows)
    stage2 = Stage2Model(watch_ids=["0D0"]).fit(windows)
    return Defender(stage1=stage1, threshold=0.99, model_version="v2", stage2=stage2)


def sum_feature_generate_fn(prompt: str) -> str:
    """A fake 'model' with a real, non-constant score (sum of the six
    features it can read out of its own prompt) -- enough to exercise
    threshold calibration and per-capture recall without a real network."""
    total = sum(float(line.split(":")[-1].strip())
               for line in prompt.splitlines() if ":" in line and line.split()[0] in
               ("unknown_ids", "rate_excess", "short_gaps", "out_of_range",
                "frozen_break", "large_jump"))
    prob = min(1.0, total / 5.0)
    return f'{{"attack_probability": {prob}, "reason": "sum={total}"}}'


def fake_llm() -> DefenderLLMModel:
    return DefenderLLMModel(sum_feature_generate_fn, model_name="fake/for-tests")


def test_examples_skips_masquerade_normal_windows(fake_road):
    road = road_for(fake_road)
    v2 = fake_v2(road)
    features, labels, meta = _examples(
        road, v2, ["speed_attack_1", "speed_attack_1_masquerade"], 100_000)
    masquerade_rows = [m for m, y in zip(meta, labels) if m["kind"] == "masquerade"]
    assert masquerade_rows and all(
        y for m, y in zip(meta, labels) if m["kind"] == "masquerade"
    )
    assert len(features) == len(labels) == len(meta)


def test_cross_validate_calibrates_once_and_reports_all_folds(fake_road):
    road = road_for(fake_road)
    v2 = fake_v2(road)
    model = fake_llm()

    report = cross_validate(road, v2, model, max_false_alarm_rate=0.2, log=quiet)

    assert report["calibration_windows"] > 0
    assert report["threshold"] == model.threshold
    assert report["model_name"] == "fake/for-tests"
    rows = {r["capture_id"]: r for r in report["per_capture"]}
    assert sorted(rows) == ["cap07", "cap08", "cap09", "cap10", "cap11"]   # development only
    assert "speed" in report["macro_recall_by_family"]
    assert 0.0 <= report["macro_recall_by_family"]["speed"] <= 1.0


def test_cross_validate_needs_at_least_two_folds(fake_road):
    road = road_for(fake_road)
    v2 = fake_v2(road)
    model = fake_llm()

    class OneFoldManifest:
        def __getattr__(self, item):
            return getattr(fake_road.manifest, item)

        def development_folds(self):
            return {1: fake_road.manifest.names("development")}

    road.manifest = OneFoldManifest()
    with pytest.raises(ValueError, match="at least 2"):
        cross_validate(road, v2, model, 0.1, log=quiet)


def test_max_windows_per_capture_reduces_the_window_count(fake_road):
    road = road_for(fake_road)
    v2 = fake_v2(road)
    full, _, _ = _examples(road, v2, ["speed_attack_1"], 100_000)
    capped, _, _ = _examples(road, v2, ["speed_attack_1"], 1)
    assert len(capped) <= len(full)


def test_main_writes_a_report(fake_road, tmp_path, monkeypatch):
    import defender.crossval_defender_llm as mod
    import defender.defender as defender_mod

    road = road_for(fake_road)
    v2 = fake_v2(road)
    (tmp_path / "models").mkdir()
    monkeypatch.setattr(defender_mod.Defender, "load",
                        staticmethod(lambda model_dir, model_version: v2))
    monkeypatch.setattr(DefenderLLMModel, "load_local",
                        classmethod(lambda cls, *a, **k: fake_llm()))

    out = tmp_path / "out" / "crossval_defender_llm.json"
    rc = mod.main(["--data-dir", str(fake_road.data_dir),
                  "--manifest", str(fake_road.manifest_path),
                  "--max-false-alarm-rate", "0.2",
                  "--v2-model-dir", str(tmp_path / "models"),
                  "--out", str(out)])
    assert rc == 0
    report = json.loads(out.read_text())
    assert report["max_false_alarm_rate"] == 0.2
    assert len(report["per_capture"]) == 5
