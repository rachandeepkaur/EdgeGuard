"""
Tests for defender/crossval.py.

Run from the EdgeGuard folder with:
    python -m pytest tests/test_defender_crossval.py -v

Uses the FAKE ROAD folder and manifest from conftest.py. Never touches the
real dataset.
"""

import json

import pytest

from defender import crossval
from defender.crossval import (ambient_loco, attack_cv, binomial_upper_95, fit_models,
                               fold_watch_ids, load_ambient, main)
from defender.stage2 import Stage2Model
from part1.pipeline import RoadData

TRAIN_VAL = ["ambient_a", "ambient_b", "ambient_c", "ambient_d", "ambient_e"]


def quiet(*_):
    pass


def road_for(fake):
    return RoadData(fake.data_dir, manifest=fake.manifest)


# ---------- Statistics ------------------------------------------------
def test_binomial_upper_zero_alarms_matches_closed_form():
    # k = 0: upper bound is 1 - 0.05 ** (1 / n)
    assert binomial_upper_95(0, 100) == pytest.approx(1 - 0.05 ** (1 / 100), rel=1e-6)


def test_binomial_upper_is_above_observed_rate():
    assert 5 / 200 < binomial_upper_95(5, 200) < 0.06
    assert binomial_upper_95(3, 3) == 1.0 and binomial_upper_95(0, 0) == 1.0


# ---------- Stage 2 precomputed range stats ---------------------------
def test_range_stats_give_the_same_stage2_model(fake_road):
    road = road_for(fake_road)
    windows, stats = load_ambient(road, TRAIN_VAL[:2], 10, ["0D0"], quiet)
    train = windows["ambient_a"] + windows["ambient_b"]
    a = Stage2Model(watch_ids=["0D0"]).fit(
        train, range_captures=[road.frames(n) for n in TRAIN_VAL[:2]])
    b = Stage2Model(watch_ids=["0D0"]).fit(train, range_stats=[stats[n] for n in TRAIN_VAL[:2]])
    assert a.field_range == b.field_range
    assert a.field_max_jump == b.field_max_jump
    assert a.reference == b.reference


# ---------- Protocol 1: normal-drive LOCO ------------------------------
def test_loco_never_fits_on_the_held_out_capture(fake_road, monkeypatch):
    names = TRAIN_VAL[:4]
    windows, stats = load_ambient(road_for(fake_road), names, 10, ["0D0"], quiet)
    fits = []
    real_fit = crossval.fit_models

    def spy(fit_names, *args):
        fits.append(list(fit_names))
        return real_fit(fit_names, *args)

    monkeypatch.setattr(crossval, "fit_models", spy)
    report = ambient_loco(names, windows, stats, 0.1, ["0D0"], log=quiet)

    per_outer = len(names)            # (len - 1) inner fits + 1 outer fit per held-out capture
    assert len(fits) == len(names) * per_outer
    for i, held_out in enumerate(names):
        for fit_names in fits[i * per_outer:(i + 1) * per_outer]:
            assert held_out not in fit_names
    assert [r["capture_id"] for r in report["per_capture"]] == ["cap01", "cap02", "cap03", "cap04"]
    for v in ("v1", "v2"):
        s = report["summary"][v]
        assert s["windows"] == sum(len(windows[n]) for n in names)
        assert 0 <= s["alarms"] <= s["windows"]
        assert s["normal_hours"] == pytest.approx(s["windows"] / 3600, abs=1e-4)


def test_loco_needs_three_captures():
    with pytest.raises(ValueError, match="at least 3"):
        ambient_loco(["a", "b"], {}, {}, 0.01, None, log=quiet)


# ---------- Protocol 2: attack 2-fold CV --------------------------------
def test_fold_watch_ids_come_from_targets_and_skip_fuzzing(fake_road):
    road = road_for(fake_road)
    assert fold_watch_ids(road, ["speed_attack_1", "fuzzing_attack_1"]) == ["0D0"]
    with pytest.raises(ValueError, match="no targeted attacks"):
        fold_watch_ids(road, ["fuzzing_attack_1"])


def test_attack_cv_uses_part1_labels_and_counts_masquerade_normal_windows_once(fake_road):
    report = attack_cv(road_for(fake_road), 0.1, 10, log=quiet)
    assert [f["watch_ids"] for f in report["folds"]] == [["0D0"], ["0D0"]]
    rows = {r["capture_id"]: r for r in report["per_capture"]}
    assert sorted(rows) == ["cap07", "cap08", "cap09", "cap10", "cap11"]   # development only
    for r in rows.values():
        assert r["attacked"] > 0
        if r["condition"] == "masquerade":
            assert r["normal"] == 0          # its normal windows are the twin's
        else:
            assert r["normal"] > 0
    # byte 5 = FF is outside the normal 00 range, so v2 must catch it
    assert report["summary"]["v2"]["fabrication"]["macro_recall"] == 1.0


def test_attack_cv_never_reads_final_test(fake_road, monkeypatch):
    road = road_for(fake_road)
    read = []
    real_path = road.path
    monkeypatch.setattr(road, "path", lambda name: read.append(name) or real_path(name))
    attack_cv(road, 0.1, 10, log=quiet)
    assert read and not set(read) & set(fake_road.manifest.names("final_test", "separate"))


# ---------- Command line ----------------------------------------------
def cli(fake, protocol, out, *extra):
    return main([protocol, "--max-false-alarm-rate", "0.1",
                 "--data-dir", str(fake.data_dir), "--manifest", str(fake.manifest_path),
                 "--max-windows-per-capture", "10", "--out", str(out), *extra])


def test_main_ambient_writes_report(fake_road, tmp_path):
    out = tmp_path / "out" / "loco.json"
    assert cli(fake_road, "ambient", out, "--stage2-watch", "0D0") == 0
    report = json.loads(out.read_text())
    assert report["captures"] == 5            # 3 train + 2 validation, never test
    assert report["settings"]["stage2_watch_ids"] == ["0D0"]
    assert "ambient" not in json.dumps(report["per_capture"])      # neutral ids only


def test_main_attacks_writes_report(fake_road, tmp_path):
    out = tmp_path / "out" / "attacks.json"
    assert cli(fake_road, "attacks", out) == 0
    assert len(json.loads(out.read_text())["per_capture"]) == 5


# ---------- bound_percentile threading ---------------------------------
def test_fit_models_bound_percentile_narrows_stage2_bounds(fake_road):
    road = road_for(fake_road)
    windows, stats = load_ambient(road, TRAIN_VAL, 10, ["0D0"], quiet)
    _, stage2_none = fit_models(TRAIN_VAL, windows, stats, ["0D0"], "width", None)
    _, stage2_p = fit_models(TRAIN_VAL, windows, stats, ["0D0"], "width", 10.0)
    assert stage2_p.bound_percentile == 10.0
    assert stage2_none.bound_percentile is None
    # Same full range either way; only the out-of-range bounds narrow.
    assert stage2_p.field_range == stage2_none.field_range
    assert stage2_p.field_bounds != stage2_none.field_bounds


def test_main_ambient_records_bound_percentile_in_settings(fake_road, tmp_path):
    out = tmp_path / "out" / "loco.json"
    assert cli(fake_road, "ambient", out, "--stage2-watch", "0D0",
              "--bound-percentile", "10") == 0
    report = json.loads(out.read_text())
    assert report["settings"]["bound_percentile"] == 10.0


def test_main_ambient_bound_percentile_defaults_to_none(fake_road, tmp_path):
    out = tmp_path / "out" / "loco.json"
    assert cli(fake_road, "ambient", out, "--stage2-watch", "0D0") == 0
    report = json.loads(out.read_text())
    assert report["settings"]["bound_percentile"] is None
