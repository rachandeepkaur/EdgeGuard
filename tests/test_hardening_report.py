"""Tests for part3/hardening_report.py: build_hardening_set() wired to a
real evasion log and manifest, plus the family/watch-coverage diagnosis.

Run from EdgeGuard: python -m pytest tests/test_hardening_report.py -v
"""

import json
from dataclasses import replace

import pytest

from part1.pipeline import RoadData
from part3.evasion_log import EvasionEntry
from part3.hardening_report import (build_real_hardening_set, development_window_variant_ids,
                                    diagnose, load_evasions, main)
from part3.hardening_set import HardeningSet


def road_for(fake):
    return RoadData(fake.data_dir, manifest=fake.manifest)


def dev_window_id(fake, name="speed_attack_1"):
    road = road_for(fake)
    return next(iter(road.windows(name))).window_id


def evasion(window_id, family="freeze", target_can_id="0D0", model_version="v2"):
    return EvasionEntry(
        attack_id="atk_001", window_id=window_id, family=family, target_can_id=target_can_id,
        start_offset_ms=0.0, duration_ms=1000.0, attack_score=0.9,
        model_version=model_version, split="development",
    )


# ---------------------------------------------------------------------
# load_evasions
# ---------------------------------------------------------------------

def test_load_evasions_round_trips_jsonl(tmp_path):
    entries = [evasion("cap07_w0001_v01"), evasion("cap07_w0002_v02", family="offset")]
    path = tmp_path / "evasions.jsonl"
    path.write_text("\n".join(json.dumps(vars(e)) for e in entries), encoding="utf-8")
    loaded = load_evasions(path)
    assert loaded == entries


def test_load_evasions_skips_blank_lines(tmp_path):
    path = tmp_path / "evasions.jsonl"
    path.write_text(json.dumps(vars(evasion("cap07_w0001_v01"))) + "\n\n", encoding="utf-8")
    assert len(load_evasions(path)) == 1


# ---------------------------------------------------------------------
# development_window_variant_ids / build_real_hardening_set
# ---------------------------------------------------------------------

def test_variant_ids_cover_every_development_window_and_family(fake_road):
    road = road_for(fake_road)
    ids = development_window_variant_ids(road, families=("freeze", "offset"))
    base = dev_window_id(fake_road)
    assert f"{base}_v01" in ids
    assert f"{base}_v02" in ids
    dev_windows = sum(1 for name in road.manifest.names("development") for _ in road.windows(name))
    assert len(ids) == dev_windows * 2


def test_build_real_hardening_set_accepts_a_real_confirmed_miss(fake_road):
    road = road_for(fake_road)
    base = dev_window_id(fake_road)
    miss = evasion(f"{base}_v01")
    hs = build_real_hardening_set([miss], road, baseline_model_version="v2")
    assert isinstance(hs, HardeningSet)
    assert hs.window_ids == (f"{base}_v01",)
    assert hs.baseline_model_version == "v2"


def test_build_real_hardening_set_rejects_a_window_not_in_development(fake_road):
    road = road_for(fake_road)
    # ambient_a is train, not development: real window ids exist but none
    # carry a development variant suffix, so this must be refused.
    with pytest.raises(ValueError, match="not in the development manifest"):
        build_real_hardening_set([evasion("cap01_w0000_v01")], road, baseline_model_version="v2")


# ---------------------------------------------------------------------
# diagnose
# ---------------------------------------------------------------------

def test_diagnose_splits_coverage_from_sensitivity_gaps():
    misses = [
        evasion("w1_v01", family="freeze", target_can_id="0D0"),   # watched, still missed
        evasion("w2_v01", family="freeze", target_can_id="033"),   # not watched
        evasion("w3_v02", family="offset", target_can_id="006"),   # not watched
    ]
    report = diagnose(misses, watch_ids=["0D0", "6E0"])
    assert report["total_misses"] == 3
    assert report["by_family"] == {"freeze": 2, "offset": 1}
    assert report["coverage_gap_misses"] == 2
    assert set(report["coverage_gap_targets"]) == {"033", "006"}
    assert report["sensitivity_gap_misses"] == 1
    assert report["sensitivity_gap_by_family"] == {"freeze": 1}


def test_diagnose_with_watch_ids_none_attributes_everything_to_sensitivity():
    misses = [evasion("w1_v01", target_can_id="0D0"), evasion("w2_v01", target_can_id="033")]
    report = diagnose(misses, watch_ids=None)
    assert report["watch_ids"] == "all"
    assert report["coverage_gap_misses"] == 0
    assert report["sensitivity_gap_misses"] == 2


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def cli_log(tmp_path, entries):
    path = tmp_path / "evasions.jsonl"
    path.write_text("\n".join(json.dumps(vars(e)) for e in entries), encoding="utf-8")
    return path


def test_cli_reports_and_saves(fake_road, tmp_path, capsys):
    base = dev_window_id(fake_road)
    log_path = cli_log(tmp_path, [evasion(f"{base}_v01")])
    out_path = tmp_path / "hardening_report.json"

    assert main(["--data-dir", str(fake_road.data_dir), "--manifest", str(fake_road.manifest_path),
                "--baseline-version", "v2", "--watch-ids", "0D0,6E0",
                "--out", str(out_path), str(log_path)]) == 0

    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["total_misses"] == 1
    assert saved["hardening_set"]["window_ids"] == [f"{base}_v01"]

    output = capsys.readouterr().out
    assert "HardeningSet" in output


def test_cli_rejects_a_log_with_the_wrong_baseline_version(fake_road, tmp_path):
    base = dev_window_id(fake_road)
    log_path = cli_log(tmp_path, [evasion(f"{base}_v01", model_version="v1")])
    with pytest.raises(SystemExit):
        main(["--data-dir", str(fake_road.data_dir), "--manifest", str(fake_road.manifest_path),
             "--baseline-version", "v2", str(log_path)])
