"""
Tests for part1/split_manifest.py, including the COMMITTED manifest itself.

Run from the EdgeGuard folder with:
    python -m pytest tests/test_part1_split_manifest.py -v
"""

import json

import pytest

from part1.split_manifest import (CaptureEntry, Manifest, build_manifest, classify,
                                  load_manifest)


def entry(name, cid, group="train", kind="ambient", **extra):
    folder = "ambient" if kind == "ambient" else "attacks"
    return CaptureEntry(name=name, capture_id=cid, folder=folder, group=group, kind=kind, **extra)


# ---------- The committed manifest (part1/split_manifest.json) ----------
def test_committed_manifest_loads_and_has_45_captures():
    m = load_manifest()
    assert len(m.captures) == 45 and m.vehicle_id == "veh01"


def test_committed_manifest_groups():
    m = load_manifest()
    assert len(m.names("train")) == 7
    assert m.names("validation") == ["ambient_dyno_drive_benign_anomaly",
                                     "ambient_dyno_drive_extended_short"]
    assert all(n.startswith("accelerator") for n in m.names("separate"))
    for name in m.names("development"):
        assert name.split("_masquerade")[0][-2:] in ("_1", "_2")


def test_committed_manifest_keeps_test_out_of_development_and_training():
    m = load_manifest()
    test = set(m.names("final_test"))
    assert "max_engine_coolant_temp_attack" in test          # the unwatched-target test
    for name in test:
        if not name.startswith("ambient"):
            assert "_3" in name or "coolant" in name
    assert not test & set(m.names("train", "validation", "development"))


def test_committed_manifest_development_folds_cover_every_family():
    m = load_manifest()
    folds = m.development_folds()
    assert sorted(folds) == [1, 2]
    families = [{m.entry(n).family for n in names if m.entry(n).kind != "fuzzing"}
                for names in folds.values()]
    assert families[0] == families[1]                        # every family on both sides


# ---------- Rules enforced on load -------------------------------------
def test_twins_must_share_a_group():
    with pytest.raises(ValueError, match="must share a split"):
        Manifest(vehicle_id="veh01", captures=[
            entry("x_attack_1", "cap01", "development", "fabrication", twin="x_attack_1_masquerade", fold=1),
            entry("x_attack_1_masquerade", "cap02", "final_test", "masquerade", twin="x_attack_1")])


def test_ids_must_be_unique_and_neutral():
    with pytest.raises(ValueError, match="share a capture_id"):
        Manifest(vehicle_id="veh01", captures=[entry("a", "cap01"), entry("b", "cap01")])
    with pytest.raises(ValueError, match="not neutral"):
        Manifest(vehicle_id="veh01", captures=[entry("a", "ambient_a")])


def test_only_development_attacks_have_folds():
    with pytest.raises(ValueError, match="fold"):
        Manifest(vehicle_id="veh01", captures=[entry("a", "cap01", fold=1)])


def test_capture_path_refuses_final_test_and_separate(fake_road):
    m = fake_road.manifest
    for name in ("ambient_f", "speed_attack_3", "accelerator_attack_drive_1"):
        with pytest.raises(ValueError, match="only the final evaluation"):
            m.capture_path(fake_road.data_dir, name)
    assert m.capture_path(fake_road.data_dir, "speed_attack_3", final_evaluation=True).exists()
    assert m.capture_path(fake_road.data_dir, "ambient_a").name == "ambient_a.log"


# ---------- Building ------------------------------------------------------
def test_classify_kinds_and_families():
    assert classify("ambient_dyno_reverse", {}) == ("ambient", None, None, None)
    assert classify("max_speedometer_attack_2", {"injection_id": "0xd0", "modified": False}) == \
        ("fabrication", "max_speedometer", 2, "max_speedometer_attack_2_masquerade")
    assert classify("max_speedometer_attack_2_masquerade", {"injection_id": "0xd0", "modified": True}) == \
        ("masquerade", "max_speedometer", 2, "max_speedometer_attack_2")
    assert classify("fuzzing_attack_3", {"injection_id": "XXX"})[:3] == ("fuzzing", "fuzzing", 3)
    assert classify("accelerator_attack_reverse_1", {"injection_id": None})[:3] == \
        ("accelerator", "accelerator", 1)
    assert classify("max_engine_coolant_temp_attack", {"injection_id": "0x4e7"})[1:3] == \
        ("max_engine_coolant_temp", None)


def test_build_manifest_from_splits_json(tmp_path):
    road = tmp_path / "road"
    for folder, records in {
        "ambient": {"ambient_x": {}, "ambient_y": {}, "ambient_z": {}},
        "attacks": {"spd_attack_1": {"injection_id": "0xd0", "modified": False},
                    "spd_attack_1_masquerade": {"injection_id": "0xd0", "modified": True},
                    "spd_attack_3": {"injection_id": "0xd0", "modified": False},
                    "accelerator_attack_drive_1": {"injection_id": None}}}.items():
        (road / folder).mkdir(parents=True)
        (road / folder / "capture_metadata.json").write_text(json.dumps(records))
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"schema_version": 2, "train": {"ambient": ["ambient_x"]},
                                  "val": {"ambient": ["ambient_y"]},
                                  "test": {"ambient": ["ambient_z"]}}))
    m = build_manifest(splits, road)
    groups = {c.name: (c.capture_id, c.group, c.fold) for c in m.captures}
    assert groups == {
        "ambient_x": ("cap01", "train", None), "ambient_y": ("cap02", "validation", None),
        "ambient_z": ("cap03", "final_test", None),
        "accelerator_attack_drive_1": ("cap04", "separate", None),
        "spd_attack_1": ("cap05", "development", 1),
        "spd_attack_1_masquerade": ("cap06", "development", 1),
        "spd_attack_3": ("cap07", "final_test", None)}
    assert m.entry("spd_attack_3").twin is None          # no masquerade version exists
