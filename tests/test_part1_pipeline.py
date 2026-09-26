"""
Tests for part1/pipeline.py and part1/reproduce.py (Block 1 completion check).

Run from the EdgeGuard folder with:
    python -m pytest tests/test_part1_pipeline.py -v
"""

import json

import pytest

from part1.cleaning import CleaningReport
from part1.decode import Decoder
from part1.pipeline import RoadData, preprocess
from part1.reproduce import compare, fingerprint, main


def road_for(fake, **kwargs):
    return RoadData(fake.data_dir, manifest=fake.manifest, **kwargs)


# ---------- Windows ----------------------------------------------------
def test_windows_are_one_second_non_overlapping_with_neutral_ids(fake_road):
    windows = list(road_for(fake_road).windows("ambient_a"))
    assert len(windows) == 19                          # 20 s capture, partial last dropped
    assert all(w.window_end - w.window_start == 1.0 for w in windows)
    assert all(a.window_end == b.window_start for a, b in zip(windows, windows[1:]))
    assert windows[0].window_id == "cap01_w0000" and windows[0].vehicle_id == "veh01"
    assert windows[0].window_start == 0.0              # elapsed, not raw log time
    assert windows[0].features == {} and windows[0].decoded_signals == {}


def test_keep_every_samples_training_windows(fake_road):
    road = road_for(fake_road)
    step = road.keep_every("ambient_a", 5)
    kept = list(road.windows("ambient_a", keep_every=step))
    assert step == 4 and len(kept) == 5
    assert [w.window_id for w in kept][:2] == ["cap01_w0000", "cap01_w0004"]


def test_frames_are_cleaned(fake_road):
    log = fake_road.data_dir / "ambient" / "ambient_a.log"
    lines = log.read_text().splitlines()
    log.write_text("\n".join(lines[:5] + [lines[4]] + lines[5:]) + "\n")   # one duplicate
    report = CleaningReport()
    list(road_for(fake_road).windows("ambient_a", report=report))
    assert report.duplicates_dropped == 1


def test_final_test_is_refused_unless_final_evaluation(fake_road):
    with pytest.raises(ValueError, match="only the final evaluation"):
        list(road_for(fake_road).windows("ambient_f"))
    assert list(road_for(fake_road, final_evaluation=True).windows("ambient_f"))


# ---------- Labels -------------------------------------------------------
def labels_of(fake_road, name):
    return [label for _, label in road_for(fake_road).labelled_windows(name)]


def test_ambient_windows_are_all_normal(fake_road):
    assert not any(l.is_attack for l in labels_of(fake_road, "ambient_a"))


def test_attack_windows_are_those_with_injected_frames(fake_road):
    for name in ("speed_attack_1", "speed_attack_1_masquerade"):
        attacked = [l.window_id for l in labels_of(fake_road, name) if l.is_attack]
        prefix = fake_road.manifest.entry(name).capture_id
        assert attacked == [f"{prefix}_w{i:04d}" for i in range(8, 12)], name


def test_fabrication_and_masquerade_twins_agree(fake_road):
    """The P2 gate idea: same recording, same number of injected frames."""
    fab = sum(l.injected_frames for l in labels_of(fake_road, "speed_attack_1"))
    masq = sum(l.injected_frames for l in labels_of(fake_road, "speed_attack_1_masquerade"))
    assert fab == masq > 0


def test_labels_and_windows_share_window_ids(fake_road):
    for window, label in road_for(fake_road).labelled_windows("fuzzing_attack_1"):
        assert window.window_id == label.window_id
    assert any(l.is_attack and l.family == "fuzzing" for l in labels_of(fake_road, "fuzzing_attack_1"))


def test_accelerator_has_no_attack_windows(fake_road):
    road = road_for(fake_road, final_evaluation=True)
    assert not any(l.is_attack for _, l in road.labelled_windows("accelerator_attack_drive_1"))


def test_metadata_record(fake_road):
    meta = road_for(fake_road).metadata("speed_attack_2_masquerade")
    assert meta.capture_id == "cap10" and meta.vehicle_id == "veh01"
    assert meta.kind == "masquerade" and meta.target == "0D0"
    assert meta.injection_interval == [8.0, 11.99]
    assert meta.first_raw_timestamp == "1110000000.000000"
    assert 19.9 < meta.duration_s <= 20.0


# ---------- preprocess ---------------------------------------------------
def test_preprocess_fills_decoded_signals_on_a_copy(fake_road):
    window = next(road_for(fake_road).windows("speed_attack_1_masquerade", keep_every=9))
    decoder = Decoder.from_data_dir(fake_road.data_dir, watch_ids=["0D0"])
    out = preprocess(window, decoder)
    assert window.decoded_signals == {}                        # original untouched
    n_0d0 = sum(1 for f in window.frames if f.can_id == "0D0")
    assert len(out.decoded_signals["0D0:Unknown_0"]) == n_0d0
    assert out.frames == window.frames and out.window_id == window.window_id


def test_preprocess_without_decoder_is_a_plain_copy(fake_road):
    window = next(road_for(fake_road).windows("ambient_a"))
    assert preprocess(window) == window and preprocess(window) is not window


# ---------- Reproducibility (Block 1 completion check) --------------------
def test_pipeline_is_reproducible(fake_road):
    first = fingerprint(road_for(fake_road), log=lambda *_: None)
    second = fingerprint(road_for(fake_road), log=lambda *_: None)
    assert first == second and compare(first, second) == []
    ids = [r["capture_id"] for r in first["captures"]]
    assert "cap06" not in ids and "cap12" not in ids           # final_test not read


def test_fingerprint_detects_a_changed_capture(fake_road):
    before = fingerprint(road_for(fake_road), ["ambient_a"], log=lambda *_: None)
    log = fake_road.data_dir / "ambient" / "ambient_a.log"
    log.write_text(log.read_text().replace("0102030405060708", "0102030405060709", 1))
    after = fingerprint(road_for(fake_road), ["ambient_a"], log=lambda *_: None)
    assert any("windows_sha256" in d for d in compare(before, after))


def test_reproduce_cli_writes_and_checks(fake_road, tmp_path, monkeypatch):
    monkeypatch.setattr("part1.pipeline.load_manifest", lambda: fake_road.manifest)
    out = tmp_path / "fp.json"
    assert main(["--data-dir", str(fake_road.data_dir), "--out", str(out)]) == 0
    assert "ambient" not in out.read_text()                    # neutral ids only
    assert main(["--data-dir", str(fake_road.data_dir), "--check", str(out)]) == 0
    data = json.loads(out.read_text())
    data["captures"][0]["windows_sha256"] = "0" * 64
    out.write_text(json.dumps(data))
    assert main(["--data-dir", str(fake_road.data_dir), "--check", str(out)]) == 1
