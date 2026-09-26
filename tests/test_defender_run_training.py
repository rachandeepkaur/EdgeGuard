"""
Tests for defender/run_training.py.

Run from the EdgeGuard folder with:
    python -m pytest tests/test_defender_run_training.py -v

Uses the FAKE ROAD folder and manifest from conftest.py. Never touches the
real dataset.
"""

import json

import pytest

from defender.defender import Defender
from defender.run_training import main


def run(fake, *extra):
    return main(["--max-false-alarm-rate", "0.0",
                 "--data-dir", str(fake.data_dir),
                 "--manifest", str(fake.manifest_path),
                 "--model-dir", str(fake.data_dir.parent / "models"),
                 "--max-windows-per-capture", "10", *extra])


def info(fake, version):
    path = fake.data_dir.parent / "models" / f"train_info_{version}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_trains_loadable_v1_and_v2(fake_road):
    assert run(fake_road) == 0
    v1 = Defender.load(fake_road.data_dir.parent / "models", "v1")
    v2 = Defender.load(fake_road.data_dir.parent / "models", "v2")
    assert v1.stage2 is None and v2.stage2 is not None
    assert v1.window_s == v2.window_s == 1.0


def test_uses_manifest_train_and_validation_groups(fake_road):
    run(fake_road)
    record = info(fake_road, "v1")
    assert record["train_capture_ids"] == ["cap01", "cap02", "cap03"]
    assert record["validation_capture_ids"] == ["cap04", "cap05"]


def test_records_non_overlapping_windows(fake_road):
    run(fake_road)
    record = info(fake_road, "v1")
    assert record["window_s"] == 1.0 and record["stride_s"] == 1.0


def test_window_cap_is_respected(fake_road):
    run(fake_road)
    record = info(fake_road, "v1")
    assert record["train_windows"] <= 3 * 10
    assert record["validation_windows"] <= 2 * 10


def test_model_files_hold_no_capture_names(fake_road):
    run(fake_road)
    model_files = " ".join(p.read_text(encoding="utf-8")
                           for p in (fake_road.data_dir.parent / "models").glob("*.json"))
    assert "ambient" not in model_files and "attack" not in model_files


def test_rerun_refuses_to_overwrite(fake_road):
    run(fake_road)
    with pytest.raises(FileExistsError):
        run(fake_road)


def test_overwrite_flag_allows_rerun(fake_road):
    run(fake_road)
    assert run(fake_road, "--overwrite") == 0


def test_false_alarm_rate_is_required(fake_road):
    with pytest.raises(SystemExit):
        main(["--data-dir", str(fake_road.data_dir)])


def test_missing_capture_gives_clear_error(fake_road):
    (fake_road.data_dir / "ambient" / "ambient_e.log").unlink()
    with pytest.raises(SystemExit):
        run(fake_road)


def test_default_watch_list_is_recorded(fake_road):
    run(fake_road)
    assert info(fake_road, "v2")["stage2_watch_ids"] == ["0D0", "6E0"]


def test_watch_all_option(fake_road):
    run(fake_road, "--stage2-watch", "all")
    assert info(fake_road, "v2")["stage2_watch_ids"] is None


def test_harden_v3_leaves_v1_and_v2_untouched(fake_road):
    run(fake_road)
    models = fake_road.data_dir.parent / "models"
    before = {p.name: p.read_text(encoding="utf-8") for p in models.glob("*_v[12].json")}
    assert run(fake_road, "--harden-v3") == 0
    after = {p.name: p.read_text(encoding="utf-8") for p in models.glob("*_v[12].json")}
    assert before == after
    record = info(fake_road, "v3")
    assert record["stage2_frozen_mode"] == "rate" and record["base_stage1_version"] == "v1"


def test_harden_v4_leaves_v1_and_v2_untouched(fake_road):
    run(fake_road)
    models = fake_road.data_dir.parent / "models"
    before = {p.name: p.read_text(encoding="utf-8") for p in models.glob("*_v[12].json")}
    assert run(fake_road, "--harden-v4", "--bound-percentile", "10") == 0
    after = {p.name: p.read_text(encoding="utf-8") for p in models.glob("*_v[12].json")}
    assert before == after
    record = info(fake_road, "v4")
    assert record["stage2_frozen_mode"] == "width" and record["base_stage1_version"] == "v1"
    assert record["stage2_bound_percentile"] == 10.0


def test_harden_v4_requires_bound_percentile(fake_road):
    run(fake_road)
    with pytest.raises(SystemExit):
        run(fake_road, "--harden-v4")


def test_harden_v3_and_v4_are_mutually_exclusive(fake_road):
    run(fake_road)
    with pytest.raises(SystemExit):
        run(fake_road, "--harden-v3", "--harden-v4", "--bound-percentile", "10")


def test_v2_bound_percentile_defaults_to_none(fake_road):
    run(fake_road)
    assert info(fake_road, "v2")["stage2_bound_percentile"] is None


def test_harden_v5_leaves_v1_and_v2_untouched(fake_road):
    run(fake_road)
    models = fake_road.data_dir.parent / "models"
    before = {p.name: p.read_text(encoding="utf-8") for p in models.glob("*_v[12].json")}
    assert run(fake_road, "--harden-v5", "--stage2-watch", "0D0") == 0
    after = {p.name: p.read_text(encoding="utf-8") for p in models.glob("*_v[12].json")}
    assert before == after
    record = info(fake_road, "v5")
    assert record["stage2_frozen_mode"] == "width" and record["base_stage1_version"] == "v1"
    assert record["stage2_bound_percentile"] is None
    assert record["stage2_watch_ids"] == ["0D0"]


def test_harden_v5_requires_a_different_watch_list(fake_road):
    run(fake_road)
    with pytest.raises(SystemExit):
        run(fake_road, "--harden-v5")  # default --stage2-watch is 0D0,6E0: no-op


def test_harden_v3_v4_v5_are_mutually_exclusive(fake_road):
    run(fake_road)
    with pytest.raises(SystemExit):
        run(fake_road, "--harden-v3", "--harden-v5", "--stage2-watch", "0D0")
    with pytest.raises(SystemExit):
        run(fake_road, "--harden-v4", "--harden-v5", "--bound-percentile", "10",
           "--stage2-watch", "0D0")
