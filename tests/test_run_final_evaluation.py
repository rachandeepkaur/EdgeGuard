"""Tests for integration/run_final_evaluation.py (build plan S9, Block 5).

Uses the FAKE ROAD folder and manifest from conftest.py, and trains real
v1/v2 models on it with defender.run_training -- never the real dataset.
final_test in the fake manifest is ["ambient_f", "speed_attack_3",
"speed_attack_3_masquerade"] (tests/conftest.py's CAPTURES list).

Run from EdgeGuard: python -m pytest tests/test_run_final_evaluation.py -v
"""

import json

import pytest

from defender.defender import Defender
from defender.run_training import main as train_main
from integration.run_final_evaluation import (FinalEvaluationReport, main,
                                              run_final_evaluation)
from part1.split_manifest import Manifest, load_manifest


def train(fake):
    model_dir = fake.data_dir.parent / "models"
    assert train_main(["--max-false-alarm-rate", "0.0",
                       "--data-dir", str(fake.data_dir),
                       "--manifest", str(fake.manifest_path),
                       "--model-dir", str(model_dir),
                       "--max-windows-per-capture", "10"]) == 0
    return model_dir


def manifest_without_final_test(fake, tmp_path):
    manifest = load_manifest(fake.manifest_path)
    filtered = [c for c in manifest.captures if c.group != "final_test"]
    no_final = Manifest(vehicle_id=manifest.vehicle_id, captures=filtered)
    path = tmp_path / "no_final_manifest.json"
    path.write_text(no_final.model_dump_json(indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------
# run_final_evaluation
# ---------------------------------------------------------------------

def test_reports_the_real_v1_vs_v2_gap_on_final_test(fake_road):
    """v1 is Stage 1 (timing) only and misses half the speed-family attacks
    on final_test; v2 adds Stage 2 (payload) and catches them all. This is
    the exact evidence Block 5 exists to produce."""
    model_dir = train(fake_road)
    v1 = Defender.load(model_dir, "v1")
    v2 = Defender.load(model_dir, "v2")

    report = run_final_evaluation(v1, v2, fake_road.data_dir, fake_road.manifest_path)

    assert isinstance(report, FinalEvaluationReport)
    assert report.final_test_captures == [
        "ambient_f", "speed_attack_3", "speed_attack_3_masquerade",
    ]
    assert report.windows == 57
    assert report.baseline_version == "v1"
    assert report.updated_version == "v2"

    assert report.baseline["tp"] == 4
    assert report.baseline["fp"] == 0
    assert report.baseline["tn"] == 49
    assert report.baseline["fn"] == 4
    assert report.baseline["recall"] == pytest.approx(0.5)
    assert report.baseline["false_alarm_rate"] == pytest.approx(0.0)
    assert report.baseline["precision"] == pytest.approx(1.0)
    assert report.baseline["f1"] == pytest.approx(0.6666666666666666)
    assert report.baseline["mean_inference_ms"] >= 0.0

    assert report.updated["tp"] == 8
    assert report.updated["fp"] == 0
    assert report.updated["tn"] == 49
    assert report.updated["fn"] == 0
    assert report.updated["recall"] == pytest.approx(1.0)
    assert report.updated["false_alarm_rate"] == pytest.approx(0.0)
    assert report.updated["precision"] == pytest.approx(1.0)
    assert report.updated["f1"] == pytest.approx(1.0)

    assert report.baseline_recall_by_family == {"speed": pytest.approx(0.5)}
    assert report.updated_recall_by_family == {"speed": pytest.approx(1.0)}

    # v1 misses half the speed attacks, so it only catches one of the two
    # interval captures (speed_attack_3 and its masquerade twin); v2 catches
    # both. Neither is a re-tune of anything -- same frozen scoring pass.
    assert report.baseline_detection_delay["captures_with_an_interval"] == 2
    assert report.baseline_detection_delay["captures_detected"] == 1
    assert report.updated_detection_delay["captures_with_an_interval"] == 2
    assert report.updated_detection_delay["captures_detected"] == 2
    assert report.updated_detection_delay["mean_delay_s"] == pytest.approx(1.0)
    assert report.updated_detection_delay["max_delay_s"] == pytest.approx(1.0)

    # Per-capture: ambient_f (cap06) has no attack windows at all, so its
    # row must show zero tp/fp/fn on both models -- this is the row that
    # answers "which capture actually produces the false alarms."
    assert set(report.updated_per_capture) == {"cap06", "cap12", "cap13"}
    assert report.updated_per_capture["cap06"]["fp"] == 0
    assert report.updated_per_capture["cap06"]["tn"] > 0
    assert report.updated_per_capture["cap12"]["tp"] + report.updated_per_capture["cap13"]["tp"] == 8


def test_detection_delay_is_none_when_final_test_has_no_interval_capture(fake_road, tmp_path):
    """A final_test made only of ambient captures has nothing to measure a
    delay against -- this must report None, never raise and never invent a
    number."""
    model_dir = train(fake_road)
    v1 = Defender.load(model_dir, "v1")
    v2 = Defender.load(model_dir, "v2")
    manifest = load_manifest(fake_road.manifest_path)
    only_ambient = [c for c in manifest.captures
                   if not (c.group == "final_test" and c.kind != "ambient")]
    path = tmp_path / "ambient_only_final_test.json"
    path.write_text(Manifest(vehicle_id=manifest.vehicle_id, captures=only_ambient)
                    .model_dump_json(indent=2), encoding="utf-8")

    report = run_final_evaluation(v1, v2, fake_road.data_dir, path)
    assert report.baseline_detection_delay is None
    assert report.updated_detection_delay is None


def test_rejects_a_manifest_with_no_final_test_captures(fake_road, tmp_path):
    model_dir = train(fake_road)
    v1 = Defender.load(model_dir, "v1")
    v2 = Defender.load(model_dir, "v2")
    no_final_path = manifest_without_final_test(fake_road, tmp_path)

    with pytest.raises(ValueError, match="no final_test captures"):
        run_final_evaluation(v1, v2, fake_road.data_dir, no_final_path)


def test_report_is_json_serializable(fake_road):
    model_dir = train(fake_road)
    v1 = Defender.load(model_dir, "v1")
    v2 = Defender.load(model_dir, "v2")
    report = run_final_evaluation(v1, v2, fake_road.data_dir, fake_road.manifest_path)
    # Round-trips through the same dump the CLI writes to disk.
    reloaded = json.loads(json.dumps(report.model_dump()))
    assert reloaded["windows"] == 57


# ---------------------------------------------------------------------
# CLI: one-time overwrite protection and version checks
# ---------------------------------------------------------------------

def cli_args(fake, model_dir, output_path, *extra):
    return ["--baseline-version", "v1", "--updated-version", "v2",
           "--data-dir", str(fake.data_dir),
           "--manifest", str(fake.manifest_path),
           "--model-dir", str(model_dir),
           "--output", str(output_path),
           *extra]


def test_cli_runs_once_and_saves_the_report(fake_road, tmp_path, capsys):
    model_dir = train(fake_road)
    out_path = tmp_path / "results" / "final_evaluation.json"

    assert main(cli_args(fake_road, model_dir, out_path)) == 0
    assert out_path.exists()

    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["windows"] == 57
    assert saved["updated"]["recall"] == pytest.approx(1.0)

    output = capsys.readouterr().out
    assert "Block 5" in output
    assert "read once" in output


def test_cli_refuses_to_overwrite_an_existing_report_without_force(fake_road, tmp_path):
    model_dir = train(fake_road)
    out_path = tmp_path / "results" / "final_evaluation.json"
    assert main(cli_args(fake_road, model_dir, out_path)) == 0

    with pytest.raises(SystemExit):
        main(cli_args(fake_road, model_dir, out_path))


def test_cli_force_allows_a_deliberate_rerun(fake_road, tmp_path):
    model_dir = train(fake_road)
    out_path = tmp_path / "results" / "final_evaluation.json"
    assert main(cli_args(fake_road, model_dir, out_path)) == 0
    assert main(cli_args(fake_road, model_dir, out_path, "--force")) == 0


def test_cli_rejects_identical_baseline_and_updated_version(fake_road, tmp_path):
    model_dir = train(fake_road)
    out_path = tmp_path / "results" / "final_evaluation.json"
    with pytest.raises(SystemExit):
        main(["--baseline-version", "v1", "--updated-version", "v1",
             "--data-dir", str(fake_road.data_dir),
             "--manifest", str(fake_road.manifest_path),
             "--model-dir", str(model_dir),
             "--output", str(out_path)])
    assert not out_path.exists()
