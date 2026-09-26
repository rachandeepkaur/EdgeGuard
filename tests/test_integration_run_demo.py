"""Tests for integration/run_demo.py (build plan S9, Block 4).

Uses the FAKE ROAD folder and manifest from conftest.py, and trains real
v1/v2 models on it with defender.run_training -- never the real dataset.

Run from EdgeGuard: python -m pytest tests/test_integration_run_demo.py -v
"""

import json

import pytest

import httpx
from fastapi.testclient import TestClient

from defender.defender import Defender
from defender.run_training import main as train_main
from integration.run_demo import (ATTACK_FAMILIES, _development_captures,
                                  calibrate_escalation_band, main, run_ordinary, run_test)
from part1.mock_cloud_endpoint import app as cloud_app


def train(fake):
    model_dir = fake.data_dir.parent / "models"
    assert train_main(["--max-false-alarm-rate", "0.0",
                       "--data-dir", str(fake.data_dir),
                       "--manifest", str(fake.manifest_path),
                       "--model-dir", str(model_dir),
                       "--max-windows-per-capture", "10"]) == 0
    return model_dir


# ---------------------------------------------------------------------
# Ordinary path
# ---------------------------------------------------------------------

def test_ordinary_scores_every_validation_window_and_forwards_all(fake_road):
    model_dir = train(fake_road)
    v1 = Defender.load(model_dir, "v1")
    summary = run_ordinary(v1, fake_road.data_dir, fake_road.manifest_path,
                           "validation", None, None, "SIMULATED_ALERT")

    assert summary.captures == ["ambient_d", "ambient_e"]
    assert summary.windows == 38
    assert summary.attack_decisions == 0
    assert summary.action_counts == {
        "SIMULATED_ALERT": 0, "SIMULATED_ISOLATION": 0, "SIMULATED_FORWARD": 38,
    }
    assert summary.latency_ms_median <= summary.latency_ms_p95 <= summary.latency_ms_max


def test_ordinary_respects_isolation_action_choice(fake_road):
    model_dir = train(fake_road)
    v1 = Defender.load(model_dir, "v1")
    # No attack decisions occur on this fake validation data either way, but
    # the choice must still reach SimulatedConsumer without erroring.
    summary = run_ordinary(v1, fake_road.data_dir, fake_road.manifest_path,
                           "validation", None, None, "SIMULATED_ISOLATION")
    assert summary.action_counts["SIMULATED_FORWARD"] == 38


def test_ordinary_respects_max_windows(fake_road):
    model_dir = train(fake_road)
    v1 = Defender.load(model_dir, "v1")
    summary = run_ordinary(v1, fake_road.data_dir, fake_road.manifest_path,
                           "validation", None, 5, "SIMULATED_ALERT")
    assert summary.windows == 5


# ---------------------------------------------------------------------
# Test path
# ---------------------------------------------------------------------

def test_development_captures_defaults_to_the_whole_group(fake_road):
    names = _development_captures(fake_road.manifest_path, None)
    assert names == ["speed_attack_1", "speed_attack_1_masquerade",
                     "speed_attack_2", "speed_attack_2_masquerade", "fuzzing_attack_1"]


def test_development_captures_rejects_non_development_names(fake_road):
    with pytest.raises(ValueError, match="Red Team attacks may only target development"):
        _development_captures(fake_road.manifest_path, ["ambient_a"])


def test_test_path_v1_misses_most_payload_attacks_v2_catches_them_all(fake_road):
    """v1 is Stage 1 (timing) only, so it should barely notice payload-only
    attacks; v2 adds Stage 2 (payload), which should catch these easily.
    This is the exact comparison the build plan's v1-vs-v2 evidence is for."""
    model_dir = train(fake_road)

    v1_summary = run_test(Defender.load(model_dir, "v1"), fake_road.data_dir,
                          fake_road.manifest_path, None, None, list(ATTACK_FAMILIES), None)
    assert v1_summary.attacks_proposed == {"freeze": 95, "offset": 95}
    assert v1_summary.attacks_skipped == {"freeze": 0, "offset": 0}
    assert v1_summary.windows_evaluated == 285          # 95 normal + 95 + 95 attacked
    assert v1_summary.recall_overall == pytest.approx(0.10526315789473684)
    assert v1_summary.recall_by_family["freeze"] == pytest.approx(0.10526315789473684)
    assert v1_summary.recall_by_family["offset"] == pytest.approx(0.10526315789473684)

    v2_summary = run_test(Defender.load(model_dir, "v2"), fake_road.data_dir,
                          fake_road.manifest_path, None, None, list(ATTACK_FAMILIES), None)
    assert v2_summary.windows_evaluated == 285
    assert v2_summary.recall_overall == pytest.approx(1.0)
    assert v2_summary.recall_by_family["freeze"] == pytest.approx(1.0)
    assert v2_summary.recall_by_family["offset"] == pytest.approx(1.0)


def test_test_path_can_run_a_single_family(fake_road):
    model_dir = train(fake_road)
    v1 = Defender.load(model_dir, "v1")
    summary = run_test(v1, fake_road.data_dir, fake_road.manifest_path, None, None,
                       ["freeze"], None)
    assert summary.families_tested == ["freeze"]
    assert set(summary.attacks_proposed) == {"freeze"}
    assert summary.windows_evaluated == 190             # 95 normal + 95 attacked


def test_test_path_respects_max_windows(fake_road):
    model_dir = train(fake_road)
    v1 = Defender.load(model_dir, "v1")
    summary = run_test(v1, fake_road.data_dir, fake_road.manifest_path, None, 3,
                       list(ATTACK_FAMILIES), None)
    # 3 base windows, each contributing 1 normal + a freeze + an offset attack
    assert summary.windows_evaluated == 9
    assert summary.attacks_proposed == {"freeze": 3, "offset": 3}


def test_test_path_writes_real_specs_to_the_evasion_log(fake_road, tmp_path):
    model_dir = train(fake_road)
    v1 = Defender.load(model_dir, "v1")
    log_path = tmp_path / "evasions.jsonl"
    summary = run_test(v1, fake_road.data_dir, fake_road.manifest_path, None, None,
                       list(ATTACK_FAMILIES), str(log_path))

    lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 170       # 190 attacks proposed, recall 0.1053 -> 20 caught, 170 missed
    assert all(entry["split"] == "development" for entry in lines)
    assert all(entry["family"] in ("freeze", "offset") for entry in lines)
    # Real attack parameters, not a placeholder -- these come straight from
    # the AttackSpec that was actually injected.
    assert all(entry["target_can_id"] == "0D0" for entry in lines)
    assert all(entry["duration_ms"] == 1000.0 for entry in lines)


def test_test_path_rejects_empty_capture_list(fake_road):
    model_dir = train(fake_road)
    v1 = Defender.load(model_dir, "v1")
    with pytest.raises(ValueError):
        run_test(v1, fake_road.data_dir, fake_road.manifest_path,
                 ["ambient_a"], None, list(ATTACK_FAMILIES), None)


# ---------------------------------------------------------------------
# Ordinary path: cloud-escalation wiring (build plan section 7)
# ---------------------------------------------------------------------

def test_calibrate_escalation_band_stays_in_the_valid_range(fake_road):
    """Regression coverage for the same off-center-threshold bug fixed in
    part1.escalation_policy: fake_road's real threshold (~0.774) is nowhere
    near 0.5, so an unclipped band would break should_escalate() outright."""
    model_dir = train(fake_road)
    v1 = Defender.load(model_dir, "v1")
    band = calibrate_escalation_band(v1, fake_road.data_dir, fake_road.manifest_path, 1.0)
    assert band == pytest.approx(0.5)


def test_run_ordinary_escalate_requires_max_escalation_rate(fake_road):
    model_dir = train(fake_road)
    v1 = Defender.load(model_dir, "v1")
    with pytest.raises(ValueError, match="max_escalation_rate"):
        run_ordinary(v1, fake_road.data_dir, fake_road.manifest_path, "validation",
                     None, None, "SIMULATED_ALERT", escalate=True)


def test_run_ordinary_without_escalate_leaves_escalation_fields_none(fake_road):
    model_dir = train(fake_road)
    v1 = Defender.load(model_dir, "v1")
    summary = run_ordinary(v1, fake_road.data_dir, fake_road.manifest_path,
                           "validation", None, None, "SIMULATED_ALERT")
    assert summary.escalation_band_half_width is None
    assert summary.egress is None


def test_run_ordinary_escalate_leaves_local_decisions_unchanged_when_cloud_unreachable(fake_road):
    """The single most important property of this feature: a failed or
    unreachable second opinion must never affect the local decision or
    action, only be recorded as a failed call."""
    model_dir = train(fake_road)
    v1 = Defender.load(model_dir, "v1")
    baseline = run_ordinary(v1, fake_road.data_dir, fake_road.manifest_path,
                            "validation", None, None, "SIMULATED_ALERT")
    escalated = run_ordinary(v1, fake_road.data_dir, fake_road.manifest_path,
                             "validation", None, None, "SIMULATED_ALERT",
                             escalate=True, max_escalation_rate=1.0,
                             cloud_url="http://127.0.0.1:9")

    assert escalated.action_counts == baseline.action_counts
    assert escalated.attack_decisions == baseline.attack_decisions
    assert escalated.escalation_band_half_width == pytest.approx(0.5)
    assert escalated.egress["windows"] == 38
    assert escalated.egress["escalations"] == 22
    assert escalated.egress["failed_calls"] == 22          # nothing is listening on :9
    assert escalated.egress["corrections"] == 0
    assert escalated.egress["correction_rate"] is None      # no answered calls at all


def test_run_ordinary_escalate_gets_a_real_second_opinion_end_to_end(fake_road, monkeypatch):
    """Route the escalation client at a real FastAPI TestClient (the actual
    route logic in part1.mock_cloud_endpoint, not a stub) instead of a real
    socket, so this exercises the genuine confirm/override decision."""
    model_dir = train(fake_road)
    v1 = Defender.load(model_dir, "v1")
    client = TestClient(cloud_app)

    def fake_post(url, json=None, timeout=None):
        return client.post("/advise", json=json)

    monkeypatch.setattr(httpx, "post", fake_post)
    summary = run_ordinary(v1, fake_road.data_dir, fake_road.manifest_path,
                           "validation", None, None, "SIMULATED_ALERT",
                           escalate=True, max_escalation_rate=1.0,
                           cloud_url="http://fake-cloud")

    assert summary.egress["escalations"] == 22
    assert summary.egress["failed_calls"] == 0              # every call actually answered
    assert summary.egress["correction_rate"] is not None
    assert 0 <= summary.egress["corrections"] <= 22


# ---------------------------------------------------------------------
# CLI: one command, both paths
# ---------------------------------------------------------------------

def cli_args(fake, model_dir, *extra):
    return ["--model-version", "v1",
           "--data-dir", str(fake.data_dir),
           "--manifest", str(fake.manifest_path),
           "--model-dir", str(model_dir),
           *extra]


def test_cli_default_mode_runs_both_paths_and_saves_both_reports(fake_road, tmp_path, capsys):
    model_dir = train(fake_road)
    out_dir = tmp_path / "results"
    assert main(cli_args(fake_road, model_dir, "--output-dir", str(out_dir))) == 0

    ordinary = json.loads((out_dir / "demo_ordinary.json").read_text(encoding="utf-8"))
    test = json.loads((out_dir / "demo_test.json").read_text(encoding="utf-8"))
    assert ordinary["windows"] == 38
    assert test["windows_evaluated"] == 285

    output = capsys.readouterr().out
    assert "Ordinary path" in output
    assert "Test path" in output


def test_cli_mode_ordinary_only_writes_only_that_report(fake_road, tmp_path):
    model_dir = train(fake_road)
    out_dir = tmp_path / "results"
    assert main(cli_args(fake_road, model_dir, "--mode", "ordinary",
                         "--output-dir", str(out_dir))) == 0
    assert (out_dir / "demo_ordinary.json").exists()
    assert not (out_dir / "demo_test.json").exists()


def test_cli_mode_test_only_writes_only_that_report(fake_road, tmp_path):
    model_dir = train(fake_road)
    out_dir = tmp_path / "results"
    assert main(cli_args(fake_road, model_dir, "--mode", "test",
                         "--output-dir", str(out_dir))) == 0
    assert (out_dir / "demo_test.json").exists()
    assert not (out_dir / "demo_ordinary.json").exists()


def test_cli_rejects_eval_only_ordinary_group(fake_road, tmp_path):
    model_dir = train(fake_road)
    with pytest.raises(SystemExit):
        main(cli_args(fake_road, model_dir, "--ordinary-group", "final_test",
                      "--output-dir", str(tmp_path / "results")))


def test_cli_rejects_unknown_test_family(fake_road, tmp_path):
    model_dir = train(fake_road)
    with pytest.raises(SystemExit):
        main(cli_args(fake_road, model_dir, "--test-families", "made_up",
                      "--output-dir", str(tmp_path / "results")))


def test_cli_rejects_non_positive_max_windows(fake_road, tmp_path):
    model_dir = train(fake_road)
    with pytest.raises(SystemExit):
        main(cli_args(fake_road, model_dir, "--ordinary-max-windows", "0",
                      "--output-dir", str(tmp_path / "results")))


def test_cli_single_test_family_via_flag(fake_road, tmp_path):
    model_dir = train(fake_road)
    out_dir = tmp_path / "results"
    assert main(cli_args(fake_road, model_dir, "--mode", "test",
                         "--test-families", "offset",
                         "--output-dir", str(out_dir))) == 0
    test = json.loads((out_dir / "demo_test.json").read_text(encoding="utf-8"))
    assert test["families_tested"] == ["offset"]


# ---------------------------------------------------------------------
# CLI: --escalate
# ---------------------------------------------------------------------

def test_cli_escalate_requires_max_escalation_rate(fake_road, tmp_path):
    model_dir = train(fake_road)
    with pytest.raises(SystemExit):
        main(cli_args(fake_road, model_dir, "--escalate",
                      "--output-dir", str(tmp_path / "results")))


def test_cli_escalate_with_test_only_mode_measures_quality_without_a_cloud(
        fake_road, tmp_path, capsys):
    """--escalate on the test path needs no cloud: it scores the band against
    the known Red Team labels (part1.escalation_quality)."""
    model_dir = train(fake_road)
    out_dir = tmp_path / "results"
    assert main(cli_args(fake_road, model_dir, "--mode", "test", "--escalate",
                         "--max-escalation-rate", "1.0",
                         "--output-dir", str(out_dir))) == 0
    assert not (out_dir / "demo_ordinary.json").exists()
    test = json.loads((out_dir / "demo_test.json").read_text(encoding="utf-8"))
    q = test["escalation_quality"]
    assert q["windows"] == test["windows_evaluated"]
    assert q["band_half_width"] == pytest.approx(0.5)
    assert "Escalation quality" in capsys.readouterr().out


def test_test_path_without_band_leaves_escalation_quality_none(fake_road):
    model_dir = train(fake_road)
    v2 = Defender.load(model_dir, "v2")
    summary = run_test(v2, fake_road.data_dir, fake_road.manifest_path,
                       None, 5, list(ATTACK_FAMILIES), None)
    assert summary.escalation_quality is None


def test_test_path_escalation_quality_is_consistent_with_detection_metrics(fake_road):
    """Same join as part3.evaluator, so the counts must agree exactly."""
    model_dir = train(fake_road)
    v1 = Defender.load(model_dir, "v1")
    band = calibrate_escalation_band(v1, fake_road.data_dir, fake_road.manifest_path, 0.5)
    summary = run_test(v1, fake_road.data_dir, fake_road.manifest_path,
                       None, None, list(ATTACK_FAMILIES), None,
                       escalation_band_half_width=band)
    q = summary.escalation_quality
    assert q["windows"] == summary.windows_evaluated
    assert q["band_half_width"] == band
    assert 0 <= q["escalated_errors"] <= q["local_errors"]
    assert q["escalated_errors"] <= q["escalations"] <= q["windows"]
    assert q["local_accuracy"] <= q["local_plus_perfect_cloud_accuracy"] <= 1.0
    # v1 misses most payload attacks on fake data, so it has real mistakes to find
    assert q["local_false_negatives"] > 0


def test_cli_both_modes_reuse_one_band_for_ordinary_and_test(fake_road, tmp_path):
    model_dir = train(fake_road)
    out_dir = tmp_path / "results"
    assert main(cli_args(fake_road, model_dir, "--escalate",
                         "--max-escalation-rate", "0.5",
                         "--cloud-url", "http://127.0.0.1:9",
                         "--output-dir", str(out_dir))) == 0
    ordinary = json.loads((out_dir / "demo_ordinary.json").read_text(encoding="utf-8"))
    test = json.loads((out_dir / "demo_test.json").read_text(encoding="utf-8"))
    assert (test["escalation_quality"]["band_half_width"]
            == ordinary["escalation_band_half_width"])


def test_cli_escalate_writes_and_prints_the_egress_report(fake_road, tmp_path, capsys):
    model_dir = train(fake_road)
    out_dir = tmp_path / "results"
    assert main(cli_args(fake_road, model_dir, "--mode", "ordinary", "--escalate",
                         "--max-escalation-rate", "1.0",
                         "--cloud-url", "http://127.0.0.1:9",   # nothing listening there
                         "--output-dir", str(out_dir))) == 0

    ordinary = json.loads((out_dir / "demo_ordinary.json").read_text(encoding="utf-8"))
    assert ordinary["escalation_band_half_width"] == pytest.approx(0.5)
    assert ordinary["egress"]["escalations"] == 22
    assert ordinary["egress"]["failed_calls"] == 22

    output = capsys.readouterr().out
    assert "Cloud escalation path" in output
    assert "Escalated:      22 / 38" in output
