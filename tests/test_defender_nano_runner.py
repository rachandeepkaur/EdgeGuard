"""
Tests for defender/nano_runner.py.

Run from the EdgeGuard folder with:
    python -m pytest tests/test_defender_nano_runner.py -v

The --mock tests use MOCK traffic only. The real-mode tests use the FAKE
ROAD folder and manifest from conftest.py (never the real dataset).
"""

import json

import pytest

from defender.defender import Defender
from defender.nano_runner import build_mock_defender, main, mock_windows, real_windows, \
    run_benchmark
from defender.run_training import main as train_main


@pytest.fixture(scope="module")
def mock_defender():
    return build_mock_defender()


def test_one_record_per_window(mock_defender):
    records, summary = run_benchmark(mock_defender, mock_windows(20), mock_traffic=True)
    assert len(records) == 20 and summary.windows == 20


def test_latency_values_are_valid(mock_defender):
    records, summary = run_benchmark(mock_defender, mock_windows(20), mock_traffic=True)
    assert all(r.latency_ms >= 0 for r in records)
    assert summary.latency_ms_median <= summary.latency_ms_p95 <= summary.latency_ms_max
    assert summary.windows_per_second > 0


def test_mock_attacks_are_counted(mock_defender):
    _, summary = run_benchmark(mock_defender, mock_windows(20), mock_traffic=True)
    assert summary.attack_decisions == 2          # windows 9 and 19 are MOCK fuzzing


def test_mock_results_are_labelled(mock_defender):
    _, summary = run_benchmark(mock_defender, mock_windows(5), mock_traffic=True)
    assert summary.mock_traffic is True
    assert "MOCK" in summary.note and "NOT a detection result" in summary.note


def test_single_window_works(mock_defender):
    _, summary = run_benchmark(mock_defender, mock_windows(1), mock_traffic=True)
    assert summary.latency_ms_p95 == summary.latency_ms_max


def test_runs_with_network_blocked(mock_defender, monkeypatch):
    """Primary detection must not need any network connection."""
    import socket

    def no_network(*args, **kwargs):
        raise AssertionError("Defender tried to open a network connection")
    monkeypatch.setattr(socket, "socket", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    records, _ = run_benchmark(mock_defender, mock_windows(10), mock_traffic=True)
    assert len(records) == 10


def test_rejects_empty_window_list(mock_defender):
    with pytest.raises(ValueError, match="No windows"):
        run_benchmark(mock_defender, [], mock_traffic=True)


def test_rejects_an_object_with_no_score_window():
    with pytest.raises(TypeError, match="needs an object with score_window"):
        run_benchmark("not a defender", mock_windows(1), mock_traffic=True)


class DuckTypedDefender:
    """No relation to defender.defender.Defender at all -- proves
    run_benchmark() only needs score_window()/model_version, matching
    TimingCNNDefender / FusedV2TimingCNNDefender / DefenderLLMDefender."""

    model_version = "duck_v1"

    def score_window(self, window):
        from shared.schemas import DefenderOutput
        return DefenderOutput(window_id=window.window_id, attack_score=0.1,
                              threshold=0.5, decision="ACCEPT", evidence="duck",
                              model_version=self.model_version, latency_ms=1.0)


def test_accepts_any_duck_typed_defender_not_just_defender():
    records, summary = run_benchmark(DuckTypedDefender(), mock_windows(5), mock_traffic=True)
    assert summary.windows == 5
    assert summary.model_version == "duck_v1"


def test_command_line_mock_run_saves_results(tmp_path, capsys):
    out = tmp_path / "bench.json"
    assert main(["--mock", "--windows", "15", "--output", str(out)]) == 0
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["summary"]["windows"] == 15
    assert saved["summary"]["mock_traffic"] is True
    assert len(saved["windows"]) == 15
    assert "MOCK traffic" in capsys.readouterr().out


def test_command_line_requires_a_mode(tmp_path):
    """Neither --mock nor --model-version given: refuse, don't guess."""
    with pytest.raises(SystemExit):
        main(["--output", str(tmp_path / "x.json")])


def test_command_line_rejects_mock_and_model_version_together(tmp_path):
    with pytest.raises(SystemExit):
        main(["--mock", "--model-version", "v1", "--output", str(tmp_path / "x.json")])


def test_command_line_rejects_eval_only_groups(tmp_path):
    """--group only accepts train/validation/development; argparse enforces this."""
    with pytest.raises(SystemExit):
        main(["--model-version", "v1", "--group", "final_test", "--output",
             str(tmp_path / "x.json")])


# ---------------------------------------------------------------------
# Real mode, on the FAKE ROAD folder (never the real dataset)
# ---------------------------------------------------------------------

def train(fake):
    """Train v1 + v2 on the fake road so real mode has something to load."""
    model_dir = fake.data_dir.parent / "models"
    assert train_main(["--max-false-alarm-rate", "0.0",
                       "--data-dir", str(fake.data_dir),
                       "--manifest", str(fake.manifest_path),
                       "--model-dir", str(model_dir),
                       "--max-windows-per-capture", "10"]) == 0
    return model_dir


def test_real_windows_reads_the_requested_group(fake_road):
    windows, names = real_windows(fake_road.data_dir, fake_road.manifest_path, "train")
    assert names == ["ambient_a", "ambient_b", "ambient_c"]
    assert len(windows) > 0


def test_real_windows_accepts_explicit_captures(fake_road):
    windows, names = real_windows(fake_road.data_dir, fake_road.manifest_path, "validation",
                                  captures=["ambient_d"])
    assert names == ["ambient_d"]
    assert len(windows) > 0


def test_real_windows_respects_max_windows(fake_road):
    windows, _ = real_windows(fake_road.data_dir, fake_road.manifest_path, "train",
                              max_windows=3)
    assert len(windows) == 3


def test_real_windows_refuses_final_test_and_separate(fake_road):
    """RoadData itself refuses these; real_windows never passes final_evaluation=True."""
    with pytest.raises(ValueError, match="final evaluation"):
        real_windows(fake_road.data_dir, fake_road.manifest_path, "train",
                     captures=["speed_attack_3"])


def test_real_windows_rejects_unknown_group(fake_road):
    with pytest.raises(ValueError):
        real_windows(fake_road.data_dir, fake_road.manifest_path, "not_a_group")


def test_command_line_real_run_scores_real_windows(fake_road, tmp_path, capsys):
    model_dir = train(fake_road)
    out = tmp_path / "bench_real.json"
    assert main(["--model-version", "v1",
                "--data-dir", str(fake_road.data_dir),
                "--manifest", str(fake_road.manifest_path),
                "--model-dir", str(model_dir),
                "--group", "train",
                "--output", str(out)]) == 0
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["summary"]["mock_traffic"] is False
    assert saved["summary"]["model_version"] == "v1"
    assert saved["summary"]["windows"] > 0
    assert len(saved["windows"]) == saved["summary"]["windows"]
    out_text = capsys.readouterr().out
    assert "REAL ROAD traffic" in out_text
    assert "ambient_a" in out_text


def test_command_line_real_run_respects_max_windows(fake_road, tmp_path):
    model_dir = train(fake_road)
    out = tmp_path / "bench_capped.json"
    assert main(["--model-version", "v1",
                "--data-dir", str(fake_road.data_dir),
                "--manifest", str(fake_road.manifest_path),
                "--model-dir", str(model_dir),
                "--group", "train",
                "--max-windows", "2",
                "--output", str(out)]) == 0
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["summary"]["windows"] == 2


def test_command_line_real_run_uses_captures_over_group(fake_road, tmp_path):
    model_dir = train(fake_road)
    out = tmp_path / "bench_captures.json"
    assert main(["--model-version", "v1",
                "--data-dir", str(fake_road.data_dir),
                "--manifest", str(fake_road.manifest_path),
                "--model-dir", str(model_dir),
                "--captures", "ambient_d,ambient_e",
                "--output", str(out)]) == 0
    out_text_path = out
    saved = json.loads(out_text_path.read_text(encoding="utf-8"))
    assert saved["summary"]["windows"] > 0


def test_command_line_real_run_missing_model_gives_clear_error(fake_road, tmp_path):
    out = tmp_path / "bench_missing.json"
    with pytest.raises(FileNotFoundError):
        main(["--model-version", "v9-does-not-exist",
             "--data-dir", str(fake_road.data_dir),
             "--manifest", str(fake_road.manifest_path),
             "--model-dir", str(tmp_path / "no_models_here"),
             "--output", str(out)])


def test_command_line_real_run_benchmarks_the_fused_detector(fake_road, tmp_path, monkeypatch):
    """--cnn-model-dir benchmarks FusedV2TimingCNNDefender end to end. Needs
    torch (TimingCNNModel/FusedV2TimingCNNDefender import it) -- skipped
    wherever torch isn't installed, same as tests/test_timing_cnn.py."""
    pytest.importorskip("torch")
    import defender.nano_runner as nano_runner_mod
    from defender.fused_defender import FusedV2TimingCNNDefender
    from defender.timing_cnn import TimingCNNModel

    model_dir = train(fake_road)

    class FakeCnn:
        fitted = True
        threshold = 0.5

        def predict_proba_attack(self, rows):
            return [0.0 for _ in rows]

    monkeypatch.setattr(TimingCNNModel, "load", classmethod(lambda cls, path: FakeCnn()))

    out = tmp_path / "bench_fused.json"
    rc = nano_runner_mod.main(["--cnn-model-dir", "irrelevant/fake/path",
                              "--data-dir", str(fake_road.data_dir),
                              "--manifest", str(fake_road.manifest_path),
                              "--model-dir", str(model_dir),
                              "--group", "train",
                              "--output", str(out)])
    assert rc == 0
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["summary"]["model_version"] == "fused_v2_timing_cnn_v1"
    assert saved["summary"]["windows"] > 0
