"""
Tests for defender/dev_check.py.

Run from the EdgeGuard folder with:
    python -m pytest tests/test_defender_dev_check.py -v
"""

from defender.dev_check import check_capture, freeze_id
from defender.run_training import main as train
from defender.defender import Defender
from part1.pipeline import RoadData
from shared.schemas import TrafficWindow


def test_freeze_id_makes_a_changed_copy():
    w = TrafficWindow(window_id="cap40_w0000", capture_id="cap40",
                      window_start=0.0, window_end=1.0,
                      frames=[{"timestamp": 0.1, "can_id": "0D0", "payload": "42710460F4000000"},
                              {"timestamp": 0.2, "can_id": "0F4", "payload": "0011"}])
    frozen = freeze_id(w, "0D0", "3A710460F5000000")
    assert frozen.frames[0].payload == "3A710460F5000000"
    assert frozen.frames[1].payload == "0011"
    assert w.frames[0].payload == "42710460F4000000"          # original unchanged
    assert frozen.window_id == "cap40_w0000_v01"


def trained_defenders(fake):
    models = fake.data_dir.parent / "models"
    train(["--max-false-alarm-rate", "0.0", "--data-dir", str(fake.data_dir),
           "--manifest", str(fake.manifest_path), "--model-dir", str(models),
           "--max-windows-per-capture", "10"])
    return {v: Defender.load(models, v) for v in ("v1", "v2")}


def test_check_capture_uses_part1_labels(fake_road):
    defenders = trained_defenders(fake_road)
    road = RoadData(fake_road.data_dir, manifest=fake_road.manifest)
    fab = check_capture(road, "speed_attack_1", defenders)
    assert fab["v1"]["attacked"] == 4 and fab["v1"]["normal"] == 15   # windows 8..11 of 19
    # byte 5 = FF is outside the normal 00 range, so v2's Stage 2 catches it
    assert fab["v2"]["detected"] == 4


def test_masquerade_contributes_only_attacked_windows(fake_road):
    defenders = trained_defenders(fake_road)
    road = RoadData(fake_road.data_dir, manifest=fake_road.manifest)
    masq = check_capture(road, "speed_attack_1_masquerade", defenders)
    assert masq["v1"]["attacked"] == 4 and masq["v1"]["normal"] == 0
