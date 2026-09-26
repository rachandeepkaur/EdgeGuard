"""Tests for Part 3 attack validation with mock traffic.

Run from EdgeGuard: python -m pytest tests/test_attack_validator.py -v
"""

from part3.attack_injector import inject_freeze, inject_offset
from part3.attack_spec import AttackSpec
from part3.attack_validator import validate_attack
from shared.schemas import TrafficWindow


def window():
    return TrafficWindow(
        window_id="cap07_w0001", capture_id="cap07",
        window_start=100.0, window_end=101.0,
        frames=[
            {"timestamp": 100.1, "can_id": "0F4", "payload": "01"},
            {"timestamp": 100.2, "can_id": "1A0", "payload": "AA"},
            {"timestamp": 100.3, "can_id": "0f4", "payload": "02"},
            {"timestamp": 100.4, "can_id": "0F4", "payload": "03"},
        ],
    )


def spec(**changes):
    fields = dict(attack_id="atk_001", family="freeze", target_can_id="0F4",
                  start_offset_ms=0, duration_ms=500)
    fields.update(changes)
    return AttackSpec(**fields)


def offset_spec(**changes):
    fields = dict(attack_id="atk_002", family="offset", target_can_id="0F4",
                  start_offset_ms=0, duration_ms=500, byte_index=0, offset=100)
    fields.update(changes)
    return AttackSpec(**fields)


# ---------------------------------------------------------------------
# freeze
# ---------------------------------------------------------------------

def test_valid_proposal_matches_injector_effect_without_modifying_source():
    original = window()
    report = validate_attack(original, spec(), allowed_can_ids=["0f4"])
    modified = inject_freeze(original, spec())
    assert report.valid and report.matching_frames == 3
    assert report.changes_expected == 2
    assert sum(a.payload != b.payload for a, b in zip(original.frames, modified.frames)) == 2
    assert original.frames[2].payload == "02"


def test_rejects_target_not_on_allowed_list():
    report = validate_attack(window(), spec(), allowed_can_ids=["1A0"])
    assert not report.valid and "allowed list" in report.reason


def test_rejects_target_without_matching_frames():
    report = validate_attack(window(), spec(target_can_id="ABC"))
    assert not report.valid and "no frames" in report.reason


def test_rejects_interval_outside_window():
    report = validate_attack(window(), spec(start_offset_ms=1000, duration_ms=100))
    assert not report.valid and "does not overlap" in report.reason


def test_rejects_constant_payloads_without_changing_them():
    original = window()
    original.frames[2].payload = "01"
    original.frames[3].payload = "01"
    report = validate_attack(original, spec())
    assert not report.valid and report.changes_expected == 0
    assert [frame.payload for frame in original.frames] == ["01", "AA", "01", "01"]


def test_rejects_processed_input():
    original = window()
    original.features = {"frame_count": 4.0}
    report = validate_attack(original, spec())
    assert not report.valid and "raw window" in report.reason


# ---------------------------------------------------------------------
# offset
# ---------------------------------------------------------------------

def test_offset_valid_proposal_matches_injector_effect_without_modifying_source():
    original = window()
    report = validate_attack(original, offset_spec(), allowed_can_ids=["0f4"])
    modified = inject_offset(original, offset_spec())
    assert report.valid and report.matching_frames == 3
    assert report.changes_expected == 3
    assert sum(a.payload != b.payload for a, b in zip(original.frames, modified.frames)) == 3
    assert original.frames[0].payload == "01"


def test_offset_rejects_target_not_on_allowed_list():
    report = validate_attack(window(), offset_spec(), allowed_can_ids=["1A0"])
    assert not report.valid and "allowed list" in report.reason


def test_offset_rejects_target_without_matching_frames():
    report = validate_attack(window(), offset_spec(target_can_id="ABC"))
    assert not report.valid and "no frames" in report.reason


def test_offset_rejects_interval_outside_window():
    report = validate_attack(window(), offset_spec(start_offset_ms=1000, duration_ms=100))
    assert not report.valid and "does not overlap" in report.reason


def test_offset_rejects_when_already_saturated_without_changing_payloads():
    original = window()
    original.frames[0].payload = "FF"
    original.frames[2].payload = "FF"
    original.frames[3].payload = "FF"
    report = validate_attack(original, offset_spec(offset=10))
    assert not report.valid and report.changes_expected == 0
    assert [frame.payload for frame in original.frames] == ["FF", "AA", "FF", "FF"]


def test_offset_rejects_processed_input():
    original = window()
    original.features = {"frame_count": 4.0}
    report = validate_attack(original, offset_spec())
    assert not report.valid and "raw window" in report.reason
