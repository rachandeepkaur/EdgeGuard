"""Tests for rule-based Red Team proposals using made-up traffic.

Run from EdgeGuard: python -m pytest tests/test_red_team_agent.py -v
"""

import pytest

from part3.attack_injector import inject_freeze, inject_offset
from part3.red_team_agent import propose, propose_freeze, propose_offset
from shared.schemas import TrafficWindow


def sample_window():
    return TrafficWindow(
        window_id="cap07_w0001",
        capture_id="cap07",
        window_start=100.0,
        window_end=101.0,
        frames=[
            {"timestamp": 100.1, "can_id": "0F4", "payload": "01"},
            {"timestamp": 100.2, "can_id": "1A0", "payload": "AA"},
            {"timestamp": 100.3, "can_id": "0f4", "payload": "02"},
            {"timestamp": 100.4, "can_id": "0F4", "payload": "03"},
            {"timestamp": 100.5, "can_id": "1A0", "payload": "AA"},
        ],
    )


# ---------------------------------------------------------------------
# freeze
# ---------------------------------------------------------------------

def test_proposes_attack_that_injector_can_apply_without_changing_source():
    source = sample_window()
    proposal = propose_freeze(source, attack_id="atk_demo")
    assert proposal.attack_id == "atk_demo"
    assert proposal.family == "freeze"
    assert proposal.target_can_id == "0F4"
    assert proposal.duration_ms == pytest.approx(1000.0)

    attacked = inject_freeze(source, proposal)
    assert [frame.payload for frame in source.frames] == ["01", "AA", "02", "03", "AA"]
    assert [frame.payload for frame in attacked.frames] == ["01", "AA", "01", "01", "AA"]


def test_rejects_constant_payloads():
    source = sample_window()
    for frame in source.frames:
        if frame.can_id.upper() == "0F4":
            frame.payload = "01"
    with pytest.raises(ValueError, match="no CAN ID has changing payloads"):
        propose_freeze(source)


def test_rejects_preprocessed_window():
    source = sample_window()
    source.features = {"frame_count": 5.0}
    with pytest.raises(ValueError, match="raw window"):
        propose_freeze(source)


def test_rejects_zero_duration_window():
    source = TrafficWindow(
        window_id="cap07_w0002",
        capture_id="cap07",
        window_start=100.0,
        window_end=100.0,
        frames=[],
    )
    with pytest.raises(ValueError, match="positive duration"):
        propose_freeze(source)


# ---------------------------------------------------------------------
# offset
# ---------------------------------------------------------------------

def test_offset_proposes_the_most_stable_byte_pushed_toward_more_headroom():
    # 1A0 is perfectly stable ("AA" = 170 both times) and outnumbers 0F4's
    # varying payloads; 170 > 127 so it should be pushed DOWN, toward 0x00.
    source = sample_window()
    proposal = propose_offset(source, attack_id="atk_demo")
    assert proposal.attack_id == "atk_demo"
    assert proposal.family == "offset"
    assert proposal.target_can_id == "1A0"
    assert proposal.byte_index == 0
    assert proposal.offset == -120
    assert proposal.duration_ms == pytest.approx(1000.0)

    attacked = inject_offset(source, proposal)
    assert [frame.payload for frame in source.frames] == ["01", "AA", "02", "03", "AA"]
    # 0xAA (170) - 120 = 50 = 0x32
    assert [frame.payload for frame in attacked.frames] == ["01", "32", "02", "03", "32"]


def test_offset_pushes_low_values_upward():
    source = TrafficWindow(
        window_id="cap07_w0003", capture_id="cap07",
        window_start=0.0, window_end=1.0,
        frames=[
            {"timestamp": 0.1, "can_id": "0D0", "payload": "05"},
            {"timestamp": 0.2, "can_id": "0D0", "payload": "05"},
        ],
    )
    proposal = propose_offset(source)
    assert proposal.target_can_id == "0D0" and proposal.byte_index == 0
    assert proposal.offset == 120          # mean (5) <= 127 -> push up


def test_offset_rejects_preprocessed_window():
    source = sample_window()
    source.features = {"frame_count": 5.0}
    with pytest.raises(ValueError, match="raw window"):
        propose_offset(source)


def test_offset_rejects_zero_duration_window():
    source = TrafficWindow(
        window_id="cap07_w0004", capture_id="cap07",
        window_start=100.0, window_end=100.0, frames=[],
    )
    with pytest.raises(ValueError, match="positive duration"):
        propose_offset(source)


def test_offset_rejects_window_with_no_payload_bytes():
    source = TrafficWindow(
        window_id="cap07_w0005", capture_id="cap07",
        window_start=100.0, window_end=101.0,
        frames=[{"timestamp": 100.1, "can_id": "0F4", "payload": ""}],
    )
    with pytest.raises(ValueError, match="no payload bytes"):
        propose_offset(source)


# ---------------------------------------------------------------------
# propose() dispatcher
# ---------------------------------------------------------------------

def test_propose_dispatches_freeze():
    assert propose(sample_window(), "freeze").family == "freeze"


def test_propose_dispatches_offset():
    assert propose(sample_window(), "offset").family == "offset"


def test_propose_rejects_unknown_family():
    with pytest.raises(ValueError, match="unsupported attack family"):
        propose(sample_window(), "made_up")
