"""
Tests for defender/simulated_consumer.py.

Run from the EdgeGuard folder with:
    python -m pytest tests/test_defender_consumer.py -v

All data here is MOCK data. The "ROAD file" is a fake temporary file.
"""

import hashlib

import pytest

from defender.simulated_consumer import (
    SIMULATED_ALERT,
    SIMULATED_FORWARD,
    SIMULATED_ISOLATION,
    SimulatedConsumer,
)
from shared.schemas import DefenderOutput, TrafficWindow


def mock_output(score, decision, window_id="cap07_w0153"):
    return DefenderOutput(window_id=window_id, attack_score=score, threshold=0.70,
                          decision=decision, evidence="MOCK evidence",
                          model_version="v1")


def mock_window():
    return TrafficWindow(
        window_id="cap07_w0153", capture_id="cap07",
        window_start=1532.0, window_end=1533.0,
        frames=[{"timestamp": 1532.001, "can_id": "0F4", "payload": "960C010204B10240"}],
    )


# ---------- Correct simulated alert / isolation -----------------------
def test_attack_gives_simulated_alert():
    record = SimulatedConsumer().handle(mock_output(0.83, "ATTACK"))
    assert record.action == SIMULATED_ALERT
    assert record.window_id == "cap07_w0153"
    assert record.attack_score == 0.83
    assert record.decision == "ATTACK"


def test_attack_can_give_simulated_isolation():
    consumer = SimulatedConsumer(attack_action=SIMULATED_ISOLATION)
    assert consumer.handle(mock_output(0.83, "ATTACK")).action == SIMULATED_ISOLATION


# ---------- Correct simulated forwarding ------------------------------
def test_accept_gives_simulated_forward():
    record = SimulatedConsumer().handle(mock_output(0.20, "ACCEPT"))
    assert record.action == SIMULATED_FORWARD
    assert record.decision == "ACCEPT"


# ---------- Clearly marked as simulated -------------------------------
def test_every_record_is_marked_simulated():
    consumer = SimulatedConsumer()
    for record in [consumer.handle(mock_output(0.83, "ATTACK")),
                   consumer.handle(mock_output(0.20, "ACCEPT"))]:
        assert record.simulated is True
        assert record.action.startswith("SIMULATED_")
        assert "[SIMULATION ONLY]" in record.message


def test_record_is_read_only():
    record = SimulatedConsumer().handle(mock_output(0.83, "ATTACK"))
    with pytest.raises(Exception):
        record.action = SIMULATED_FORWARD


def test_summary_counts_actions():
    consumer = SimulatedConsumer()
    consumer.handle(mock_output(0.83, "ATTACK", "cap07_w0001"))
    consumer.handle(mock_output(0.20, "ACCEPT", "cap07_w0002"))
    consumer.handle(mock_output(0.10, "ACCEPT", "cap07_w0003"))
    assert consumer.summary() == {SIMULATED_ALERT: 1, SIMULATED_ISOLATION: 0,
                                  SIMULATED_FORWARD: 2}


# ---------- Invalid inputs --------------------------------------------
def test_rejects_non_attack_action_setting():
    with pytest.raises(ValueError):
        SimulatedConsumer(attack_action=SIMULATED_FORWARD)


def test_rejects_plain_dictionary():
    with pytest.raises(TypeError, match="DefenderOutput"):
        SimulatedConsumer().handle({"decision": "ATTACK"})


def test_rejects_raw_traffic_window():
    """The consumer takes Defender results, never CAN traffic."""
    with pytest.raises(TypeError):
        SimulatedConsumer().handle(mock_window())


# ---------- Nothing is modified ---------------------------------------
def test_no_traffic_or_road_file_is_modified(tmp_path):
    # MOCK "ROAD" log file in a temporary folder (never the real dataset).
    fake_log = tmp_path / "cap07.log"
    fake_log.write_text("(1080000000.000000) can0 00E#2052D60208097550\n", encoding="utf-8")
    file_hash_before = hashlib.sha256(fake_log.read_bytes()).hexdigest()

    window = mock_window()
    window_before = window.model_dump()
    output = mock_output(0.83, "ATTACK")
    output_before = output.model_dump()

    consumer = SimulatedConsumer()
    consumer.handle(output)
    consumer.handle(mock_output(0.20, "ACCEPT"))

    assert hashlib.sha256(fake_log.read_bytes()).hexdigest() == file_hash_before
    assert window.model_dump() == window_before
    assert output.model_dump() == output_before