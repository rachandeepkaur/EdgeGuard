"""Tests for part3/attack_injector.py; all frames here are mock data.

Run from EdgeGuard: python -m pytest tests/test_attack_injector.py -v
"""

import pytest

from part3.attack_injector import inject, inject_freeze, inject_offset
from part3.attack_spec import AttackSpec
from shared.schemas import TrafficWindow


def sample_window():
    return TrafficWindow(
        window_id="cap07_w0153",
        capture_id="cap07",
        window_start=100.0,
        window_end=101.0,
        frames=[
            {"timestamp": 100.100, "can_id": "0F4", "payload": "01"},
            {"timestamp": 100.150, "can_id": "1A0", "payload": "99"},
            {"timestamp": 100.200, "can_id": "0f4", "payload": "02"},
            {"timestamp": 100.300, "can_id": "0F4", "payload": "03"},
            {"timestamp": 100.800, "can_id": "0F4", "payload": "04"},
        ],
    )


def freeze_spec(**changes):
    values = dict(
        attack_id="atk_001",
        family="freeze",
        target_can_id="0F4",
        start_offset_ms=0,
        duration_ms=500,
    )
    values.update(changes)
    return AttackSpec(**values)


def offset_spec(**changes):
    values = dict(
        attack_id="atk_002",
        family="offset",
        target_can_id="0F4",
        start_offset_ms=0,
        duration_ms=500,
        byte_index=0,
        offset=100,
    )
    values.update(changes)
    return AttackSpec(**values)


# ---------------------------------------------------------------------
# freeze
# ---------------------------------------------------------------------

def test_freeze_changes_only_matching_frames_and_preserves_source():
    source = sample_window()
    attacked = inject_freeze(source, freeze_spec())

    assert attacked.window_id == "cap07_w0153_v01"
    assert attacked.capture_id == source.capture_id
    assert [f.payload for f in attacked.frames] == ["01", "99", "01", "01", "04"]
    assert [f.payload for f in source.frames] == ["01", "99", "02", "03", "04"]
    assert [f.timestamp for f in attacked.frames] == [f.timestamp for f in source.frames]
    assert attacked.features == {} and attacked.decoded_signals == {}


def test_second_variant_gets_a_distinct_neutral_id():
    attacked = inject_freeze(sample_window(), freeze_spec(), variant_index=2)
    assert attacked.window_id == "cap07_w0153_v02"


def test_interval_uses_window_start_in_milliseconds():
    attacked = inject_freeze(
        sample_window(), freeze_spec(start_offset_ms=150, duration_ms=200)
    )
    # The frame at 100.100 is outside; 100.200 and 100.300 are inside.
    assert [f.payload for f in attacked.frames] == ["01", "99", "02", "02", "04"]


@pytest.mark.parametrize(
    "changes",
    [
        {"target_can_id": "ABC"},
        {"start_offset_ms": 900, "duration_ms": 100},
        {"start_offset_ms": 0, "duration_ms": 150},
    ],
)
def test_rejects_attack_that_changes_no_frames(changes):
    with pytest.raises(ValueError):
        inject_freeze(sample_window(), freeze_spec(**changes))


def test_rejects_preprocessed_input():
    source = sample_window()
    source.features = {"frame_count": 5.0}
    with pytest.raises(ValueError, match="raw window"):
        inject_freeze(source, freeze_spec())


def test_rejects_invalid_variant_index():
    with pytest.raises(ValueError, match="positive integer"):
        inject_freeze(sample_window(), freeze_spec(), variant_index=0)


# ---------------------------------------------------------------------
# offset
# ---------------------------------------------------------------------

def test_offset_adds_delta_and_preserves_source():
    source = sample_window()
    # Default interval [100.0, 100.5) covers the 0F4 frames at .100/.200/.300
    # but not the one at .800. "01"/"02"/"03" (hex) + 100 -> 0x65/0x66/0x67.
    attacked = inject_offset(source, offset_spec(offset=100))

    assert attacked.window_id == "cap07_w0153_v01"
    assert [f.payload for f in attacked.frames] == ["65", "99", "66", "67", "04"]
    assert [f.payload for f in source.frames] == ["01", "99", "02", "03", "04"]
    assert [f.timestamp for f in attacked.frames] == [f.timestamp for f in source.frames]


def test_offset_clamps_at_255_instead_of_wrapping():
    source = TrafficWindow(
        window_id="cap07_w0199", capture_id="cap07",
        window_start=0.0, window_end=1.0,
        frames=[{"timestamp": 0.1, "can_id": "0F4", "payload": "FE"}],
    )
    # 0xFE (254) + the max allowed offset (127) would be 381; clamp to 255.
    attacked = inject_offset(source, offset_spec(offset=127, duration_ms=1000))
    assert attacked.frames[0].payload == "FF"


def test_offset_clamps_at_zero():
    source = TrafficWindow(
        window_id="cap07_w0200", capture_id="cap07",
        window_start=0.0, window_end=1.0,
        frames=[{"timestamp": 0.1, "can_id": "0F4", "payload": "05"}],
    )
    attacked = inject_offset(source, offset_spec(offset=-128, duration_ms=1000))
    assert attacked.frames[0].payload == "00"


def test_offset_leaves_frames_too_short_for_byte_index_unchanged():
    source = TrafficWindow(
        window_id="cap07_w0201", capture_id="cap07",
        window_start=0.0, window_end=1.0,
        frames=[
            {"timestamp": 0.1, "can_id": "0F4", "payload": "01"},         # 1 byte
            {"timestamp": 0.2, "can_id": "0F4", "payload": "0102030405"},  # 5 bytes
        ],
    )
    attacked = inject_offset(source, offset_spec(byte_index=4, offset=50, duration_ms=1000))
    assert attacked.frames[0].payload == "01"                # too short, untouched
    assert attacked.frames[1].payload == "0102030437"         # byte 4: 0x05+50=0x37


def test_second_offset_variant_gets_a_distinct_neutral_id():
    attacked = inject_offset(sample_window(), offset_spec(), variant_index=3)
    assert attacked.window_id == "cap07_w0153_v03"


@pytest.mark.parametrize(
    "changes",
    [
        {"target_can_id": "ABC"},
        {"start_offset_ms": 900, "duration_ms": 100},
    ],
)
def test_offset_rejects_attack_with_no_matching_frames(changes):
    with pytest.raises(ValueError):
        inject_offset(sample_window(), offset_spec(**changes))


def test_offset_rejects_when_value_is_already_saturated():
    source = TrafficWindow(
        window_id="cap07_w0202", capture_id="cap07",
        window_start=0.0, window_end=1.0,
        frames=[{"timestamp": 0.1, "can_id": "0F4", "payload": "FF"}],
    )
    with pytest.raises(ValueError, match="changed no frames"):
        inject_offset(source, offset_spec(offset=10, duration_ms=1000))


def test_offset_rejects_preprocessed_input():
    source = sample_window()
    source.features = {"frame_count": 5.0}
    with pytest.raises(ValueError, match="raw window"):
        inject_offset(source, offset_spec())


def test_offset_rejects_invalid_variant_index():
    with pytest.raises(ValueError, match="positive integer"):
        inject_offset(sample_window(), offset_spec(), variant_index=0)


def test_offset_rejects_freeze_spec():
    with pytest.raises(ValueError, match="unsupported attack family"):
        inject_offset(sample_window(), freeze_spec())


def test_freeze_rejects_offset_spec():
    with pytest.raises(ValueError, match="unsupported attack family"):
        inject_freeze(sample_window(), offset_spec())


# ---------------------------------------------------------------------
# inject() dispatcher
# ---------------------------------------------------------------------

def test_inject_dispatches_freeze():
    assert inject(sample_window(), freeze_spec()).frames[2].payload == "01"


def test_inject_dispatches_offset():
    assert inject(sample_window(), offset_spec(offset=100)).frames[2].payload == "66"
