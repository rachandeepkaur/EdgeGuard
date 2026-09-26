"""
Tests for part1/fleet_simulator.py.

Run from the EdgeGuard folder:
    python -m pytest tests/test_part1_fleet_simulator.py -v

Log lines here copy the real ROAD format. Files are written to a temporary
folder; the real ROAD data is never touched.
"""

import pytest

from part1.fleet_simulator import (
    canonical_can_id,
    discover_captures,
    read_capture_stats,
    read_frames,
)
from shared.schemas import TrafficWindow


def write_log(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


REAL_LINES = [
    "(1110000000.000000) can0 0F4#960C010204B10240",
    "(1110000000.000001) can0 32D#0000042758010000",
    "(1110000000.001020) can0 69E#0440047E1FC01542",
]


# ---------- parsing ---------------------------------------------------
def test_parses_real_road_line_format(tmp_path):
    log = write_log(tmp_path / "a.log", REAL_LINES)
    frames = list(read_frames(str(log)))
    assert len(frames) == 3
    assert frames[0].can_id == "0F4"
    assert frames[0].payload == "960C010204B10240"


def test_timestamps_are_elapsed_from_zero(tmp_path):
    log = write_log(tmp_path / "a.log", REAL_LINES)
    stamps = [f.timestamp for f in read_frames(str(log))]
    assert stamps[0] == 0.0
    assert stamps[2] == pytest.approx(0.00102)
    assert all(t < 1000 for t in stamps), "raw epoch timestamps must not leak"


def test_elapsed_time_is_exact_not_float_blurred(tmp_path):
    """Regression: float subtraction at ~1.1e9 gave 0.09999990463256836.

    That is enough to put a frame in the wrong one-second window, so elapsed
    time is computed in integer nanoseconds.
    """
    log = write_log(tmp_path / "a.log", [
        "(1110000000.000000) can0 0F4#00",
        "(1110000000.100000) can0 0F4#00",
        "(1110000001.000000) can0 0F4#00",
    ])
    stamps = [f.timestamp for f in read_frames(str(log))]
    assert stamps == [0.0, 0.1, 1.0]


def test_frame_on_exact_boundary_stays_on_boundary(tmp_path):
    """A frame recorded at exactly +1 s must not come out as 0.9999999."""
    log = write_log(tmp_path / "a.log", [
        "(1110000000.000000) can0 0F4#00",
        "(1110000001.000000) can0 0F4#00",
    ])
    assert [f.timestamp for f in read_frames(str(log))][1] >= 1.0


def test_skips_blank_and_malformed_lines(tmp_path):
    log = write_log(tmp_path / "a.log", [
        REAL_LINES[0], "", "not a can line", "# comment", REAL_LINES[1],
    ])
    assert len(list(read_frames(str(log)))) == 2


def test_lowercase_ids_and_payloads_are_uppercased(tmp_path):
    log = write_log(tmp_path / "a.log", ["(1.0) can0 0f4#960c"])
    frame = next(read_frames(str(log)))
    assert (frame.can_id, frame.payload) == ("0F4", "960C")


def test_empty_payload_is_parsed(tmp_path):
    log = write_log(tmp_path / "a.log", ["(1.0) can0 0F4#"])
    assert next(read_frames(str(log))).payload == ""


def test_stats_report_skipped_lines(tmp_path):
    log = write_log(tmp_path / "a.log", REAL_LINES + ["garbage", "more garbage"])
    stats = read_capture_stats(str(log))
    assert stats["frames"] == 3
    assert stats["skipped_lines"] == 2
    assert stats["unique_ids"] == 3


# ---------- CAN id formats --------------------------------------------
@pytest.mark.parametrize("raw, expected", [
    ("0F4", "0F4"),        # log file form, already canonical
    ("0x6e0", "6E0"),      # capture_metadata.json form
    ("0xd0", "0D0"),       # metadata, only 2 digits -> must be padded
    ("0x4e7", "4E7"),
    (244, "0F4"),          # signal_extractions CSV decimal form
    ("XXX", "XXX"),        # fuzzing: every id injected
])
def test_canonical_can_id(raw, expected):
    assert canonical_can_id(raw) == expected


def test_metadata_id_matches_log_id():
    """The trap: metadata '0xd0' must equal the log file's '0D0'."""
    assert canonical_can_id("0xd0") == canonical_can_id("0D0") == "0D0"


# ---------- capture ids -----------------------------------------------
def make_road_tree(root):
    write_log(root / "ambient" / "ambient_dyno_drive_basic_long.log", REAL_LINES)
    write_log(root / "ambient" / "ambient_dyno_drive_winter.log", REAL_LINES)
    write_log(root / "attacks" / "max_speedometer_attack_1.log", REAL_LINES)
    write_log(root / "attacks" / "correlated_signal_attack_1.log", REAL_LINES)
    (root / "attacks" / "capture_metadata.json").write_text("{}")


def test_capture_ids_are_sequential_and_ambient_first(tmp_path):
    make_road_tree(tmp_path)
    caps = discover_captures(str(tmp_path))
    assert [c.capture_id for c in caps] == ["cap01", "cap02", "cap03", "cap04"]
    assert caps[0].original_name.startswith("ambient")
    assert caps[1].original_name.startswith("ambient")


def test_capture_mapping_is_deterministic(tmp_path):
    """Every teammate must get the same cap07 -> file mapping."""
    make_road_tree(tmp_path)
    first = [(c.capture_id, c.original_name) for c in discover_captures(str(tmp_path))]
    again = [(c.capture_id, c.original_name) for c in discover_captures(str(tmp_path))]
    assert first == again


def test_metadata_json_is_not_treated_as_a_capture(tmp_path):
    make_road_tree(tmp_path)
    names = [c.original_name for c in discover_captures(str(tmp_path))]
    assert not any("metadata" in n for n in names)


def test_capture_ids_pass_the_schema_neutrality_check(tmp_path):
    make_road_tree(tmp_path)
    for cap in discover_captures(str(tmp_path)):
        TrafficWindow(window_id=f"{cap.capture_id}_w0000", capture_id=cap.capture_id,
                      window_start=0.0, window_end=1.0, frames=[])


def test_missing_road_folder_returns_nothing(tmp_path):
    assert discover_captures(str(tmp_path / "nowhere")) == []
