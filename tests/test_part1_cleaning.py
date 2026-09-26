"""
Tests for part1/cleaning.py and fleet_simulator.iter_raw / capture_span_us.

Run from the EdgeGuard folder with:
    python -m pytest tests/test_part1_cleaning.py -v
"""

import pytest

from part1.cleaning import CleaningGateError, CleaningReport, clean
from part1.fleet_simulator import capture_span_us, iter_raw


def test_exact_duplicates_are_dropped_and_counted():
    frames = [(0, "0D0", "AA"), (0, "0D0", "AA"), (0, "0F4", "AA"), (10, "0D0", "AA")] + \
             [(100 + i, "0F4", "00") for i in range(5000)]
    report = CleaningReport()
    kept = list(clean(frames, report))
    assert kept[:3] == [(0, "0D0", "AA"), (0, "0F4", "AA"), (10, "0D0", "AA")]
    assert report.duplicates_dropped == 1 and report.frames_kept == len(frames) - 1


def test_same_frame_at_another_time_is_not_a_duplicate():
    kept = list(clean([(0, "0D0", "AA"), (1, "0D0", "AA")]))
    assert len(kept) == 2


def test_filler_frames_are_counted_not_removed():
    report = CleaningReport()
    kept = list(clean([(0, "FFF", "00"), (5, "FFF", "00"), (9, "0D0", "11")], report))
    assert len(kept) == 3 and report.filler_frames == 2


def test_unusual_content_is_never_removed():
    """Out-of-range values and unknown IDs are what attacks look like."""
    frames = [(0, "0D0", "FFFFFFFFFFFFFFFF"), (1, "7FF", "FFFFFFFFFFFFFFFF")]
    assert list(clean(frames)) == frames


def test_gate_stops_a_capture_that_loses_too_much():
    frames = [(0, "0D0", "AA")] * 10          # 9 of 10 are duplicates
    with pytest.raises(CleaningGateError, match="above the 0.1% limit"):
        list(clean(frames))


def test_out_of_order_frames_raise():
    with pytest.raises(ValueError, match="out of order"):
        list(clean([(10, "0D0", "AA"), (5, "0D0", "AA")]))


def test_iter_raw_counts_malformed_lines_and_uses_integer_microseconds(tmp_path):
    log = tmp_path / "x.log"
    log.write_text("(1110000000.000000) can0 0D0#AA\n"
                   "garbage line\n"
                   "\n"
                   "(1110000000.000001) can0 0f4#bb\n", encoding="utf-8")
    report = CleaningReport()
    frames = list(iter_raw(str(log), report))
    assert frames == [(0, "0D0", "AA"), (1, "0F4", "BB")]      # 1 us exactly, upper-cased
    assert report.skipped_lines == 1                           # blank lines are not counted
    assert report.first_raw_timestamp == "1110000000.000000"


def test_capture_span_reads_first_and_last_frame(tmp_path):
    log = tmp_path / "x.log"
    lines = [f"(1110000000.{i:06d}) can0 0D0#AA" for i in range(0, 900000, 300)]
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert capture_span_us(str(log)) == 899700
