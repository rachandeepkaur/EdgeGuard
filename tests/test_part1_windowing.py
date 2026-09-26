"""
Tests for part1/windowing.py.

Run from the EdgeGuard folder:
    python -m pytest tests/test_part1_windowing.py -v

Covers the LOCKED team decisions: 1.0 s windows, 1.0 s stride, no overlap,
empty windows kept, final partial window dropped.
"""

import pytest

from part1.windowing import (
    STRIDE_S,
    WINDOW_S,
    make_windows,
    window_id_for,
    windows_from_frames,
)
from shared.schemas import Frame, TrafficWindow


def frame(t, can_id="0F4"):
    return Frame(timestamp=t, can_id=can_id, payload="00")


# ---------- locked decisions ------------------------------------------
def test_locked_window_and_stride():
    assert WINDOW_S == 1.0
    assert STRIDE_S == 1.0


def test_window_id_format():
    assert window_id_for("cap07", 153) == "cap07_w0153"
    assert window_id_for("cap07", 0) == "cap07_w0000"


# ---------- grouping --------------------------------------------------
def test_frames_grouped_into_one_second_windows():
    wins = list(windows_from_frames([frame(0.1), frame(0.9), frame(1.2), frame(2.5)],
                                    "cap07"))
    assert [len(w.frames) for w in wins] == [2, 1]
    assert (wins[0].window_start, wins[0].window_end) == (0.0, 1.0)
    assert (wins[1].window_start, wins[1].window_end) == (1.0, 2.0)


def test_boundary_frame_goes_to_the_later_window():
    """Half-open [start, end): a frame at exactly 1.0 is counted once."""
    wins = list(windows_from_frames([frame(0.5), frame(1.0), frame(2.5)], "cap07"))
    assert [f.timestamp for f in wins[0].frames] == [0.5]
    assert [f.timestamp for f in wins[1].frames] == [1.0]


def test_no_frame_appears_in_two_windows():
    stream = [frame(i * 0.1) for i in range(55)]
    seen = [f.timestamp for w in windows_from_frames(stream, "cap07") for f in w.frames]
    assert len(seen) == len(set(seen))


def test_gap_produces_empty_windows():
    wins = list(windows_from_frames([frame(0.5), frame(3.5), frame(4.5)], "cap07"))
    assert [w.window_id for w in wins] == [
        "cap07_w0000", "cap07_w0001", "cap07_w0002", "cap07_w0003"]
    assert wins[1].frames == [] and wins[2].frames == []


def test_final_partial_window_is_dropped():
    wins = list(windows_from_frames([frame(0.5), frame(1.5), frame(2.3)], "cap07"))
    assert all(f.timestamp != 2.3 for w in wins for f in w.frames)
    assert wins[-1].window_id == "cap07_w0001"


def test_single_window_capture_yields_nothing():
    """Everything is in the final partial window, which is dropped."""
    assert list(windows_from_frames([frame(0.1), frame(0.5)], "cap07")) == []


def test_empty_stream_yields_nothing():
    assert list(windows_from_frames([], "cap07")) == []


# ---------- output contract -------------------------------------------
def test_windows_are_valid_traffic_windows():
    for w in windows_from_frames([frame(0.1), frame(1.1), frame(2.1)], "cap07"):
        assert isinstance(w, TrafficWindow)


def test_features_and_signals_start_empty():
    """preprocess() fills these later, AFTER any injection."""
    for w in windows_from_frames([frame(0.1), frame(1.1), frame(2.1)], "cap07"):
        assert w.features == {} and w.decoded_signals == {}


def test_window_ids_carry_the_capture_id():
    for w in windows_from_frames([frame(0.1), frame(1.1), frame(2.1)], "cap12"):
        assert w.window_id.startswith("cap12_w")
        assert w.capture_id == "cap12"


# ---------- guards ----------------------------------------------------
def test_rejects_overlapping_stride():
    with pytest.raises(NotImplementedError, match="Overlapping"):
        list(windows_from_frames([frame(0.1)], "cap07", window_s=1.0, stride_s=0.5))


def test_rejects_non_positive_window():
    with pytest.raises(ValueError):
        list(windows_from_frames([frame(0.1)], "cap07", window_s=0, stride_s=0))


def test_rejects_out_of_order_frames():
    with pytest.raises(ValueError, match="out of order"):
        list(windows_from_frames([frame(1.5), frame(0.5), frame(2.5)], "cap07"))


# ---------- from a real file ------------------------------------------
def test_make_windows_reads_a_road_log(tmp_path):
    log = tmp_path / "cap.log"
    # 30 frames, 0.1 s apart -> spans 0.0 .. 2.9 s
    log.write_text("\n".join(
        f"({1110000000 + i // 10}.{(i % 10) * 100000:06d}) can0 0F4#960C010204B10240"
        for i in range(30)
    ) + "\n")
    wins = list(make_windows(str(log), "cap01"))
    assert [w.window_id for w in wins] == ["cap01_w0000", "cap01_w0001"]
    assert wins[0].window_start == 0.0


# ---------- sampling, vehicle id, exact boundaries ---------------------
def test_keep_every_returns_every_nth_window_with_its_own_index():
    frames = [frame(i + 0.5) for i in range(10)]
    kept = list(windows_from_frames(frames, "cap07", keep_every=3))
    assert [w.window_id for w in kept] == ["cap07_w0000", "cap07_w0003", "cap07_w0006"]
    assert all(len(w.frames) == 1 for w in kept)


def test_keep_every_must_be_a_positive_whole_number():
    with pytest.raises(ValueError, match="keep_every"):
        list(windows_from_frames([frame(0.1)], "cap07", keep_every=0))


def test_vehicle_id_is_carried_on_every_window():
    wins = list(windows_from_frames([frame(0.1), frame(1.1), frame(2.1)], "cap07",
                                    vehicle_id="veh01"))
    assert {w.vehicle_id for w in wins} == {"veh01"}


def test_boundary_is_exact_in_microseconds():
    """0.999999 s belongs to window 0 and 1.000000 s to window 1, exactly."""
    wins = list(windows_from_frames([frame(0.999999), frame(1.0), frame(2.5)], "cap07"))
    assert [f.timestamp for f in wins[0].frames] == [0.999999]
    assert [f.timestamp for f in wins[1].frames] == [1.0]


def test_make_windows_cleans_duplicates(tmp_path):
    from part1.cleaning import CleaningReport
    line = "(1110000000.100000) can0 0F4#960C010204B10240"
    log = tmp_path / "cap.log"
    rows = [line, line] + [f"(1110000000.{100000 + i * 400:06d}) can0 0D0#00" for i in range(1, 2200)] \
        + ["(1110000002.000000) can0 0D0#00"]
    log.write_text("\n".join(rows) + "\n")
    report = CleaningReport()
    wins = list(make_windows(str(log), "cap01", report=report))
    assert report.duplicates_dropped == 1
    assert sum(1 for f in wins[0].frames if f.can_id == "0F4") == 1


def test_keep_every_for_limits_windows():
    from part1.windowing import keep_every_for
    assert keep_every_for(20_000_000, 1.0, 10) == 2      # 20 windows, keep 10
    assert keep_every_for(5_000_000, 1.0, 10) == 1
