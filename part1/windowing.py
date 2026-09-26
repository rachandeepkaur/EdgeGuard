"""
Part 1 -- windowing. THE one windowing function of EdgeGuard.

Groups a capture's frame stream into TrafficWindow objects. The Defender
(Part 2), the injector (Part 3), training, cross-validation and the dev check
all get their windows from here; there is no second implementation.

LOCKED TEAM DECISIONS (build plan S3: "fixed, non-overlapping 1-second window")
    window size   1.0 s
    stride        1.0 s   (stride == window -> no overlap)
    window_id     "{capture_id}_w{index:04d}", index from 0 within a capture
    empty windows EMITTED  (a quiet second is real traffic; dropping it would
                  silently shrink the normal-traffic denominator)
    final window  DROPPED  (it only covers part of a second, so its timing
                  statistics are not comparable to a full window)
    time          ELAPSED seconds from the capture's first frame, computed in
                  integer microseconds (the same clock as ROAD's
                  injection_interval)

Frame assignment is half-open, in integer microseconds: a frame belongs to
window k when
    k * window_us  <=  t_us  <  (k + 1) * window_us
so a frame landing exactly on a boundary is counted once, in the later window.

keep_every=N returns only every Nth window (skipped windows are never built).
It exists to limit TRAINING memory (~1 MB per real window). Evaluation must
use keep_every=1: the build plan requires every evaluation window be scored.

Output windows have frames filled and features / decoded_signals EMPTY.
part1.pipeline.preprocess() fills those afterwards -- after any red-team
injection -- so normal and attacked traffic go through identical steps.
"""

from __future__ import annotations

import math
from typing import Iterable, Iterator, List, Optional, Tuple

from shared.schemas import Frame, TrafficWindow

from part1.cleaning import CleaningReport, clean
from part1.fleet_simulator import RawFrame, iter_raw

WINDOW_S = 1.0     # locked
STRIDE_S = 1.0     # locked -- equal to WINDOW_S, so windows never overlap

# (window index, start_us, end_us, frames of this window)
RawWindow = Tuple[int, int, int, List[RawFrame]]


def window_id_for(capture_id: str, index: int) -> str:
    """cap07, 153 -> cap07_w0153. Deterministic, so Part 3 can look it up."""
    return f"{capture_id}_w{index:04d}"


def window_us_for(window_s: float, stride_s: float) -> int:
    """Validate the locked decisions and return the window length in us."""
    if window_s <= 0:
        raise ValueError(f"window_s must be positive, got {window_s}")
    if not math.isclose(stride_s, window_s):
        raise NotImplementedError(
            f"stride_s={stride_s} != window_s={window_s}. Overlapping windows "
            f"are not supported: the build plan fixes non-overlapping windows so "
            f"each frame lands in exactly one window. Change the decision before the code."
        )
    window_us = round(window_s * 1_000_000)
    if not math.isclose(window_us, window_s * 1_000_000, abs_tol=1e-6):
        raise ValueError(f"window_s={window_s} is not a whole number of microseconds")
    return window_us


def raw_windows(frames: Iterable[RawFrame], window_us: int, keep_every: int = 1,
                capture: str = "capture") -> Iterator[RawWindow]:
    """The windowing core: group (elapsed_us, can_id, payload) frames into
    windows. Every other function here is a thin wrapper around this one."""
    if not isinstance(keep_every, int) or isinstance(keep_every, bool) or keep_every < 1:
        raise ValueError(f"keep_every must be a whole number >= 1, got {keep_every}")
    current = 0
    bucket: List[RawFrame] = []
    last_us = None
    for frame in frames:
        us = frame[0]
        if last_us is not None and us < last_us:
            raise ValueError(
                f"{capture}: frames out of order ({us} us after {last_us} us); "
                f"windowing assumes a time-ordered stream")
        if us < 0:
            raise ValueError(f"{capture}: negative elapsed time {us} us")
        last_us = us
        index = us // window_us
        # Close every window that ends before this frame, including empty ones
        # in a gap, before placing the frame.
        while current < index:
            if current % keep_every == 0:
                yield current, current * window_us, (current + 1) * window_us, bucket
            bucket = []
            current += 1
        if current % keep_every == 0:
            bucket.append(frame)
    # The window holding the final frame is only partly covered by the
    # recording, so it is dropped on purpose.


def to_traffic_window(raw: RawWindow, capture_id: str,
                      vehicle_id: Optional[str] = None) -> TrafficWindow:
    index, start_us, end_us, frames = raw
    return TrafficWindow(
        window_id=window_id_for(capture_id, index),
        capture_id=capture_id,
        vehicle_id=vehicle_id,
        window_start=start_us / 1_000_000,
        window_end=end_us / 1_000_000,
        frames=[Frame(timestamp=us / 1_000_000, can_id=can_id, payload=payload)
                for us, can_id, payload in frames],
    )


def windows_from_frames(
    frames: Iterable[Frame],
    capture_id: str,
    window_s: float = WINDOW_S,
    stride_s: float = STRIDE_S,
    vehicle_id: Optional[str] = None,
    keep_every: int = 1,
) -> Iterator[TrafficWindow]:
    """Group an elapsed-time Frame stream into TrafficWindows.

    For made-up frames in tests and for Part 3, which re-windows an injected
    frame list without touching files. Frames must arrive in timestamp order,
    starting near 0.0.
    """
    window_us = window_us_for(window_s, stride_s)
    raw = ((round(f.timestamp * 1_000_000), f.can_id, f.payload) for f in frames)
    for window in raw_windows(raw, window_us, keep_every, capture_id):
        yield to_traffic_window(window, capture_id, vehicle_id)


def make_windows(
    capture_path: str,
    capture_id: str,
    window_s: float = WINDOW_S,
    stride_s: float = STRIDE_S,
    vehicle_id: Optional[str] = None,
    keep_every: int = 1,
    report: Optional[CleaningReport] = None,
) -> Iterator[TrafficWindow]:
    """Agreed entry point for Parts 2 and 3.

    Streams one ROAD capture file, CLEANS it (part1.cleaning), and yields
    TrafficWindow objects with frames filled and features / decoded_signals
    empty.

        for window in make_windows(path, "cap07", vehicle_id="veh01"):
            ...

    capture_id must be the neutral id from the split manifest, never the
    original file name. Pass a CleaningReport to see what cleaning did.
    """
    window_us = window_us_for(window_s, stride_s)
    report = report if report is not None else CleaningReport()
    frames = clean(iter_raw(capture_path, report), report, capture_id)
    for window in raw_windows(frames, window_us, keep_every, capture_id):
        yield to_traffic_window(window, capture_id, vehicle_id)


def keep_every_for(span_us: int, window_s: float, max_windows: int) -> int:
    """How often to keep a window so a capture of span_us yields at most
    max_windows (for training memory only)."""
    if max_windows < 1:
        raise ValueError(f"max_windows must be at least 1, got {max_windows}")
    window_us = round(window_s * 1_000_000)
    estimated = max(0, span_us // window_us)      # full windows only
    return max(1, math.ceil(estimated / max_windows))


if __name__ == "__main__":
    import os
    import sys
    import time

    from part1.fleet_simulator import discover_captures

    road = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Downloads/road")
    caps = discover_captures(road)
    if not caps:
        sys.exit(f"no captures found under {road}")

    cap = caps[0]
    print(f"windowing {cap.capture_id} ...")
    t0 = time.perf_counter()
    count = frames = empty = 0
    first = None
    rep = CleaningReport()
    for w in make_windows(cap.path, cap.capture_id, report=rep):
        if first is None:
            first = w
        count += 1
        frames += len(w.frames)
        empty += not w.frames
    elapsed = time.perf_counter() - t0

    print(f"  windows        {count}")
    print(f"  frames         {frames}")
    print(f"  empty windows  {empty}")
    print(f"  avg frames/win {frames / count:.0f}" if count else "")
    print(f"  cleaning       {rep.to_dict()}")
    print(f"  took           {elapsed:.1f}s")
    if first:
        print(f"\nfirst window: {first.window_id}  "
              f"[{first.window_start}, {first.window_end})  "
              f"{len(first.frames)} frames")
