"""
Part 1 -- cleaning (capture artifacts only).

GOVERNING RULE: cleaning may remove only things provable from the capture
MECHANISM, never unusual bus CONTENT. On an intrusion dataset the unusual
content (out-of-range values, frozen signals, unknown IDs) IS the attack, so
removing it would delete the answer.

What this does
    duplicates  exact (timestamp, CAN ID, payload) repeats are DROPPED. Two
                identical frames at the same microsecond are physically
                impossible on a 500 kbit/s bus; they are the logger reading
                its buffer twice (~0.005 % of ROAD frames). Kept, they would
                put a 0-second gap into Stage 1's short_gaps check.
    filler      0xFFF is a logger filler frame (~136 Hz, one payload). It is
                COUNTED, not removed: it is real bus traffic that the
                Defender must also see at runtime.
    malformed   lines that are not CAN frames are COUNTED (see
                fleet_simulator.iter_raw), never silently lost.
    order       frames must arrive in time order; anything else stops the
                read with an error, because windowing depends on it.

GATE: if duplicates + malformed lines exceed MAX_REMOVED_FRACTION (0.1 %)
of a capture, CleaningGateError is raised. Removing that much means the step
is removing content, not artifacts.

Measured on ROAD by the preprocessing study in ~/Downloads/road (docs/04).
"""

from dataclasses import asdict, dataclass
from typing import Iterable, Iterator, Optional

from part1.fleet_simulator import RawFrame

FILLER_IDS = frozenset({"FFF"})
MAX_REMOVED_FRACTION = 0.001


class CleaningGateError(ValueError):
    """Cleaning would remove more than MAX_REMOVED_FRACTION of a capture."""


@dataclass
class CleaningReport:
    frames_read: int = 0            # parsed frames, before cleaning
    frames_kept: int = 0
    duplicates_dropped: int = 0
    skipped_lines: int = 0          # non-empty lines that are not CAN frames
    filler_frames: int = 0          # kept, counted
    first_raw_timestamp: Optional[str] = None   # e.g. "1110000000.000000"

    @property
    def removed_fraction(self) -> float:
        total = self.frames_read + self.skipped_lines
        removed = self.duplicates_dropped + self.skipped_lines
        return removed / total if total else 0.0

    def to_dict(self) -> dict:
        return {**asdict(self), "removed_fraction": round(self.removed_fraction, 8)}


def clean(frames: Iterable[RawFrame], report: Optional[CleaningReport] = None,
          capture: str = "capture") -> Iterator[RawFrame]:
    """Yield frames with exact duplicates removed. Fills `report` as it goes
    and checks the gate once the stream is exhausted."""
    report = report if report is not None else CleaningReport()
    current_us = None
    seen_at_current = set()          # (can_id, payload) already seen at current_us
    for frame in frames:
        us, can_id, payload = frame
        report.frames_read += 1
        if current_us is not None and us < current_us:
            raise ValueError(
                f"{capture}: frames out of order ({us} us after {current_us} us); "
                f"windowing assumes a time-ordered stream")
        if us != current_us:
            current_us = us
            seen_at_current = set()
        key = (can_id, payload)
        if key in seen_at_current:
            report.duplicates_dropped += 1
            continue
        seen_at_current.add(key)
        if can_id in FILLER_IDS:
            report.filler_frames += 1
        report.frames_kept += 1
        yield frame
    if report.removed_fraction > MAX_REMOVED_FRACTION:
        raise CleaningGateError(
            f"{capture}: cleaning removed {report.removed_fraction:.4%} of the capture "
            f"({report.duplicates_dropped} duplicates, {report.skipped_lines} malformed "
            f"lines), above the {MAX_REMOVED_FRACTION:.1%} limit. That is content, "
            f"not capture artifacts: check the file.")
