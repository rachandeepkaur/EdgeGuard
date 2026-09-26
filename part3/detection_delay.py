"""
Part 3: detection delay -- build plan item "detection delay per attack
(time to first alert after the interval starts)" (CLAUDE.md's "What to do
next" #7; also listed in the plan's own Metrics section).

Only meaningful for a capture with a real, continuous injection interval
(fabrication/masquerade real ROAD attacks: part1.labels.InjectionRule).
Fuzzing has no single interval (injected frames are scattered across the
whole capture; see part1/labels.py's docstring) and is excluded on purpose,
not by accident -- there is no well-defined "attack onset" to measure from.

Delay is measured to the END of the first ATTACK-decision window that
overlaps or follows the interval's start, not its start: the Defender can
only act once a window closes (build plan: "window collection time and
inference time reported separately"; inference is ~1-2 ms and already
reported elsewhere, so window-close time dominates this number by
construction). A capture where no window in or after the interval was
flagged ATTACK is a full miss for that capture -- its delay is None, and is
excluded from the mean/percentiles, not counted as zero.
"""

import statistics
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional

from shared.schemas import DefenderOutput, GroundTruthLabel, TrafficWindow


class Scored(NamedTuple):
    # `window` only needs to expose capture_id/window_start/window_end --
    # never .frames. A real TrafficWindow works, and so does any lighter
    # duck-typed stand-in (see integration/run_final_evaluation.py's
    # _WindowTiming, which avoids keeping ~2,400 raw frames per window
    # alive for the whole final_test run).
    window: TrafficWindow
    output: DefenderOutput
    label: GroundTruthLabel


@dataclass(frozen=True)
class CaptureDelay:
    capture_id: str
    family: Optional[str]
    interval_start_s: float
    detected: bool
    delay_s: Optional[float]   # None if detected is False


@dataclass(frozen=True)
class DetectionDelayReport:
    per_capture: List[CaptureDelay]
    captures_with_an_interval: int
    captures_detected: int
    mean_delay_s: Optional[float]
    median_delay_s: Optional[float]
    p95_delay_s: Optional[float]
    max_delay_s: Optional[float]


def _p95(sorted_values: List[float]) -> float:
    index = max(0, min(len(sorted_values) - 1, round(0.95 * (len(sorted_values) - 1))))
    return sorted_values[index]


def compute_detection_delay(scored: List[Scored]) -> DetectionDelayReport:
    """scored must hold every window of each capture being measured, in
    any order (this function sorts by window_start itself) -- a partial
    capture would make "first ATTACK window after onset" meaningless.
    Windows are grouped by GroundTruthLabel.injection_interval; a capture
    with no interval on any of its windows (ambient, fuzzing, a clean
    negative control) is skipped, not scored as zero delay.
    """
    if not scored:
        raise ValueError("scored is empty: need at least one capture's windows")

    by_capture: Dict[str, List[Scored]] = {}
    for item in scored:
        by_capture.setdefault(item.window.capture_id, []).append(item)

    per_capture: List[CaptureDelay] = []
    for capture_id, items in sorted(by_capture.items()):
        interval_starts = {
            item.label.injection_interval[0]
            for item in items
            if item.label.injection_interval is not None
        }
        if not interval_starts:
            continue   # no real injection interval on this capture -- not measurable
        if len(interval_starts) != 1:
            raise ValueError(
                f"{capture_id}: windows disagree on injection_interval start "
                f"({sorted(interval_starts)}); one capture must have one interval"
            )
        interval_start = interval_starts.pop()
        family = next((item.label.family for item in items if item.label.family), None)

        ordered = sorted(items, key=lambda item: item.window.window_start)
        first_alert = next(
            (item for item in ordered
             if item.window.window_end >= interval_start and item.output.decision == "ATTACK"),
            None,
        )
        if first_alert is None:
            per_capture.append(CaptureDelay(capture_id, family, interval_start, False, None))
        else:
            delay = first_alert.window.window_end - interval_start
            per_capture.append(CaptureDelay(capture_id, family, interval_start, True, delay))

    if not per_capture:
        raise ValueError(
            "no capture in `scored` has a real injection_interval; detection delay "
            "needs fabrication/masquerade captures, not fuzzing or ambient ones"
        )

    delays = [c.delay_s for c in per_capture if c.detected]
    sorted_delays = sorted(delays)
    return DetectionDelayReport(
        per_capture=per_capture,
        captures_with_an_interval=len(per_capture),
        captures_detected=len(delays),
        mean_delay_s=statistics.fmean(delays) if delays else None,
        median_delay_s=statistics.median(delays) if delays else None,
        p95_delay_s=_p95(sorted_delays) if sorted_delays else None,
        max_delay_s=max(delays) if delays else None,
    )
