"""
Stage 1 timing detector for the EdgeGuard Defender (Part 2).

Stage 1 looks only at WHEN frames arrive and WHICH CAN IDs appear.
It never reads payload meaning, decoded signals or labels.

It learns "normal" from NORMAL TRAINING windows only, then checks a new
window for three timing signs:

    unknown_ids   frames from CAN IDs never seen in normal training
                  (typical of fuzzing)
    rate_excess   an ID sends more frames per second than it normally does
                  (typical of fabrication with message confliction, flooding)
    short_gaps    two frames of the same ID arrive much closer together
                  than normal (typical of injection at a higher rate)

HOW RAW VALUES BECOME SCORES (no hand-picked constants):
Each check produces a raw number. During fit(), the same checks are run
on every normal training window, which gives a reference list of "normal"
raw values. A new window's sub-score says how unusual its raw value is
compared with that list:
  - inside the normal range: fraction of normal windows with a lower value,
    scaled into [0, n/(n+1))
  - above everything seen in normal training: a value in (n/(n+1), 1),
    growing the further it is above the normal maximum
So scores are always below 1.0, and "higher than all normal training
windows" always scores above anything seen in normal training.
The Stage 1 score is the highest of the three sub-scores.

KNOWN LIMITS (expected, not bugs):
  - Masquerade and payload-only attacks keep normal timing -> not detected
    by Stage 1. Stage 2 (payload/signals) is needed for those.
  - ROAD accelerator captures contain no injected frames -> not visible here.
"""

import bisect
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from pydantic import BaseModel, ConfigDict, Field

from shared.schemas import TrafficWindow

CHECKS = ("unknown_ids", "rate_excess", "short_gaps")

# A gap of exactly 0 seconds (two frames with the same timestamp) is
# treated as this tiny gap so we never divide by zero.
_MIN_GAP_SECONDS = 1e-6


class Stage1Result(BaseModel):
    """Stage 1 output for one window (internal to the Defender)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    score: float = Field(ge=0.0, le=1.0)
    sub_scores: Dict[str, float]
    evidence: str


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _window_duration(window: TrafficWindow) -> float:
    duration = window.window_end - window.window_start
    if duration <= 0:
        raise ValueError(
            f"window {window.window_id} has duration {duration}s; "
            f"Stage 1 needs window_end > window_start"
        )
    return duration


def _frames_by_id(window: TrafficWindow) -> Dict[str, List[float]]:
    """Group frame timestamps by CAN ID (IDs upper-cased: '0f4' == '0F4')."""
    grouped: Dict[str, List[float]] = defaultdict(list)
    for frame in window.frames:
        grouped[frame.can_id.upper()].append(frame.timestamp)
    for times in grouped.values():
        times.sort()
    return grouped


def _unusualness(value: float, reference: List[float]) -> float:
    """Turn a raw value into a 0..1 score using normal reference values."""
    n = len(reference)
    maximum = reference[-1]
    if value <= maximum:
        below = bisect.bisect_left(reference, value)   # normal values < value
        return below / (n + 1)
    scale = maximum if maximum > 0 else 1.0
    tail = 1.0 - 1.0 / (1.0 + (value - maximum) / scale)   # 0..1, grows with distance
    return (n + tail) / (n + 1)


# ---------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------
class Stage1Model:
    def __init__(self):
        self.fitted = False
        self.known_ids: List[str] = []
        self.normal_rate: Dict[str, float] = {}        # frames per second
        self.normal_median_gap: Dict[str, float] = {}  # seconds
        self.reference: Dict[str, List[float]] = {}    # sorted raw values per check
        self.training_windows = 0

    # ---------- training ----------------------------------------------
    def fit(self, normal_windows: List[TrafficWindow]) -> "Stage1Model":
        """Learn normal timing from NORMAL TRAINING windows only."""
        if not isinstance(normal_windows, (list, tuple)) or len(normal_windows) < 2:
            raise ValueError("fit() needs a list of at least 2 normal training windows")
        for i, window in enumerate(normal_windows):
            if not isinstance(window, TrafficWindow):
                raise TypeError(
                    f"normal_windows[{i}] is {type(window).__name__}, expected TrafficWindow"
                )

        total_seconds = 0.0
        total_frames: Dict[str, int] = defaultdict(int)
        gaps: Dict[str, List[float]] = defaultdict(list)
        for window in normal_windows:
            total_seconds += _window_duration(window)
            for can_id, times in _frames_by_id(window).items():
                total_frames[can_id] += len(times)
                gaps[can_id].extend(b - a for a, b in zip(times, times[1:]))

        self.known_ids = sorted(total_frames)
        self.normal_rate = {i: total_frames[i] / total_seconds for i in self.known_ids}
        self.normal_median_gap = {
            i: max(statistics.median(g), _MIN_GAP_SECONDS) for i, g in gaps.items() if g
        }
        self.fitted = True   # needed so _raw_values can run

        raw_lists: Dict[str, List[float]] = {check: [] for check in CHECKS}
        for window in normal_windows:
            raw, _ = self._raw_values(window)
            for check in CHECKS:
                raw_lists[check].append(raw[check])
        self.reference = {check: sorted(values) for check, values in raw_lists.items()}
        self.training_windows = len(normal_windows)
        return self

    # ---------- raw checks --------------------------------------------
    def _raw_values(self, window: TrafficWindow) -> Tuple[Dict[str, float], Dict[str, str]]:
        """Raw value and evidence text for each check."""
        duration = _window_duration(window)
        grouped = _frames_by_id(window)
        known = set(self.known_ids)
        raw: Dict[str, float] = {}
        evidence: Dict[str, str] = {}

        # 1. unknown IDs
        unknown = {i: len(t) for i, t in grouped.items() if i not in known}
        raw["unknown_ids"] = float(sum(unknown.values()))
        if unknown:
            worst = max(unknown, key=unknown.get)
            evidence["unknown_ids"] = (
                f"{len(unknown)} unknown CAN ID(s), e.g. {worst} ({unknown[worst]} frames)"
            )

        # 2. rate excess: highest (rate in window / normal rate) over known IDs
        best_ratio, best_id = 0.0, None
        for can_id, times in grouped.items():
            if can_id in self.normal_rate:
                ratio = (len(times) / duration) / self.normal_rate[can_id]
                if ratio > best_ratio:
                    best_ratio, best_id = ratio, can_id
        raw["rate_excess"] = best_ratio
        if best_id:
            evidence["rate_excess"] = f"ID {best_id}: {best_ratio:.1f}x normal frame rate"

        # 3. short gaps: highest (normal median gap / shortest gap in window)
        best_ratio, best_id, best_gap = 0.0, None, 0.0
        for can_id, times in grouped.items():
            if can_id in self.normal_median_gap and len(times) >= 2:
                shortest = max(min(b - a for a, b in zip(times, times[1:])), _MIN_GAP_SECONDS)
                ratio = self.normal_median_gap[can_id] / shortest
                if ratio > best_ratio:
                    best_ratio, best_id, best_gap = ratio, can_id, shortest
        raw["short_gaps"] = best_ratio
        if best_id:
            evidence["short_gaps"] = (
                f"ID {best_id}: shortest gap {best_gap * 1000:.2f} ms, "
                f"normal ~{self.normal_median_gap[best_id] * 1000:.2f} ms"
            )
        return raw, evidence

    # ---------- scoring -----------------------------------------------
    def score(self, window: TrafficWindow) -> Stage1Result:
        """Score one window. Does not modify the window."""
        if not self.fitted:
            raise RuntimeError("Stage 1 is not trained. Call fit() or load() first.")
        if not isinstance(window, TrafficWindow):
            raise TypeError(f"score() needs a TrafficWindow, got {type(window).__name__}")

        raw, evidence = self._raw_values(window)
        sub_scores = {c: _unusualness(raw[c], self.reference[c]) for c in CHECKS}
        top_check = max(CHECKS, key=lambda c: sub_scores[c])
        top_score = sub_scores[top_check]
        normal_top = self.training_windows / (self.training_windows + 1)
        if top_score > normal_top and top_check in evidence:
            text = f"Timing anomaly ({top_check}): {evidence[top_check]}"
        else:
            text = "No timing anomaly beyond normal training range"
        return Stage1Result(score=top_score, sub_scores=sub_scores, evidence=text)

    # ---------- save / load -------------------------------------------
    def save(self, path, model_version: str) -> None:
        if not self.fitted:
            raise RuntimeError("Cannot save an untrained Stage 1 model")
        if not isinstance(model_version, str) or not model_version:
            raise ValueError("model_version must be a non-empty string, e.g. 'v1'")
        record = {
            "model_version": model_version,
            "known_ids": self.known_ids,
            "normal_rate": self.normal_rate,
            "normal_median_gap": self.normal_median_gap,
            "reference": self.reference,
            "training_windows": self.training_windows,
        }
        file = Path(path)
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(json.dumps(record, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path, expected_model_version: str) -> "Stage1Model":
        file = Path(path)
        if not file.exists():
            raise FileNotFoundError(f"No Stage 1 model at {file}. Train and save it first.")
        record = json.loads(file.read_text(encoding="utf-8"))
        if record.get("model_version") != expected_model_version:
            raise ValueError(
                f"Model version mismatch: Stage 1 file is {record.get('model_version')!r}, "
                f"expected {expected_model_version!r}"
            )
        model = cls()
        model.known_ids = record["known_ids"]
        model.normal_rate = record["normal_rate"]
        model.normal_median_gap = record["normal_median_gap"]
        model.reference = record["reference"]
        model.training_windows = record["training_windows"]
        model.fitted = True
        return model