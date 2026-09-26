"""
Stage 2 payload detector for the EdgeGuard Defender (Part 2).

Stage 2 looks at PAYLOAD CONTENT, never timing (that is Stage 1's job).
It is meant to catch masquerade and other attacks that keep normal
message timing but change what a message says.

WATCH-LIST (important for real data): Stage 2 can be limited to a short
list of safety-relevant CAN IDs (watch_ids). On real ROAD data, watching
all ~100 IDs flagged 170 of 184 NORMAL validation windows, because some
field in some ID always does something new in a new drive. Watching only
the attacked IDs (0D0, 6E0, chosen from DEVELOPMENT attacks) flagged 0 of
184. Attacks on IDs outside the watch-list are NOT seen by Stage 2; that
limit must be stated in results. watch_ids=None watches every ID.

It learns "normal" from NORMAL TRAINING windows only, per CAN ID and per
"field" inside that ID's payload:
    byte fields   each payload byte position (byte0, byte1, ...)
    pair fields   each pair of consecutive bytes, as a 16-bit number,
                  in BOTH byte orders (pair0_be, pair0_le, pair1_be, ...)
                  This catches 2-byte signals regardless of byte order,
                  without needing Part 1's decoded signals.

For a new window it checks three signs, per field:
    out_of_range   a value outside the range ever seen in normal training
                   (typical of "max_speedometer", "max_engine_coolant_temp")
    frozen_break   a field that normally varies is held at one exact value
                   for the whole window (typical of "reverse_light_on/off",
                   and of masquerade attacks that hold a forced value)
    large_jump     two consecutive frames of the same field differ by more
                   than any jump ever seen in normal training

Raw values become 0..1 scores the same way as Stage 1 (see
defender.stage1._unusualness): compared against the same checks run on
every normal training window. The Stage 2 score is the highest sub-score
found on any field.

OUT-OF-RANGE BOUNDS (bound_percentile):
    None (v1/v2/v3, default)  out_of_range compares against the exact
             min/max ever seen in normal training. A single rare-but-normal
             extreme value (e.g. one drive's brief spike) becomes part of
             the "normal" range forever, which is safe but means one odd
             training drive can also make the bound too WIDE to catch a
             real attack that stays inside it.
    a number 0 < p < 50   out_of_range instead compares against the p-th and
             (100-p)-th percentile of the values seen at each field CHANGE
             during training (not frame-weighted -- see fit()'s docstring),
             so the most extreme observed values no longer define the
             boundary. This directly targets false alarms coming from one
             unusual training drive (e.g. "extended_short" in leave-one-out
             cross-validation) at the cost of maybe missing an attack whose
             values fall inside the trimmed tail. Choose p by comparing
             `python -m defender.crossval ambient` false-alarm rates with
             and without it -- never by looking at final_test.
             Only out_of_range is affected: frozen_break's "width" strength
             and large_jump's baseline still use the full training range,
             so this cannot make those checks MORE sensitive as a side
             effect of a change aimed only at out_of_range.

FROZEN CHECK MODES (frozen_mode):
    "width"  (v2, default)  strength = the field's normal range width.
             Weakness found by Part 3 on real data: normal windows already
             contain wide fields that are constant within one second, so a
             frozen rolling counter (0D0 bytes 1 and 5) did not stand out.
    "rate"   (v3)  strength = how SURPRISING it is for this field to be
             constant for a whole window, learned from normal training
             windows: surprise = -log((constant_windows + 1) / (windows + 2)).
             A counter or checksum that changes in every normal window is
             maximally surprising when frozen; a slow signal that is often
             constant is not.
    Old model files without frozen_mode load as "width", so v2 is unchanged.

KNOWN LIMITS (expected, not bugs):
  - "accelerator" ROAD attacks inject no frames at all -> Stage 2 has
    nothing to see; it will not catch them (neither does Stage 1).
  - "correlated_signal_attack" changes a RELATIONSHIP between two CAN IDs,
    not one field's own range. This detector checks fields independently
    and is NOT expected to reliably catch it. That needs a separate
    cross-ID check (a later addition), or Part 1's decoded_signals.
  - A field that is a rolling counter or checksum in training will show
    a wide "normal" range, so out_of_range contributes little for it.
    This is fine: the window score is the MAX across all fields, so a
    noisy field does not hide a real anomaly elsewhere.
"""

import heapq
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from defender.stage1 import _unusualness
from shared.schemas import TrafficWindow

CHECKS = ("out_of_range", "frozen_break", "large_jump")


class Stage2Result(BaseModel):
    """Stage 2 output for one window (internal to the Defender)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    score: float = Field(ge=0.0, le=1.0)
    sub_scores: Dict[str, float]
    evidence: str


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def normalize_can_id(value: str) -> str:
    """'0xd0' / '0D0' / 'd0' -> '0D0' (the log and schema format)."""
    text = str(value).strip().upper()
    if text.startswith("0X"):
        text = text[2:]
    if not text or any(c not in "0123456789ABCDEF" for c in text):
        raise ValueError(f"{value!r} is not a hexadecimal CAN ID")
    return text.zfill(3)


def _payload_bytes(payload: str) -> List[int]:
    return [int(payload[i:i + 2], 16) for i in range(0, len(payload), 2)]


def _payload_fields(data: List[int]):
    """(field_name, value) for every byte and byte-pair field of one payload."""
    for i, byte in enumerate(data):
        yield f"byte{i}", byte
    for i in range(len(data) - 1):
        yield f"pair{i}_be", (data[i] << 8) | data[i + 1]
        yield f"pair{i}_le", (data[i + 1] << 8) | data[i]


def _field_values(window: TrafficWindow, only_ids=None) -> Dict[Tuple[str, str], List[int]]:
    """(can_id, field_name) -> values, in timestamp order, for this window.
    only_ids: optional set of CAN IDs; other IDs are skipped early (speed)."""
    by_id: Dict[str, List[Tuple[float, List[int]]]] = defaultdict(list)
    for frame in window.frames:
        can_id = frame.can_id.upper()
        if only_ids is not None and can_id not in only_ids:
            continue
        by_id[can_id].append((frame.timestamp, _payload_bytes(frame.payload)))

    fields: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for can_id, entries in by_id.items():
        entries.sort(key=lambda e: e[0])
        for _, data in entries:
            for name, value in _payload_fields(data):
                fields[(can_id, name)].append(value)
    return fields


# ---------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------
FROZEN_MODES = ("width", "rate")


# low, high, max_jump, raw_values (values seen at each field change, for
# percentile out-of-range bounds -- see bound_percentile above)
RangeStats = Tuple[Dict[str, int], Dict[str, int], Dict[str, int], Dict[str, List[int]]]


def merge_range_stats(stats: Iterable[RangeStats]) -> RangeStats:
    """Combine per-capture (low, high, max_jump, raw_values) into one:
    min, max, max, and a SORTED merge (each capture's raw_values arrives
    pre-sorted from capture_range_stats(), so this is an O(n log k) k-way
    merge across k captures, not an O(n log n) re-sort of everything --
    matters when this runs inside cross-validation's nested refits)."""
    low: Dict[str, int] = {}
    high: Dict[str, int] = {}
    jump: Dict[str, int] = {}
    raw_lists: Dict[str, List[List[int]]] = defaultdict(list)
    for lo, hi, ju, rv in stats:
        for k, v in lo.items():
            low[k] = min(low.get(k, v), v)
        for k, v in hi.items():
            high[k] = max(high.get(k, v), v)
        for k, v in ju.items():
            jump[k] = max(jump.get(k, v), v)
        for k, values in rv.items():
            raw_lists[k].append(values)
    raw_values = {k: list(heapq.merge(*lists)) for k, lists in raw_lists.items()}
    return low, high, jump, raw_values


def _percentile(sorted_values: List[float], pct: float) -> float:
    """Linear-interpolation percentile of an already-sorted, non-empty list."""
    if not sorted_values:
        raise ValueError("cannot take a percentile of an empty list")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    lo_idx, hi_idx = math.floor(rank), math.ceil(rank)
    if lo_idx == hi_idx:
        return float(sorted_values[lo_idx])
    frac = rank - lo_idx
    return sorted_values[lo_idx] + (sorted_values[hi_idx] - sorted_values[lo_idx]) * frac


def _percentile_bounds(sorted_values: List[float], bound_percentile: float) -> Tuple[float, float]:
    return (_percentile(sorted_values, bound_percentile),
            _percentile(sorted_values, 100.0 - bound_percentile))


class Stage2Model:
    def __init__(self, watch_ids: Optional[Iterable[str]] = None,
                 frozen_mode: str = "width",
                 bound_percentile: Optional[float] = None):
        if frozen_mode not in FROZEN_MODES:
            raise ValueError(f"frozen_mode must be one of {FROZEN_MODES}, got {frozen_mode!r}")
        if bound_percentile is not None and not (0 < bound_percentile < 50):
            raise ValueError(
                f"bound_percentile must be between 0 and 50 (exclusive), e.g. 1.0 to trim "
                f"the extreme 1% off each side, got {bound_percentile!r}"
            )
        self.frozen_mode = frozen_mode
        self.bound_percentile = bound_percentile
        # "ID|field" -> [windows where constant, windows with >= 2 values]
        self.frozen_counts: Dict[str, List[int]] = {}
        if watch_ids is not None:
            watch = sorted({normalize_can_id(i) for i in watch_ids})
            if not watch:
                raise ValueError("watch_ids is empty; use None to watch every ID")
            self.watch_ids: Optional[List[str]] = watch
        else:
            self.watch_ids = None
        self.fitted = False
        self.field_range: Dict[str, Tuple[int, int]] = {}    # "ID|field" -> (min, max), always full range
        # out_of_range bounds actually used for scoring: == field_range unless
        # bound_percentile narrows it (see fit()).
        self.field_bounds: Dict[str, Tuple[float, float]] = {}
        self.field_max_jump: Dict[str, int] = {}
        self.reference: Dict[str, List[float]] = {}          # sorted raw values per check
        self.training_windows = 0

    @staticmethod
    def _key(can_id: str, field_name: str) -> str:
        return f"{can_id}|{field_name}"

    def _watched(self, can_id: str) -> bool:
        return self.watch_ids is None or can_id in self.watch_ids

    # ---------- training ----------------------------------------------
    def capture_range_stats(self, capture: Iterable[Tuple[float, str, str]]) -> RangeStats:
        """Field ranges, largest jumps, and raw changed-values of ONE full
        capture (watched IDs only).

        Returns (low, high, max_jump, raw_values), keyed "ID|field". Stats of
        several captures combine exactly by min / max / max / concatenate
        (merge_range_stats), so cross-validation can read each capture once
        and reuse the result in every fold instead of re-reading the logs per
        fold. raw_values is always collected (cheap: the watch-list is small)
        so a Stage2Model can be fit with any bound_percentile from the same
        stats without re-reading the logs.
        """
        low: Dict[str, int] = {}
        high: Dict[str, int] = {}
        max_jump: Dict[str, int] = defaultdict(int)
        raw_values: Dict[str, List[int]] = defaultdict(list)
        last_payload: Dict[str, str] = {}
        last_value: Dict[str, int] = {}
        for _, can_id, payload in capture:
            can_id = can_id.upper()
            if not self._watched(can_id):
                continue          # not on the watch-list
            if last_payload.get(can_id) == payload:
                continue          # identical payload: no new range or jump
            last_payload[can_id] = payload
            for name, value in _payload_fields(_payload_bytes(payload)):
                key = can_id + "|" + name
                previous = last_value.get(key)
                if previous == value:
                    continue      # this field did not change
                last_value[key] = value
                raw_values[key].append(value)
                if previous is None:
                    if key not in low or value < low[key]:
                        low[key] = value
                    if key not in high or value > high[key]:
                        high[key] = value
                    continue
                if value < low[key]:
                    low[key] = value
                elif value > high[key]:
                    high[key] = value
                jump = value - previous if value > previous else previous - value
                if jump > max_jump[key]:
                    max_jump[key] = jump
        # Sorted once here (per capture) so merge_range_stats can k-way-merge
        # instead of re-sorting the concatenation on every fit() call.
        return low, high, dict(max_jump), {k: sorted(v) for k, v in raw_values.items()}

    def fit(self, normal_windows: List[TrafficWindow],
            range_captures: Optional[Iterable[Iterable[Tuple[float, str, str]]]] = None,
            range_stats: Optional[Iterable[RangeStats]] = None,
            ) -> "Stage2Model":
        """Learn normal payload content from NORMAL TRAINING data only.

        normal_windows: training windows. Always used for calibration.
        range_captures: optional. The FULL frame streams of the SAME training
            captures, one iterable of (timestamp, can_id, payload) per capture
            (e.g. part1.pipeline.RoadData.frames(name)). If given, field ranges and
            jump sizes are learned from EVERY frame instead of only the
            windows, which matters when windows are sampled: rare but normal
            changes are then not mistaken for attacks.
        range_stats: optional, instead of range_captures. Precomputed
            capture_range_stats() of the SAME training captures, from a
            Stage2Model with the SAME watch-list. Gives an identical model.

        NOTE on bound_percentile: the percentile is computed over the values
        seen at each field CHANGE (the same de-duplicated sequence
        capture_range_stats() and _field_values() already track for jumps),
        not over every frame weighted by how long it held that value. This is
        a deliberate simplification, not a frame-weighted percentile -- rare
        brief extremes still appear once in the sample, which is what
        out-of-range trimming cares about.
        """
        if not isinstance(normal_windows, (list, tuple)) or len(normal_windows) < 2:
            raise ValueError("fit() needs a list of at least 2 normal training windows")
        for i, window in enumerate(normal_windows):
            if not isinstance(window, TrafficWindow):
                raise TypeError(
                    f"normal_windows[{i}] is {type(window).__name__}, expected TrafficWindow"
                )
        if range_captures is not None and range_stats is not None:
            raise ValueError("pass range_captures or range_stats, not both")

        low: Dict[str, int] = {}
        high: Dict[str, int] = {}
        max_jump: Dict[str, int] = defaultdict(int)
        raw_values: Dict[str, List[int]] = defaultdict(list)

        def update(key, values):
            low[key] = min(low.get(key, values[0]), min(values))
            high[key] = max(high.get(key, values[0]), max(values))
            for a, b in zip(values, values[1:]):
                if abs(b - a) > max_jump[key]:
                    max_jump[key] = abs(b - a)
            raw_values[key].extend(values)

        if range_captures is not None:
            range_stats = [self.capture_range_stats(c) for c in range_captures]
        if range_stats is None:
            for window in normal_windows:
                for (can_id, field_name), values in _field_values(window).items():
                    if self._watched(can_id):
                        update(self._key(can_id, field_name), values)
        else:
            merged = merge_range_stats(range_stats)
            low.update(merged[0])
            high.update(merged[1])
            max_jump.update(merged[2])
            for k, values in merged[3].items():
                raw_values[k].extend(values)
            # Windows come from the same captures, but include them too so no
            # field seen in calibration is missing a range.
            for window in normal_windows:
                for (can_id, field_name), values in _field_values(window).items():
                    if self._watched(can_id):
                        update(self._key(can_id, field_name), values)

        self.field_range = {k: (low[k], high[k]) for k in low}
        self.field_max_jump = dict(max_jump)
        if self.bound_percentile is not None:
            self.field_bounds = {
                k: _percentile_bounds(sorted(values), self.bound_percentile)
                for k, values in raw_values.items() if values
            }
            for k in low:
                self.field_bounds.setdefault(k, (float(low[k]), float(high[k])))
        else:
            self.field_bounds = {k: (float(low[k]), float(high[k])) for k in low}

        # How often is each field constant within a normal training window?
        counts: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
        only = set(self.watch_ids) if self.watch_ids is not None else None
        for window in normal_windows:
            for (can_id, field_name), values in _field_values(window, only).items():
                if len(values) >= 2:
                    entry = counts[self._key(can_id, field_name)]
                    entry[1] += 1
                    if min(values) == max(values):
                        entry[0] += 1
        self.frozen_counts = {k: list(v) for k, v in counts.items()}
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
        raw = {check: 0.0 for check in CHECKS}
        evidence: Dict[str, str] = {}

        only = set(self.watch_ids) if self.watch_ids is not None else None
        for (can_id, field_name), values in _field_values(window, only).items():
            if not self._watched(can_id):
                continue    # not on the watch-list
            key = self._key(can_id, field_name)
            if key not in self.field_range:
                continue    # unseen field: covered by Stage 1's unknown_ids, not here
            low, high = self.field_range[key]            # full training range
            width = high - low                            # used by frozen_break only
            bound_low, bound_high = self.field_bounds.get(key, (low, high))
            bound_width = bound_high - bound_low          # used by out_of_range only

            # 1. out_of_range (percentile-trimmed bounds when bound_percentile is set)
            overage = max((bound_low - v if v < bound_low else v - bound_high if v > bound_high else 0)
                          for v in values)
            if overage > 0:
                ratio = overage / (bound_width if bound_width > 0 else 1)
                if ratio > raw["out_of_range"]:
                    raw["out_of_range"] = ratio
                    note = "" if self.bound_percentile is None else                         f" (training range [{low}, {high}])"
                    evidence["out_of_range"] = (
                        f"ID {can_id} {field_name}: value out of normal range "
                        f"[{bound_low}, {bound_high}] by {overage}{note}"
                    )

            # 2. frozen_break: held constant for the whole window (always the
            # FULL training width/rate -- never narrowed by bound_percentile,
            # which is aimed only at out_of_range)
            if len(values) >= 2 and min(values) == max(values):
                if self.frozen_mode == "width":
                    if width > 0 and width > raw["frozen_break"]:
                        raw["frozen_break"] = float(width)
                        evidence["frozen_break"] = (
                            f"ID {can_id} {field_name}: frozen at {values[0]} for the whole "
                            f"window (normally varies {low}-{high})"
                        )
                elif key in self.frozen_counts:
                    constant, seen = self.frozen_counts[key]
                    surprise = -math.log((constant + 1) / (seen + 2))
                    if surprise > raw["frozen_break"]:
                        raw["frozen_break"] = surprise
                        evidence["frozen_break"] = (
                            f"ID {can_id} {field_name}: frozen at {values[0]} for the whole "
                            f"window (constant in only {constant} of {seen} normal windows)"
                        )

            # 3. large_jump
            if len(values) >= 2:
                jump = max(abs(b - a) for a, b in zip(values, values[1:]))
                normal_jump = self.field_max_jump.get(key, 0)
                ratio = jump / normal_jump if normal_jump > 0 else float(jump)
                if jump > normal_jump and ratio > raw["large_jump"]:
                    raw["large_jump"] = ratio
                    evidence["large_jump"] = (
                        f"ID {can_id} {field_name}: jump of {jump} between consecutive "
                        f"frames, normal max jump {normal_jump}"
                    )
        return raw, evidence

    # ---------- scoring -----------------------------------------------
    def score(self, window: TrafficWindow) -> Stage2Result:
        """Score one window. Does not modify the window."""
        if not self.fitted:
            raise RuntimeError("Stage 2 is not trained. Call fit() or load() first.")
        if not isinstance(window, TrafficWindow):
            raise TypeError(f"score() needs a TrafficWindow, got {type(window).__name__}")

        raw, evidence = self._raw_values(window)
        sub_scores = {c: _unusualness(raw[c], self.reference[c]) for c in CHECKS}
        top_check = max(CHECKS, key=lambda c: sub_scores[c])
        top_score = sub_scores[top_check]
        normal_top = self.training_windows / (self.training_windows + 1)
        if top_score > normal_top and top_check in evidence:
            text = f"Payload anomaly ({top_check}): {evidence[top_check]}"
        else:
            text = "No payload anomaly beyond normal training range"
        return Stage2Result(score=top_score, sub_scores=sub_scores, evidence=text)

    # ---------- save / load -------------------------------------------
    def save(self, path, model_version: str) -> None:
        if not self.fitted:
            raise RuntimeError("Cannot save an untrained Stage 2 model")
        if not isinstance(model_version, str) or not model_version:
            raise ValueError("model_version must be a non-empty string, e.g. 'v2'")
        record = {
            "model_version": model_version,
            "watch_ids": self.watch_ids,
            "frozen_mode": self.frozen_mode,
            "bound_percentile": self.bound_percentile,
            "frozen_counts": self.frozen_counts,
            "field_range": self.field_range,
            "field_bounds": self.field_bounds,
            "field_max_jump": self.field_max_jump,
            "reference": self.reference,
            "training_windows": self.training_windows,
        }
        file = Path(path)
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(json.dumps(record, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path, expected_model_version: str) -> "Stage2Model":
        file = Path(path)
        if not file.exists():
            raise FileNotFoundError(f"No Stage 2 model at {file}. Train and save it first.")
        record = json.loads(file.read_text(encoding="utf-8"))
        if record.get("model_version") != expected_model_version:
            raise ValueError(
                f"Model version mismatch: Stage 2 file is {record.get('model_version')!r}, "
                f"expected {expected_model_version!r}"
            )
        model = cls(watch_ids=record.get("watch_ids"),
                    frozen_mode=record.get("frozen_mode", "width"),
                    bound_percentile=record.get("bound_percentile"))
        model.frozen_counts = record.get("frozen_counts", {})
        model.field_range = {k: tuple(v) for k, v in record["field_range"].items()}
        # Old model files (no field_bounds) predate percentile bounds: bounds == range.
        model.field_bounds = {k: tuple(v) for k, v in
                              record.get("field_bounds", record["field_range"]).items()}
        model.field_max_jump = record["field_max_jump"]
        model.reference = record["reference"]
        model.training_windows = record["training_windows"]
        model.fitted = True
        return model