"""
Training for the EdgeGuard Defender (Part 2).

v1 training does two things, on two SEPARATE groups of NORMAL captures:
  1. TRAIN split       -> fit Stage 1 (learn normal timing)
  2. VALIDATION split  -> score those windows, choose the threshold with
                          choose_threshold(scores, max_false_alarm_rate)
Then it saves, for one model version:
  models/stage1_<version>.json
  models/threshold_<version>.json
  models/train_info_<version>.json   (what was used, for reproducibility)

v2 training (train_v2) REUSES v1's Stage 1 model UNCHANGED and adds a NEW
Stage 2 (payload) model, fused (see defender.fusion). Stage 1 is never
retrained for v2, so the only possible difference between v1 and v2 is
Stage 2: this keeps the evidence-gate claim exact. It additionally saves
  models/stage2_<version>.json
and requires the SAME train and validation captures as v1 was built with,
so the comparison isolates Stage 2's contribution.

Rules enforced here:
  - The false-alarm rate has NO default for v1 (the caller must supply
    it); train_v2 defaults to v1's own rate unless told otherwise, so the
    two versions are compared on the same false-alarm budget.
  - A capture may not appear in both train and validation (data leak).
  - Existing model files are never overwritten unless overwrite=True,
    so a model already judged by the evidence gate cannot change silently.

Rules the CALLER must follow (cannot be checked here, labels are hidden):
  - Train and validation captures must be NORMAL (ambient) captures.
  - Never pass development or final-test captures.

Part 1 interface this file is written against (agreed signature):
  make_windows(capture_path, capture_id, window_s, stride_s)
      -> Iterator[TrafficWindow]   (frames filled, features empty)
"""

import json
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field

from defender.defender import stage1_path, stage2_path, threshold_path, train_info_file
from defender.fusion import fuse
from defender.stage1 import Stage1Model
from defender.stage2 import Stage2Model
from defender.threshold import ATTACK, choose_threshold, make_decision, save_threshold
from shared.schemas import TrafficWindow

MakeWindows = Callable[[str, str, float, float], Iterator[TrafficWindow]]


def train_info_path(model_dir, model_version: str) -> Path:
    return train_info_file(model_dir, model_version)


class TrainingReport(BaseModel):
    """Summary of one training run, saved as train_info_<version>.json."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_version: str
    window_s: float
    stride_s: float
    train_capture_ids: List[str]
    validation_capture_ids: List[str]
    train_windows: int
    validation_windows: int
    max_false_alarm_rate: float
    threshold: float = Field(ge=0.0, le=1.0)
    validation_false_alarm_rate: float = Field(ge=0.0, le=1.0)
    # Set only by train_v2: the v1 model whose Stage 1 was reused unchanged.
    # None for a v1 report, so old train_info files still match this shape.
    base_stage1_version: Optional[str] = None
    # True if Stage 2 learned field ranges from ALL frames of the train
    # captures (not only the sampled windows). False for v1 reports.
    stage2_ranges_from_full_captures: bool = False
    # Stage 2 watch-list used for this model (None = every ID; None for v1).
    stage2_watch_ids: Optional[List[str]] = None
    # Stage 2 frozen check mode ("width" = v2, "rate" = v3). None for v1.
    stage2_frozen_mode: Optional[str] = None
    # Stage 2 out-of-range bound percentile (None = strict min/max). None for v1.
    stage2_bound_percentile: Optional[float] = None


def _to_list(windows: Iterable[TrafficWindow], name: str) -> List[TrafficWindow]:
    result = list(windows)
    for i, window in enumerate(result):
        if not isinstance(window, TrafficWindow):
            raise TypeError(f"{name}[{i}] is {type(window).__name__}, expected TrafficWindow")
    return result


def train_defender(
    train_windows: Iterable[TrafficWindow],
    validation_windows: Iterable[TrafficWindow],
    max_false_alarm_rate: float,
    model_version: str,
    model_dir,
    window_s: float,
    stride_s: float,
    overwrite: bool = False,
) -> TrainingReport:
    """Fit Stage 1 on train windows, choose the threshold on validation windows,
    save everything for model_version. Returns a TrainingReport."""
    if not isinstance(model_version, str) or not model_version:
        raise ValueError("model_version must be a non-empty string, e.g. 'v1'")
    if window_s <= 0 or stride_s <= 0:
        raise ValueError(f"window_s and stride_s must be positive, got {window_s}, {stride_s}")

    train = _to_list(train_windows, "train_windows")
    validation = _to_list(validation_windows, "validation_windows")
    if len(train) < 2:
        raise ValueError(f"need at least 2 train windows, got {len(train)}")
    if len(validation) == 0:
        raise ValueError("validation_windows is empty: the threshold needs validation data")

    train_ids = sorted({w.capture_id for w in train})
    validation_ids = sorted({w.capture_id for w in validation})
    shared_ids = set(train_ids) & set(validation_ids)
    if shared_ids:
        raise ValueError(
            f"Data leak: capture(s) {sorted(shared_ids)} appear in BOTH train and "
            f"validation. Each capture must belong to exactly one split."
        )

    outputs = [
        stage1_path(model_dir, model_version),
        threshold_path(model_dir, model_version),
        train_info_path(model_dir, model_version),
    ]
    existing = [str(p) for p in outputs if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Model {model_version!r} already exists ({existing}). Use a new version "
            f"name, or overwrite=True if you are sure it was never evaluated."
        )

    # 1. Learn normal timing from TRAIN only.
    stage1 = Stage1Model().fit(train)

    # 2. Choose the threshold from VALIDATION only.
    validation_scores = [stage1.score(w).score for w in validation]
    threshold = choose_threshold(validation_scores, max_false_alarm_rate)
    alarms = sum(1 for s in validation_scores if make_decision(s, threshold) == ATTACK)

    report = TrainingReport(
        model_version=model_version,
        window_s=window_s,
        stride_s=stride_s,
        train_capture_ids=train_ids,
        validation_capture_ids=validation_ids,
        train_windows=len(train),
        validation_windows=len(validation),
        max_false_alarm_rate=max_false_alarm_rate,
        threshold=threshold,
        validation_false_alarm_rate=alarms / len(validation),
    )

    # 3. Save all three files for this model version.
    stage1.save(outputs[0], model_version)
    save_threshold(threshold, model_version, outputs[1])
    outputs[2].write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
    return report


def train_from_captures(
    train_captures: Sequence[Tuple[str, str]],
    validation_captures: Sequence[Tuple[str, str]],
    make_windows: MakeWindows,
    window_s: float,
    stride_s: float,
    max_false_alarm_rate: float,
    model_version: str,
    model_dir,
    overwrite: bool = False,
) -> TrainingReport:
    """Same as train_defender, but builds windows with Part 1's make_windows().

    train_captures / validation_captures: lists of (capture_path, capture_id),
    e.g. [("data/road/ambient/<file>.log", "cap01"), ...] from the split manifest.
    """
    def collect(captures):
        windows = []
        for path, capture_id in captures:
            windows.extend(make_windows(path, capture_id, window_s, stride_s))
        return windows

    return train_defender(
        train_windows=collect(train_captures),
        validation_windows=collect(validation_captures),
        max_false_alarm_rate=max_false_alarm_rate,
        model_version=model_version,
        model_dir=model_dir,
        window_s=window_s,
        stride_s=stride_s,
        overwrite=overwrite,
    )


def train_v2(
    v1_model_dir,
    v1_version: str,
    train_windows: Iterable[TrafficWindow],
    validation_windows: Iterable[TrafficWindow],
    model_version: str,
    model_dir,
    window_s: float,
    stride_s: float,
    max_false_alarm_rate: Optional[float] = None,
    overwrite: bool = False,
    stage2_range_captures=None,
    stage2_watch_ids: Optional[Sequence[str]] = None,
    stage2_frozen_mode: str = "width",
    stage2_bound_percentile: Optional[float] = None,
) -> TrainingReport:
    """Build v2 = v1's Stage 1 (UNCHANGED) + a NEW Stage 2, fused, with a
    threshold re-chosen on the fused score. Returns a TrainingReport.

    train_windows / validation_windows MUST come from the SAME captures
    v1 was trained with (checked against v1's own train_info file), so
    Stage 2 is the only thing that differs between v1 and v2.

    max_false_alarm_rate: if None (the default), uses the SAME rate v1
    was trained with, so the two versions are compared on the same
    false-alarm budget. Pass a value explicitly to use a different one.

    stage2_range_captures: optional full frame streams of the SAME train
    captures (see Stage2Model.fit). Strongly recommended with real data when
    train windows are sampled.

    stage2_watch_ids: optional Stage 2 watch-list of CAN IDs (e.g. ["0D0",
    "6E0"]), chosen from DEVELOPMENT attacks only. None watches every ID.

    stage2_frozen_mode: "width" (v2) or "rate" (v3 hardening, see stage2.py).

    stage2_bound_percentile: optional Stage 2 out-of-range bound percentile
    (None = strict min/max, v1/v2/v3 behavior; see stage2.py for the trade-off).
    """
    if not isinstance(model_version, str) or not model_version:
        raise ValueError("model_version must be a non-empty string, e.g. 'v2'")
    if window_s <= 0 or stride_s <= 0:
        raise ValueError(f"window_s and stride_s must be positive, got {window_s}, {stride_s}")

    v1_stage1 = Stage1Model.load(stage1_path(v1_model_dir, v1_version), v1_version)
    v1_info = json.loads(train_info_path(v1_model_dir, v1_version).read_text(encoding="utf-8"))
    if max_false_alarm_rate is None:
        max_false_alarm_rate = v1_info["max_false_alarm_rate"]

    train = _to_list(train_windows, "train_windows")
    validation = _to_list(validation_windows, "validation_windows")
    if len(train) < 2:
        raise ValueError(f"need at least 2 train windows, got {len(train)}")
    if len(validation) == 0:
        raise ValueError("validation_windows is empty: the threshold needs validation data")

    train_ids = sorted({w.capture_id for w in train})
    validation_ids = sorted({w.capture_id for w in validation})
    shared_ids = set(train_ids) & set(validation_ids)
    if shared_ids:
        raise ValueError(
            f"Data leak: capture(s) {sorted(shared_ids)} appear in BOTH train and "
            f"validation. Each capture must belong to exactly one split."
        )
    if train_ids != v1_info["train_capture_ids"]:
        raise ValueError(
            f"train_v2 must use the SAME train captures as {v1_version} "
            f"({v1_info['train_capture_ids']}), got {train_ids}. This keeps the "
            f"only difference between {v1_version} and {model_version} as Stage 2."
        )
    if validation_ids != v1_info["validation_capture_ids"]:
        raise ValueError(
            f"train_v2 must use the SAME validation captures as {v1_version} "
            f"({v1_info['validation_capture_ids']}), got {validation_ids}."
        )

    outputs = [
        stage1_path(model_dir, model_version),
        stage2_path(model_dir, model_version),
        threshold_path(model_dir, model_version),
        train_info_path(model_dir, model_version),
    ]
    existing = [str(p) for p in outputs if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Model {model_version!r} already exists ({existing}). Use a new version "
            f"name, or overwrite=True if you are sure it was never evaluated."
        )

    # NEW Stage 2 only. Stage 1 is v1's, unchanged.
    stage2 = Stage2Model(watch_ids=stage2_watch_ids, frozen_mode=stage2_frozen_mode,
                         bound_percentile=stage2_bound_percentile).fit(
        train, range_captures=stage2_range_captures)

    fused_scores = [fuse(v1_stage1.score(w), stage2.score(w)).score for w in validation]
    threshold = choose_threshold(fused_scores, max_false_alarm_rate)
    alarms = sum(1 for s in fused_scores if make_decision(s, threshold) == ATTACK)

    report = TrainingReport(
        model_version=model_version,
        window_s=window_s,
        stride_s=stride_s,
        train_capture_ids=train_ids,
        validation_capture_ids=validation_ids,
        train_windows=len(train),
        validation_windows=len(validation),
        max_false_alarm_rate=max_false_alarm_rate,
        threshold=threshold,
        validation_false_alarm_rate=alarms / len(validation),
        base_stage1_version=v1_version,
        stage2_ranges_from_full_captures=stage2_range_captures is not None,
        stage2_watch_ids=stage2.watch_ids,
        stage2_frozen_mode=stage2.frozen_mode,
        stage2_bound_percentile=stage2.bound_percentile,
    )

    # Save v1's Stage 1 again under the v2 name (identical content, just
    # relabelled), so Defender.load(model_dir, model_version) finds a
    # self-consistent set of files, plus the NEW Stage 2 and threshold.
    v1_stage1.save(outputs[0], model_version)
    stage2.save(outputs[1], model_version)
    save_threshold(threshold, model_version, outputs[2])
    outputs[3].write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
    return report