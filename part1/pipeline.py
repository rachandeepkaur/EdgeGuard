"""
Part 1 -- the one entry point for data (build plan Block 1).

    from part1.pipeline import RoadData, preprocess
    road = RoadData("~/Downloads/road/dataset")          # frozen manifest by default
    for window in road.windows("ambient_dyno_drive_winter"):
        ...
    for window, label in road.labelled_windows("max_speedometer_attack_1"):
        ...                                               # label: evaluator only

Everything goes through the same steps, in the same order:
    read (fleet_simulator.iter_raw)  ->  clean (cleaning.clean)
    ->  window (windowing.raw_windows)  ->  label (labels.window_label)
and preprocess(window, decoder) fills decoded_signals AFTER any injection, so
normal and attacked windows are treated identically.

RoadData refuses final_test / separate captures unless it was created with
final_evaluation=True (split_manifest.Manifest.capture_path).

Windows carry only neutral ids (capture_id, vehicle_id). Names, kinds and
labels stay on the evaluator side: CaptureMetadata and GroundTruthLabel.
"""

from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict

from shared.schemas import GroundTruthLabel, TrafficWindow

from part1.cleaning import CleaningReport, clean
from part1.decode import Decoder
from part1.fleet_simulator import RawFrame, capture_span_us, iter_raw
from part1.labels import injection_rule, window_label
from part1.split_manifest import Manifest, load_manifest, load_road_metadata
from part1.windowing import (WINDOW_S, keep_every_for, raw_windows, to_traffic_window,
                             window_id_for, window_us_for)


class CaptureMetadata(BaseModel):
    """PRIVATE per-capture record for the evaluator (never sent to the Defender)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    capture_id: str
    vehicle_id: str
    group: str
    kind: str
    family: Optional[str]
    target: Optional[str]
    injection_interval: Optional[List[float]]     # elapsed seconds, closed
    duration_s: float                             # first to last frame
    first_raw_timestamp: str                      # as written in the log


class RoadData:
    def __init__(self, data_dir, manifest: Optional[Manifest] = None,
                 road_metadata: Optional[dict] = None, window_s: float = WINDOW_S,
                 final_evaluation: bool = False):
        self.data_dir = Path(data_dir).expanduser()
        self.manifest = manifest if manifest is not None else load_manifest()
        self.road_metadata = road_metadata if road_metadata is not None \
            else load_road_metadata(self.data_dir)
        self.window_s = window_s
        self.window_us = window_us_for(window_s, window_s)
        self.final_evaluation = final_evaluation

    @property
    def vehicle_id(self) -> str:
        return self.manifest.vehicle_id

    def path(self, name: str) -> Path:
        return self.manifest.capture_path(self.data_dir, name, self.final_evaluation)

    def capture_id(self, name: str) -> str:
        return self.manifest.entry(name).capture_id

    # ---------- frames ------------------------------------------------------
    def frames(self, name: str, report: Optional[CleaningReport] = None) -> Iterator[RawFrame]:
        """Cleaned (elapsed_us, can_id, payload) frames of one capture."""
        path = self.path(name)
        report = report if report is not None else CleaningReport()
        return clean(iter_raw(str(path), report), report, self.capture_id(name))

    # ---------- windows -----------------------------------------------------
    def keep_every(self, name: str, max_windows: int) -> int:
        """Step that keeps at most max_windows windows (training memory only)."""
        return keep_every_for(capture_span_us(str(self.path(name))), self.window_s, max_windows)

    def windows(self, name: str, keep_every: int = 1,
                report: Optional[CleaningReport] = None) -> Iterator[TrafficWindow]:
        """TrafficWindows of one capture. keep_every > 1 is for TRAINING only."""
        capture_id = self.capture_id(name)
        for raw in raw_windows(self.frames(name, report), self.window_us, keep_every, capture_id):
            yield to_traffic_window(raw, capture_id, self.vehicle_id)

    def labelled_windows(self, name: str, keep_every: int = 1,
                         report: Optional[CleaningReport] = None
                         ) -> Iterator[Tuple[TrafficWindow, GroundTruthLabel]]:
        """(window, private label) pairs. The label never goes to the Defender."""
        entry = self.manifest.entry(name)
        rule = injection_rule(entry, self.road_metadata)
        for raw in raw_windows(self.frames(name, report), self.window_us, keep_every,
                               entry.capture_id):
            index, start_us, end_us, frames = raw
            label = window_label(rule, window_id_for(entry.capture_id, index),
                                 start_us, end_us, frames)
            yield to_traffic_window(raw, entry.capture_id, self.vehicle_id), label

    # ---------- metadata ----------------------------------------------------
    def metadata(self, name: str) -> CaptureMetadata:
        entry = self.manifest.entry(name)
        rule = injection_rule(entry, self.road_metadata)
        path = str(self.path(name))
        report = CleaningReport()
        next(iter_raw(path, report), None)            # reads only the first line
        return CaptureMetadata(
            name=name, capture_id=entry.capture_id, vehicle_id=self.vehicle_id,
            group=entry.group, kind=entry.kind, family=entry.family, target=rule.target,
            injection_interval=rule.interval_s,
            duration_s=capture_span_us(path) / 1_000_000,
            first_raw_timestamp=report.first_raw_timestamp or "")


def preprocess(window: TrafficWindow, decoder: Optional[Decoder] = None) -> TrafficWindow:
    """Fill the optional parsed fields of a window, AFTER any injection.

    Returns a COPY; the input window is never modified (the plan keeps the
    original window for all checks). With no decoder the copy is unchanged.
    """
    data = window.model_dump()
    if decoder is not None:
        data["decoded_signals"] = decoder.decode_frames(window.frames)
    return TrafficWindow(**data)
