"""
Part 1 -- capture metadata and PRIVATE labels (build plan S3, Block 1).

Labels belong to the evaluator record and never reach the Defender: they are
produced here, next to (not inside) the TrafficWindow.

FRAME RULES, by capture kind (verified against ORNL's own post-processing by
the preprocessing study in ~/Downloads/road, docs/01 P2: the fabrication
count equals its masquerade twin's count for 13 of 13 pairs)
    masquerade   every frame of the target ID inside the injection interval
    fabrication  same, AND the payload matches injection_data_str, where X is
                 a wildcard nibble (the real frames of that ID keep flowing
                 in fabrication captures; only the matching ones are injected)
    fuzzing      payload FFFFFFFFFFFFFFFF, over the WHOLE capture (1-3 injected
                 frames per capture fall outside the declared interval)
    accelerator  no injected frames exist (label 0; reported separately)
    ambient      no injected frames

The interval is CLOSED and compared in integer microseconds: its endpoints
are the first and last injected frames. Comparing floats drops them.

WINDOW LABEL: a window is an attack if it holds at least one injected frame.
Each attack label also records injected_frames and interval_overlap_s, so
results can separate windows only partly covered by an attack.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from shared.schemas import GroundTruthLabel

from part1.fleet_simulator import RawFrame, canonical_can_id
from part1.split_manifest import CaptureEntry

FUZZ_PAYLOAD = "FFFFFFFFFFFFFFFF"


@dataclass(frozen=True)
class InjectionRule:
    kind: str                                   # manifest kind
    family: Optional[str]
    target: Optional[str]                       # canonical CAN ID, e.g. "0D0"
    interval_us: Optional[Tuple[int, int]]      # closed, elapsed microseconds
    mask: Optional[str]                         # injection_data_str, X = wildcard

    @property
    def interval_s(self) -> Optional[List[float]]:
        if self.interval_us is None:
            return None
        return [self.interval_us[0] / 1_000_000, self.interval_us[1] / 1_000_000]


def injection_rule(entry: CaptureEntry, road_metadata: dict) -> InjectionRule:
    """The labelling rule of one capture, from the manifest + ROAD metadata."""
    meta = road_metadata.get(entry.name, {})
    interval = meta.get("injection_interval")
    interval_us = (round(interval[0] * 1_000_000), round(interval[1] * 1_000_000)) \
        if interval else None
    target = meta.get("injection_id")
    if entry.kind in ("fabrication", "masquerade"):
        if target is None or interval_us is None:
            raise ValueError(f"{entry.name}: {entry.kind} capture needs injection_id and "
                             f"injection_interval in capture_metadata.json")
        target = canonical_can_id(target)
    else:
        target = None
    return InjectionRule(entry.kind, entry.family, target, interval_us,
                         meta.get("injection_data_str"))


def mask_match(payload: str, mask: str) -> bool:
    """Payload hex vs a mask like 'XXXXXXXXXXFFXXXX' (X = any nibble)."""
    if len(payload) != len(mask):
        return False
    return all(m == "X" or m == p for p, m in zip(payload.upper(), mask.upper()))


def is_injected(rule: InjectionRule, frame: RawFrame) -> bool:
    us, can_id, payload = frame
    if rule.kind == "fuzzing":
        return payload == FUZZ_PAYLOAD
    if rule.kind not in ("fabrication", "masquerade"):
        return False
    lo, hi = rule.interval_us
    if can_id != rule.target or not (lo <= us <= hi):
        return False
    return rule.kind == "masquerade" or mask_match(payload, rule.mask)


def window_label(rule: InjectionRule, window_id: str, start_us: int, end_us: int,
                 frames: Sequence[RawFrame]) -> GroundTruthLabel:
    """GroundTruthLabel of one window from its (cleaned) frames."""
    injected = sum(1 for f in frames if is_injected(rule, f))
    if injected == 0:
        return GroundTruthLabel(window_id=window_id, is_attack=False)
    overlap = None
    if rule.interval_us is not None:
        lo, hi = rule.interval_us
        overlap = max(0, min(end_us, hi) - max(start_us, lo)) / 1_000_000
    return GroundTruthLabel(window_id=window_id, is_attack=True, family=rule.family,
                            target=rule.target, injection_interval=rule.interval_s,
                            injected_frames=injected, interval_overlap_s=overlap)
