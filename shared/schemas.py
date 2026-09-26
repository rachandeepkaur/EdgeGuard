"""Shared data contracts for EdgeGuard.

This is the one shared schema file for all three parts. Change field names
only after the whole team agrees.

Part 1 creates a TrafficWindow with empty features and decoded signals.
Part 3 injects an attack into a copy of the raw window. Part 1 preprocesses
the modified window; Part 2 receives it and returns DefenderOutput. Part 3
joins that result with GroundTruthLabel by window_id.

Attack specifications, cloud messages, and evaluation results are owned by
their respective parts and are not defined here.
"""

import math
import re
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_HEX = re.compile(r"^[0-9A-Fa-f]*$")
_FORBIDDEN_ID_WORDS = ("ambient", "attack")


class StrictModel(BaseModel):
    """Reject unknown fields and validate individual fields on assignment."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


def _check_neutral_id(value: str, field_name: str) -> str:
    lowered = value.lower()
    for word in _FORBIDDEN_ID_WORDS:
        if word in lowered:
            raise ValueError(
                f"{field_name} {value!r} contains {word!r}; identifiers must be "
                "neutral (e.g. 'cap07') so they do not reveal the label"
            )
    return value


class Frame(StrictModel):
    timestamp: float
    can_id: str = Field(min_length=1)  # hex, e.g. "0F4"
    payload: str  # hex, e.g. "960C010204B10240"

    @field_validator("can_id")
    @classmethod
    def can_id_is_hex(cls, value: str) -> str:
        if not _HEX.match(value):
            raise ValueError(f"can_id {value!r} is not hexadecimal (expected e.g. '0F4')")
        return value

    @field_validator("payload")
    @classmethod
    def payload_is_hex_bytes(cls, value: str) -> str:
        if not _HEX.match(value):
            raise ValueError(f"payload {value!r} contains non-hex characters")
        if len(value) % 2 != 0:
            raise ValueError(
                f"payload {value!r} has an odd number of hex digits; "
                "each byte needs exactly 2"
            )
        if len(value) > 16:
            raise ValueError(
                f"payload {value!r} is {len(value) // 2} bytes; a CAN payload "
                "has at most 8 bytes (16 hex digits)"
            )
        return value


# The Defender receives this object only; labels and attack specs stay separate.
class TrafficWindow(StrictModel):
    window_id: str = Field(min_length=1)  # e.g. "cap07_w0153"
    capture_id: str = Field(min_length=1)  # e.g. "cap07"
    window_start: float    # elapsed seconds from the start of the capture
    window_end: float
    frames: List[Frame]
    # Build plan S3: every window carries its vehicle. Part 1 always sets it;
    # optional only so windows built by hand in tests stay valid.
    vehicle_id: Optional[str] = Field(default=None, min_length=1)  # e.g. "veh01"
    # Part 1 fills these after any injection (part1.pipeline.preprocess).
    # decoded_signals: "<CAN ID>:<signal name>" -> one value per frame of that
    # ID, in frame order. NaN = the payload is too short to hold the signal.
    features: Dict[str, float] = Field(default_factory=dict)
    decoded_signals: Dict[str, List[float]] = Field(default_factory=dict)

    @field_validator("window_id", "capture_id", "vehicle_id")
    @classmethod
    def ids_are_neutral(cls, value: Optional[str], info) -> Optional[str]:
        if value is None:
            return value
        return _check_neutral_id(value, info.field_name)

    @model_validator(mode="after")
    def check_window(self):
        if self.window_end < self.window_start:
            raise ValueError(
                f"window {self.window_id}: window_end ({self.window_end}) is "
                f"before window_start ({self.window_start})"
            )
        for i, frame in enumerate(self.frames):
            if not (self.window_start <= frame.timestamp <= self.window_end):
                raise ValueError(
                    f"window {self.window_id}: frame {i} timestamp "
                    f"{frame.timestamp} is outside the window "
                    f"[{self.window_start}, {self.window_end}]"
                )
        return self


Decision = Literal["ATTACK", "ACCEPT"]


class DefenderOutput(StrictModel):
    window_id: str = Field(min_length=1)
    # A score, not a calibrated probability unless calibration is shown.
    attack_score: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    decision: Decision
    evidence: str
    model_version: str = Field(min_length=1)
    # Inference time only; window collection time is separate.
    latency_ms: Optional[float] = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def check_decision_matches_score(self):
        expected = "ATTACK" if self.attack_score >= self.threshold else "ACCEPT"
        if self.decision != expected:
            raise ValueError(
                f"window {self.window_id}: decision is {self.decision} but "
                f"score {self.attack_score} vs threshold {self.threshold} "
                f"means it should be {expected}"
            )
        return self


# Evaluator only. ROAD injection intervals are elapsed seconds from the
# start of the FULL capture, never the raw log timestamps. Obtain the first
# timestamp of that capture from Part 1's metadata before evaluating overlap.
# The interval may be absent for attacks that began before recording started.
class GroundTruthLabel(StrictModel):
    window_id: str = Field(min_length=1)
    is_attack: bool
    family: Optional[str] = None
    target: Optional[str] = None
    injection_interval: Optional[List[float]] = None  # [start, end], elapsed seconds
    # Filled by Part 1 (part1.labels): how many frames of this window were
    # injected, and how many seconds of the window lie inside the injection
    # interval. Lets results separate windows only partly covered by an attack.
    injected_frames: int = Field(default=0, ge=0)
    interval_overlap_s: Optional[float] = Field(default=None, ge=0.0)

    @field_validator("injection_interval")
    @classmethod
    def interval_is_valid(cls, value):
        if value is None:
            return value
        if len(value) != 2:
            raise ValueError(
                f"injection_interval {value} must have exactly 2 numbers: [start, end]"
            )
        start, end = value
        if not (math.isfinite(start) and math.isfinite(end)):
            raise ValueError("injection_interval must contain finite numbers")
        if start < 0:
            raise ValueError(
                f"injection_interval start {start} is negative; it must be "
                "elapsed seconds from the capture start"
            )
        if end <= start:
            raise ValueError(f"injection_interval {value}: end must be > start")
        return value

    @model_validator(mode="after")
    def check_consistency(self):
        if self.is_attack and not self.family:
            raise ValueError(
                f"window {self.window_id}: is_attack is true, so family is required"
            )
        if not self.is_attack and (
            self.family or self.target or self.injection_interval is not None
            or self.injected_frames or self.interval_overlap_s is not None
        ):
            raise ValueError(
                f"window {self.window_id}: normal traffic (is_attack false) "
                "must not have family, target, injection_interval, "
                "injected_frames or interval_overlap_s"
            )
        return self