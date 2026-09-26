"""Check Part 3 attack proposals before modifying any traffic.

This is a simulation guard, not a proof that an attack is realistic or safe.
The actual injector still rejects a proposal if no bytes change.
"""

from dataclasses import dataclass
from typing import Iterable, Optional

from part3.attack_spec import AttackSpec
from shared.schemas import TrafficWindow


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    reason: str
    matching_frames: int
    changes_expected: int


def validate_attack(
    window: TrafficWindow,
    spec: AttackSpec,
    *,
    allowed_can_ids: Optional[Iterable[str]] = None,
) -> ValidationResult:
    """Check that a freeze or offset proposal can change this raw window.

    Time offsets are milliseconds from the beginning of this window.
    If Part 1 supplies a watch-list, use allowed_can_ids to constrain
    proposals to those IDs. No ground-truth label is read here.
    """
    if window.features or window.decoded_signals:
        return ValidationResult(False, "input must be a raw window", 0, 0)
    if spec.family not in ("freeze", "offset"):
        return ValidationResult(False, "unsupported attack family", 0, 0)

    target = spec.target_can_id.upper()
    if allowed_can_ids is not None:
        allowed = {can_id.upper() for can_id in allowed_can_ids}
        if target not in allowed:
            return ValidationResult(False, "CAN ID is not in the allowed list", 0, 0)

    start = window.window_start + spec.start_offset_ms / 1000.0
    end = start + spec.duration_ms / 1000.0
    if start >= window.window_end or end <= window.window_start:
        return ValidationResult(False, "attack interval does not overlap window", 0, 0)

    matching = [
        frame for frame in window.frames
        if start <= frame.timestamp < end and frame.can_id.upper() == target
    ]
    if not matching:
        return ValidationResult(False, "no frames with target CAN ID in interval", 0, 0)

    if spec.family == "freeze":
        frozen_payload = matching[0].payload
        changed = sum(frame.payload != frozen_payload for frame in matching[1:])
        if changed == 0:
            return ValidationResult(False, "freeze would change no payloads", len(matching), 0)
        return ValidationResult(True, "proposal changes traffic", len(matching), changed)

    # family == "offset"
    changed = 0
    for frame in matching:
        raw = bytearray.fromhex(frame.payload)
        if spec.byte_index >= len(raw):
            continue
        clamped = min(255, max(0, raw[spec.byte_index] + spec.offset))
        if clamped != raw[spec.byte_index]:
            changed += 1
    if changed == 0:
        return ValidationResult(
            False, "offset would change no payloads (too short, or already clamped)",
            len(matching), 0,
        )
    return ValidationResult(True, "proposal changes traffic", len(matching), changed)
