"""Part 3 Red Team: propose attack specifications, never edit traffic.

This first version is deterministic and uses development windows only.
It does not query the Defender or read hidden ground-truth labels. Later,
feedback-based proposals must use development-set scores, never final-test
results or private Defender evidence.

Two families (build plan S5): propose_freeze() stops a CAN ID's payload
from changing; propose_offset() pushes one stable payload byte toward an
extreme, like ROAD's max_speedometer_attack. propose() dispatches by name.
"""

from collections import defaultdict
from typing import Dict, List, Tuple

from part3.attack_spec import AttackFamily, AttackSpec
from shared.schemas import TrafficWindow

OFFSET_MAGNITUDE = 120  # bounded push toward whichever extreme has more room


def propose(window: TrafficWindow, family: AttackFamily, *,
           attack_id: str = "atk_001") -> AttackSpec:
    """Propose an attack of the given family. Dispatches to propose_freeze /
    propose_offset."""
    if family == "freeze":
        return propose_freeze(window, attack_id=attack_id)
    if family == "offset":
        return propose_offset(window, attack_id=attack_id)
    raise ValueError(f"unsupported attack family: {family}")


def propose_freeze(window: TrafficWindow, *, attack_id: str = "atk_001") -> AttackSpec:
    """Pick a CAN ID that varies in this window, then propose a full-window freeze.

    The first matching payload becomes the frozen value; later distinct
    payloads are the ones the injector can change. Choose the ID with the
    most changes; resolve ties by hexadecimal ID for reproducibility.
    """
    if window.features or window.decoded_signals:
        raise ValueError("propose an attack from a raw window before preprocessing")
    duration_ms = (window.window_end - window.window_start) * 1000.0
    if duration_ms <= 0:
        raise ValueError("window must have positive duration")

    payloads_by_id: Dict[str, List[str]] = defaultdict(list)
    for frame in window.frames:
        payloads_by_id[frame.can_id.upper()].append(frame.payload)

    candidates = [
        (sum(payload != values[0] for payload in values[1:]), can_id)
        for can_id, values in payloads_by_id.items()
        if len(values) >= 2
    ]
    candidates = [(changes, can_id) for changes, can_id in candidates if changes > 0]
    if not candidates:
        raise ValueError("no CAN ID has changing payloads in this window")

    _, target_id = min(candidates, key=lambda pair: (-pair[0], int(pair[1], 16)))
    return AttackSpec(
        attack_id=attack_id,
        family="freeze",
        target_can_id=target_id,
        start_offset_ms=0.0,
        duration_ms=duration_ms,
        reasoning="Freeze a changing CAN payload to test payload-aware detection.",
    )


def propose_offset(window: TrafficWindow, *, attack_id: str = "atk_001") -> AttackSpec:
    """Pick the most STABLE payload byte in this window, then propose pushing
    it toward whichever extreme (0x00 or 0xFF) has more headroom -- the
    ROAD-style "spoof a sensor reading" attack (e.g. max_speedometer),
    as opposed to propose_freeze's "stop a changing value".

    Ties broken by (CAN ID, byte index) for reproducibility.
    """
    if window.features or window.decoded_signals:
        raise ValueError("propose an attack from a raw window before preprocessing")
    duration_ms = (window.window_end - window.window_start) * 1000.0
    if duration_ms <= 0:
        raise ValueError("window must have positive duration")

    bytes_by_field: Dict[Tuple[str, int], List[int]] = defaultdict(list)
    for frame in window.frames:
        can_id = frame.can_id.upper()
        raw = bytes.fromhex(frame.payload)
        for index, value in enumerate(raw):
            bytes_by_field[(can_id, index)].append(value)

    if not bytes_by_field:
        raise ValueError("window has no payload bytes to target")

    candidates = [
        (max(values) - min(values), can_id, index, values)
        for (can_id, index), values in bytes_by_field.items()
    ]
    spread, target_id, byte_index, values = min(
        candidates, key=lambda item: (item[0], int(item[1], 16), item[2])
    )
    mean_value = sum(values) / len(values)
    offset = OFFSET_MAGNITUDE if mean_value <= 127 else -OFFSET_MAGNITUDE

    return AttackSpec(
        attack_id=attack_id,
        family="offset",
        target_can_id=target_id,
        start_offset_ms=0.0,
        duration_ms=duration_ms,
        byte_index=byte_index,
        offset=offset,
        reasoning=(
            f"Push the most stable byte (byte {byte_index} of {target_id}, range "
            f"{spread} over {len(values)} frames) toward an extreme, like a "
            "spoofed sensor reading."
        ),
    )
