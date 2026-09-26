"""Part 3: apply an attack to a COPY of a raw traffic window.

Call Part 1's preprocess() on the returned window before Defender inference.
Ground-truth labels and attack specifications must remain outside that window.

inject(window, spec) dispatches on spec.family; inject_freeze() and
inject_offset() are the two concrete families (build plan S5).
"""

from shared.schemas import TrafficWindow
from part3.attack_spec import AttackSpec


def _check_common(window: TrafficWindow, variant_index: int) -> None:
    if isinstance(variant_index, bool) or not isinstance(variant_index, int) or variant_index < 1:
        raise ValueError("variant_index must be a positive integer")
    if window.features or window.decoded_signals:
        raise ValueError("inject into a raw window before preprocessing")


def _interval(window: TrafficWindow, spec: AttackSpec) -> tuple[float, float]:
    start = window.window_start + spec.start_offset_ms / 1000.0
    end = start + spec.duration_ms / 1000.0
    if start >= window.window_end or end <= window.window_start:
        raise ValueError("injection interval does not overlap the window")
    return start, end


def inject(window: TrafficWindow, spec: AttackSpec, *, variant_index: int = 1) -> TrafficWindow:
    """Apply spec to a copy of window, dispatching on spec.family."""
    if spec.family == "freeze":
        return inject_freeze(window, spec, variant_index=variant_index)
    if spec.family == "offset":
        return inject_offset(window, spec, variant_index=variant_index)
    raise ValueError(f"unsupported attack family: {spec.family}")


def inject_freeze(
    window: TrafficWindow, spec: AttackSpec, *, variant_index: int = 1
) -> TrafficWindow:
    """Repeat the first matching payload for one CAN ID within a time interval.

    start_offset_ms is relative to this window's start, not the capture's
    start or the ROAD metadata injection_interval. Timestamps and ordering
    are preserved. Raise ValueError if the attack changes no frame.
    """
    if spec.family != "freeze":
        raise ValueError(f"unsupported attack family: {spec.family}")
    _check_common(window, variant_index)
    start, end = _interval(window, spec)

    attacked = window.model_copy(deep=True)
    frozen_payload = None
    changed = 0
    for frame in attacked.frames:
        if start <= frame.timestamp < end and frame.can_id.upper() == spec.target_can_id.upper():
            if frozen_payload is None:
                frozen_payload = frame.payload
            elif frame.payload != frozen_payload:
                frame.payload = frozen_payload
                changed += 1

    if changed == 0:
        raise ValueError("freeze changed no frames; choose another window, ID, or interval")

    attacked.window_id = f"{window.window_id}_v{variant_index:02d}"
    return attacked


def inject_offset(
    window: TrafficWindow, spec: AttackSpec, *, variant_index: int = 1
) -> TrafficWindow:
    """Add a bounded signed offset to one payload byte of one CAN ID.

    Spoofs a sensor value pushed toward an extreme (e.g. ROAD's
    max_speedometer_attack), rather than freeze's "stop a changing value".
    The byte at spec.byte_index is read as an unsigned 0-255 integer,
    spec.offset is added, and the result is clamped back into 0-255. A
    frame whose payload is too short to hold byte_index is left unchanged
    (matches part1.decode's "too short" handling). start_offset_ms is
    relative to this window's start. Raise ValueError if no frame changes.
    """
    if spec.family != "offset":
        raise ValueError(f"unsupported attack family: {spec.family}")
    if spec.byte_index is None or spec.offset is None:
        raise ValueError("an 'offset' attack needs byte_index and offset")
    _check_common(window, variant_index)
    start, end = _interval(window, spec)

    attacked = window.model_copy(deep=True)
    changed = 0
    for frame in attacked.frames:
        if not (start <= frame.timestamp < end
               and frame.can_id.upper() == spec.target_can_id.upper()):
            continue
        raw = bytearray.fromhex(frame.payload)
        if spec.byte_index >= len(raw):
            continue    # payload too short to hold this byte; leave it alone
        original = raw[spec.byte_index]
        clamped = min(255, max(0, original + spec.offset))
        if clamped != original:
            raw[spec.byte_index] = clamped
            frame.payload = raw.hex().upper()
            changed += 1

    if changed == 0:
        raise ValueError(
            "offset changed no frames; choose another window, ID, byte_index, offset, "
            "or interval"
        )

    attacked.window_id = f"{window.window_id}_v{variant_index:02d}"
    return attacked
