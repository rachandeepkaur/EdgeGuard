"""Record development-set attacks the Defender missed.

The log stores only attack specifications and scores, not raw CAN frames.
Final-test results must never be used to select hardening examples.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Union

from part3.attack_spec import AttackSpec
from part3.evaluator import EvaluationResult


@dataclass(frozen=True)
class EvasionEntry:
    attack_id: str
    window_id: str
    family: str
    target_can_id: str
    start_offset_ms: float
    duration_ms: float
    attack_score: float
    model_version: str
    split: str


def make_evasion_entry(
    spec: AttackSpec, result: EvaluationResult, *, split: str
) -> EvasionEntry:
    """Accept only confirmed false negatives from development data."""
    if split != "development":
        raise ValueError("only development-set misses may enter the evasion log")
    if result.outcome != "FN":
        raise ValueError("an evasion must be a confirmed false negative (FN)")
    if result.family != spec.family:
        raise ValueError("attack specification family does not match the hidden label")
    if not result.window_id:
        raise ValueError("evasion needs a window_id")

    return EvasionEntry(
        attack_id=spec.attack_id,
        window_id=result.window_id,
        family=spec.family,
        target_can_id=spec.target_can_id.upper(),
        start_offset_ms=spec.start_offset_ms,
        duration_ms=spec.duration_ms,
        attack_score=result.attack_score,
        model_version=result.model_version,
        split=split,
    )


def append_evasion(path: Union[str, Path], entry: EvasionEntry) -> None:
    """Append one JSON record; keep raw frames and labels out of this file."""
    if entry.split != "development":
        raise ValueError("only development-set misses may enter the evasion log")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(asdict(entry), sort_keys=True) + "\n")
