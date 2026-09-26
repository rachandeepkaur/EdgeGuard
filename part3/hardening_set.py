"""Choose confirmed development misses to hand to Part 2 for retraining.

This is a manifest of window IDs, not a copy of CAN traffic. Part 2 owns
retraining and must load the matching development windows through Part 1's
data pipeline, then evaluate the updated model on held-out data.
"""

from dataclasses import dataclass
from typing import Iterable

from part3.evasion_log import EvasionEntry


@dataclass(frozen=True)
class HardeningSet:
    baseline_model_version: str
    window_ids: tuple[str, ...]
    source_split: str = "development"


def build_hardening_set(
    evasions: Iterable[EvasionEntry],
    development_window_ids: Iterable[str],
    *,
    baseline_model_version: str,
) -> HardeningSet:
    """Select distinct missed windows from one baseline model.

    The development_window_ids allow-list must come from Part 1's split
    manifest. We reject missing IDs so a typo cannot silently select the
    wrong data. No final-test windows may appear in this allow-list.
    """
    if not baseline_model_version:
        raise ValueError("baseline_model_version must not be empty")

    allowed = set(development_window_ids)
    selected: set[str] = set()
    for entry in evasions:
        if entry.split != "development":
            raise ValueError(f"evasion {entry.window_id} is not from development")
        if entry.model_version != baseline_model_version:
            raise ValueError(f"evasion {entry.window_id} has a different model version")
        if entry.window_id not in allowed:
            raise ValueError(f"evasion {entry.window_id} is not in the development manifest")
        selected.add(entry.window_id)

    return HardeningSet(
        baseline_model_version=baseline_model_version,
        window_ids=tuple(sorted(selected)),
    )
