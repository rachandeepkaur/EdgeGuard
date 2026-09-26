"""Part 3: compare two frozen Defender versions on the same final-test set.

Run this only after model training, threshold selection, and Red Team
development are complete. Part 1 must provide the trusted final-test IDs;
this module cannot independently prove their provenance or enforce that a
human has not already examined the final test.
"""

from dataclasses import dataclass
from typing import Iterable, Tuple

from part3.evaluator import evaluate_batch
from part3.metrics import MetricsReport, compute_metrics
from shared.schemas import DefenderOutput, GroundTruthLabel


@dataclass(frozen=True)
class EvidenceReport:
    baseline_version: str
    updated_version: str
    baseline: MetricsReport
    updated: MetricsReport
    source_split: str = "final_test"


def _indexed_outputs(
    outputs: Iterable[DefenderOutput], *, name: str, expected_ids: set
) -> Tuple[list, str]:
    items = list(outputs)
    if not items:
        raise ValueError(f"{name} outputs are empty")
    ids = [item.window_id for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{name} has duplicate window IDs")
    if set(ids) != expected_ids:
        raise ValueError(f"{name} output IDs do not match final-test IDs")
    versions = {item.model_version for item in items}
    if len(versions) != 1:
        raise ValueError(f"{name} mixes model versions")
    return items, versions.pop()


def compare_final_models(
    baseline_outputs: Iterable[DefenderOutput],
    updated_outputs: Iterable[DefenderOutput],
    labels: Iterable[GroundTruthLabel],
    *,
    final_test_window_ids: Iterable[str],
    training_window_ids: Iterable[str],
) -> EvidenceReport:
    """Compute paired, window-level metrics; never use these results to tune.

    The two allow-lists must come from Part 1's capture-level split manifest.
    Pairing makes comparisons meaningful; no automatic 'improved' claim is
    made when recall and false-alarm rate move in different directions.
    """
    final_ids = list(final_test_window_ids)
    if not final_ids or len(final_ids) != len(set(final_ids)):
        raise ValueError("final-test window IDs must be nonempty and unique")
    final_set = set(final_ids)
    if final_set & set(training_window_ids):
        raise ValueError("training windows overlap final-test windows")

    baseline, baseline_version = _indexed_outputs(
        baseline_outputs, name="baseline", expected_ids=final_set
    )
    updated, updated_version = _indexed_outputs(
        updated_outputs, name="updated", expected_ids=final_set
    )
    if baseline_version == updated_version:
        raise ValueError("baseline and updated model versions must differ")

    hidden_labels = list(labels)
    label_ids = [item.window_id for item in hidden_labels]
    if len(label_ids) != len(set(label_ids)) or set(label_ids) != final_set:
        raise ValueError("labels must match final-test IDs exactly once")

    return EvidenceReport(
        baseline_version=baseline_version,
        updated_version=updated_version,
        baseline=compute_metrics(evaluate_batch(baseline, hidden_labels)),
        updated=compute_metrics(evaluate_batch(updated, hidden_labels)),
    )
