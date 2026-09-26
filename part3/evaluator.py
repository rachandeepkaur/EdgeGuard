"""Part 3 evaluation: join Defender outputs to hidden window labels.

Only the evaluator receives GroundTruthLabel. The Defender never does.
"""

from dataclasses import dataclass
from typing import Iterable, Literal, Optional

from shared.schemas import DefenderOutput, GroundTruthLabel

Outcome = Literal["TP", "FP", "TN", "FN"]


@dataclass(frozen=True)
class EvaluationResult:
    window_id: str
    outcome: Outcome
    attack_score: float
    model_version: str
    family: Optional[str]
    latency_ms: Optional[float]


def evaluate_one(output: DefenderOutput, label: GroundTruthLabel) -> EvaluationResult:
    """Evaluate one decision; never alter the Defender output or label."""
    if output.window_id != label.window_id:
        raise ValueError("Defender output and label have different window_id values")

    if label.is_attack:
        outcome: Outcome = "TP" if output.decision == "ATTACK" else "FN"
    else:
        outcome = "FP" if output.decision == "ATTACK" else "TN"

    return EvaluationResult(
        window_id=output.window_id,
        outcome=outcome,
        attack_score=output.attack_score,
        model_version=output.model_version,
        family=label.family,
        latency_ms=output.latency_ms,
    )


def evaluate_batch(
    outputs: Iterable[DefenderOutput], labels: Iterable[GroundTruthLabel]
) -> list[EvaluationResult]:
    """Join by window_id and reject missing or duplicate records.

    Pass only records from the split you are evaluating; the split manifest
    and final-test access rules are managed separately by the team.
    """
    output_by_id = {}
    for output in outputs:
        if output.window_id in output_by_id:
            raise ValueError(f"duplicate Defender output: {output.window_id}")
        output_by_id[output.window_id] = output

    label_by_id = {}
    for label in labels:
        if label.window_id in label_by_id:
            raise ValueError(f"duplicate ground-truth label: {label.window_id}")
        label_by_id[label.window_id] = label

    missing_labels = output_by_id.keys() - label_by_id.keys()
    missing_outputs = label_by_id.keys() - output_by_id.keys()
    if missing_labels or missing_outputs:
        raise ValueError(
            "unmatched window IDs: "
            f"outputs without labels={sorted(missing_labels)}, "
            f"labels without outputs={sorted(missing_outputs)}"
        )

    return [evaluate_one(output, label_by_id[window_id]) for window_id, output in output_by_id.items()]
