"""
Part 3: per-capture metrics -- part3/metrics.py's own docstring flags this
as needed ("Overlapping windows are correlated; report capture-level
results separately") and it was never built. Also closes the "highway
generalization" question directly: which final_test captures actually
produce v2's false alarms, rather than inferring it from a cross-validation
comparison alone.

Measurement only: groups the SAME evaluate_batch() results already computed
for the aggregate final_test report by capture_id, via a lightweight
window_id -> capture_id map (see integration/run_final_evaluation.py's
_WindowTiming -- never the full TrafficWindow, for the same memory reason
as part3/detection_delay.py).
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List

from part3.evaluator import EvaluationResult, evaluate_batch
from part3.metrics import MetricsReport, compute_metrics
from shared.schemas import DefenderOutput, GroundTruthLabel


@dataclass(frozen=True)
class CaptureMetrics:
    capture_id: str
    metrics: MetricsReport


def compute_per_capture_metrics(
    outputs: Iterable[DefenderOutput],
    labels: Iterable[GroundTruthLabel],
    window_id_to_capture_id: Dict[str, str],
) -> List[CaptureMetrics]:
    results: List[EvaluationResult] = evaluate_batch(outputs, labels)
    missing = [r.window_id for r in results if r.window_id not in window_id_to_capture_id]
    if missing:
        raise ValueError(
            f"{len(missing)} window(s) have no capture_id mapping, e.g. {missing[:3]}"
        )

    by_capture: Dict[str, List[EvaluationResult]] = {}
    for result in results:
        capture_id = window_id_to_capture_id[result.window_id]
        by_capture.setdefault(capture_id, []).append(result)

    return [
        CaptureMetrics(capture_id=capture_id, metrics=compute_metrics(capture_results))
        for capture_id, capture_results in sorted(by_capture.items())
    ]
