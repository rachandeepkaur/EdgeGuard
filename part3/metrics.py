"""Window-level evaluation metrics for EdgeGuard Part 3.

Call only after the evaluator joins Defender outputs with hidden labels.
Overlapping windows are correlated; report capture-level results separately.
"""

from dataclasses import dataclass
from statistics import mean
from typing import Iterable, Optional

from part3.evaluator import EvaluationResult


def _divide(numerator: int, denominator: int) -> Optional[float]:
    """Return None when a metric cannot be calculated."""
    return numerator / denominator if denominator else None


@dataclass(frozen=True)
class MetricsReport:
    windows: int
    tp: int
    fp: int
    tn: int
    fn: int
    recall: Optional[float]
    false_alarm_rate: Optional[float]
    precision: Optional[float]
    f1: Optional[float]
    mean_inference_ms: Optional[float]


def compute_metrics(results: Iterable[EvaluationResult]) -> MetricsReport:
    """Calculate rates from window outcomes without silently inventing zeros.

    false_alarm_rate = FP / (FP + TN), rather than FP / all windows.
    mean_inference_ms is Defender inference time; it excludes the time spent
    collecting a traffic window and cannot be called detection latency.
    """
    counts = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    latencies: list[float] = []
    seen_ids: set[str] = set()

    for result in results:
        if result.window_id in seen_ids:
            raise ValueError(f"duplicate evaluation result: {result.window_id}")
        seen_ids.add(result.window_id)
        if result.outcome not in counts:
            raise ValueError(f"unknown evaluation outcome: {result.outcome!r}")
        counts[result.outcome] += 1
        if result.latency_ms is not None:
            if result.latency_ms < 0:
                raise ValueError("inference latency must be nonnegative")
            latencies.append(result.latency_ms)

    tp, fp, tn, fn = (counts[key] for key in ("TP", "FP", "TN", "FN"))
    precision = _divide(tp, tp + fp)
    recall = _divide(tp, tp + fn)
    f1 = _divide(2 * tp, 2 * tp + fp + fn)

    return MetricsReport(
        windows=tp + fp + tn + fn,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        recall=recall,
        false_alarm_rate=_divide(fp, fp + tn),
        precision=precision,
        f1=f1,
        mean_inference_ms=mean(latencies) if latencies else None,
    )
