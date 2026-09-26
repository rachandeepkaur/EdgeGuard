"""
Part 1: escalation-path metrics for build plan section 7.

    "Record escalation rate, added response latency and whether the
    second opinion corrects any errors."

This is separate from part3/metrics.py, which scores the Defender's own
accuracy against hidden labels. This file scores the cloud-escalation
PATH itself: how often it's used, what it costs in latency, and whether
it actually catches anything the local decision got wrong.
"""

from dataclasses import dataclass
from statistics import mean
from typing import Iterable, Optional


@dataclass(frozen=True)
class EscalationOutcome:
    """One escalation-path record for one window.

    escalated=False means this window's score was outside the
    uncertainty band; added_latency_ms and corrected must both be None
    in that case, since no call was made.

    corrected=True means the simulated cloud's advice disagreed with the
    local decision (see mock_cloud_endpoint.advise: "override_to_attack"
    or "override_to_accept"). corrected=None means the call failed or
    timed out (see request_second_opinion returning advice=None) -- a
    failure is NOT the same as "confirmed", so it must not silently
    count as a non-correction.
    """
    window_id: str
    escalated: bool
    added_latency_ms: Optional[float] = None
    corrected: Optional[bool] = None

    def __post_init__(self):
        if not self.escalated and (self.added_latency_ms is not None or self.corrected is not None):
            raise ValueError(
                f"{self.window_id}: escalated=False must not carry "
                f"added_latency_ms or corrected (no call was made)"
            )
        if self.escalated and self.added_latency_ms is None:
            raise ValueError(
                f"{self.window_id}: escalated=True must record added_latency_ms, "
                f"even for a failed or timed-out call"
            )
        if self.added_latency_ms is not None and self.added_latency_ms < 0:
            raise ValueError(f"{self.window_id}: added_latency_ms must be nonnegative")


@dataclass(frozen=True)
class EgressReport:
    windows: int
    escalations: int
    escalation_rate: float
    failed_calls: int
    mean_added_latency_ms: Optional[float]
    p95_added_latency_ms: Optional[float]
    corrections: int
    correction_rate: Optional[float]   # corrections / (escalations - failed_calls)


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        raise ValueError("cannot take a percentile of an empty list")
    index = max(0, min(len(sorted_values) - 1, round(p * (len(sorted_values) - 1))))
    return sorted_values[index]


def compute_egress_report(outcomes: Iterable[EscalationOutcome]) -> EgressReport:
    """Summarize the escalation path. corrections and correction_rate only
    count calls that actually got a response: a failed/timed-out call
    (corrected=None) adds to failed_calls, not to either side of the
    correction count, since a failure tells you nothing about whether the
    local decision was right."""
    windows = 0
    escalations = 0
    failed_calls = 0
    corrections = 0
    answered_calls = 0
    latencies: list[float] = []

    for outcome in outcomes:
        windows += 1
        if not outcome.escalated:
            continue
        escalations += 1
        latencies.append(outcome.added_latency_ms)
        if outcome.corrected is None:
            failed_calls += 1
        else:
            answered_calls += 1
            if outcome.corrected:
                corrections += 1

    sorted_latencies = sorted(latencies)
    return EgressReport(
        windows=windows,
        escalations=escalations,
        escalation_rate=(escalations / windows) if windows else 0.0,
        failed_calls=failed_calls,
        mean_added_latency_ms=mean(latencies) if latencies else None,
        p95_added_latency_ms=_percentile(sorted_latencies, 0.95) if sorted_latencies else None,
        corrections=corrections,
        correction_rate=(corrections / answered_calls) if answered_calls else None,
    )
