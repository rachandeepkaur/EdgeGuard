"""
Part 1: is the escalation band sending the RIGHT windows to the cloud?

part1.egress_metrics reports how often the simulated cloud disagreed with
the local decision ("correction rate"). That number says nothing about
whether escalation targets the local model's real mistakes -- the
simulated cloud is a stand-in with its own fixed threshold, so its
disagreement rate mostly reflects how far that threshold sits from the
model's own. This file measures what the HP brief actually asks for (an
"explicit, defensible, measurable" escalation decision) directly, on
windows whose true label is known, with no cloud involved at all:

    escalation precision   of the windows escalated, how many did the
                           local model actually get wrong?
    escalation recall      of the local model's mistakes, how many were
                           escalated, so a second opinion could catch them?
    error rate outside the band
                           compared with precision (the error rate INSIDE
                           the band): is the band really where the
                           uncertainty is?
    local accuracy, and a ceiling for local + cloud
                           the ceiling assumes a PERFECT second opinion on
                           every escalated window -- the most escalation
                           could ever add, not a claim about any real cloud.
    bundle bytes           size of the sanitized bundle that would leave the
                           device per escalation. Raw CAN bytes sent is zero
                           by construction: bundle_fields lists everything
                           SanitizedBundle's allow-list permits, and none of
                           it is a frame or payload.

Label source: the Red Team test path (integration.run_demo.run_test) --
we made those attacks, so the label is certain. The band must be
calibrated on VALIDATION normal scores (integration.run_demo
.calibrate_escalation_band), never on the windows scored here, and never
on final_test.
"""

from dataclasses import dataclass
from statistics import fmean
from typing import Iterable, List, Optional

from part1.escalation_policy import should_escalate
from part1.sanitizer import SanitizedBundle, sanitize
from part3.evaluator import evaluate_batch
from shared.schemas import DefenderOutput, GroundTruthLabel


@dataclass(frozen=True)
class EscalationQuality:
    windows: int
    band_half_width: float
    escalations: int
    escalation_rate: float
    local_errors: int
    local_false_positives: int
    local_false_negatives: int
    escalated_errors: int
    escalated_false_positives: int
    escalated_false_negatives: int
    escalation_precision: Optional[float]       # escalated_errors / escalations (= error rate inside the band)
    escalation_recall: Optional[float]          # escalated_errors / local_errors
    error_rate_outside_band: Optional[float]    # compare with escalation_precision
    local_accuracy: float
    local_plus_perfect_cloud_accuracy: float    # ceiling, not a measured cloud result
    bundle_bytes_mean: Optional[float]
    bundle_bytes_max: Optional[int]
    bundle_bytes_total: int
    bundle_fields: List[str]


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    return numerator / denominator if denominator else None


def compute_escalation_quality(
    outputs: Iterable[DefenderOutput],
    labels: Iterable[GroundTruthLabel],
    band_half_width: float,
) -> EscalationQuality:
    """Score the escalation band against known labels.

    The join (and its missing/duplicate checks) is part3.evaluator
    .evaluate_batch's, so this can never disagree with the detection
    metrics about which windows were right or wrong.
    """
    outputs = list(outputs)
    if not outputs:
        raise ValueError("no Defender outputs: nothing to measure")
    results = evaluate_batch(outputs, labels)
    output_by_id = {output.window_id: output for output in outputs}

    escalations = 0
    inside_errors = inside_fp = inside_fn = 0
    outside_windows = outside_errors = 0
    local_fp = local_fn = 0
    bundle_sizes: List[int] = []

    for result in results:
        output = output_by_id[result.window_id]
        wrong = result.outcome in ("FP", "FN")
        if result.outcome == "FP":
            local_fp += 1
        elif result.outcome == "FN":
            local_fn += 1

        if should_escalate(output, band_half_width).escalate:
            escalations += 1
            bundle_sizes.append(len(sanitize(output).model_dump_json().encode("utf-8")))
            if wrong:
                inside_errors += 1
                if result.outcome == "FP":
                    inside_fp += 1
                else:
                    inside_fn += 1
        else:
            outside_windows += 1
            if wrong:
                outside_errors += 1

    windows = len(results)
    local_errors = local_fp + local_fn
    return EscalationQuality(
        windows=windows,
        band_half_width=band_half_width,
        escalations=escalations,
        escalation_rate=escalations / windows,
        local_errors=local_errors,
        local_false_positives=local_fp,
        local_false_negatives=local_fn,
        escalated_errors=inside_errors,
        escalated_false_positives=inside_fp,
        escalated_false_negatives=inside_fn,
        escalation_precision=_ratio(inside_errors, escalations),
        escalation_recall=_ratio(inside_errors, local_errors),
        error_rate_outside_band=_ratio(outside_errors, outside_windows),
        local_accuracy=(windows - local_errors) / windows,
        local_plus_perfect_cloud_accuracy=(windows - outside_errors) / windows,
        bundle_bytes_mean=fmean(bundle_sizes) if bundle_sizes else None,
        bundle_bytes_max=max(bundle_sizes) if bundle_sizes else None,
        bundle_bytes_total=sum(bundle_sizes),
        bundle_fields=sorted(SanitizedBundle.model_fields),
    )
