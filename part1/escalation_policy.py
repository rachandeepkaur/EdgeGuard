"""
Part 1: decide which Defender results are uncertain enough to escalate.

Build plan section 7: "Reserve an uncertainty interval around the defender
threshold using validation scores... Escalate only windows inside the
chosen interval... The immediate local decision remains available even
if the second service fails or is slow."

That last sentence is the most important rule in this file: escalation is
ALWAYS a side channel. output.decision is the real, immediate, local
answer, and nothing here ever blocks on or overrides it -- this module
only decides whether to ALSO ask for a second opinion.
"""

import math
from dataclasses import dataclass
from typing import List

from shared.schemas import DefenderOutput


def _check_unit_interval(value, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    if math.isnan(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")


@dataclass(frozen=True)
class EscalationDecision:
    window_id: str
    escalate: bool
    band_low: float
    band_high: float
    reason: str


def should_escalate(output: DefenderOutput, band_half_width: float) -> EscalationDecision:
    """Escalate only when attack_score falls in [threshold - band, threshold + band].

    A score confidently below or above the band is NOT escalated: the
    local decision (output.decision) is trusted on its own. The band is
    clipped to [0, 1] at the edges, same as threshold.py's ATTACK/ACCEPT
    rule, so a threshold near 0 or 1 does not produce an out-of-range band.
    """
    _check_unit_interval(output.attack_score, "output.attack_score")
    _check_unit_interval(output.threshold, "output.threshold")
    if isinstance(band_half_width, bool) or not isinstance(band_half_width, (int, float)):
        raise TypeError(f"band_half_width must be a number, got {type(band_half_width).__name__}")
    if band_half_width < 0.0 or band_half_width > 0.5:
        raise ValueError(f"band_half_width must be between 0 and 0.5, got {band_half_width}")

    low = max(0.0, output.threshold - band_half_width)
    high = min(1.0, output.threshold + band_half_width)
    uncertain = low <= output.attack_score <= high

    reason = (
        f"score {output.attack_score:.3f} is inside the uncertainty band "
        f"[{low:.3f}, {high:.3f}] around threshold {output.threshold:.3f}"
        if uncertain else
        f"score {output.attack_score:.3f} is outside the uncertainty band "
        f"[{low:.3f}, {high:.3f}]; local decision is confident, no escalation needed"
    )
    return EscalationDecision(
        window_id=output.window_id, escalate=uncertain,
        band_low=low, band_high=high, reason=reason,
    )


def choose_band_half_width(
    normal_scores: List[float], threshold: float, max_escalation_rate: float,
) -> float:
    """Return the LARGEST band half-width whose escalation rate on NORMAL
    VALIDATION scores stays at or below max_escalation_rate.

    Wider bands catch more genuinely-uncertain cases (the whole point of
    escalating), so "largest width within budget" is the useful target --
    not "smallest", which would trivially be near 0 and defeat the purpose.

    Mirrors defender/threshold.py's choose_threshold: normal_scores must
    be validation-only (never development or final-test), and there is no
    default max_escalation_rate -- the caller must decide it, since this
    is a real cost/coverage trade-off, not something to hardcode.

    Checks candidate widths directly (rather than deriving one from sorted
    distances with nextafter arithmetic) because two close-but-unequal
    validation scores can be separated by less than one float ULP at this
    scale, which makes boundary arithmetic unreliable; counting directly
    at each candidate width is slower but cannot be wrong in that way.
    """
    if not isinstance(normal_scores, (list, tuple)):
        raise TypeError("normal_scores must be a list of numbers")
    if len(normal_scores) == 0:
        raise ValueError("normal_scores is empty: need normal validation scores")
    _check_unit_interval(threshold, "threshold")
    _check_unit_interval(max_escalation_rate, "max_escalation_rate")
    for i, score in enumerate(normal_scores):
        _check_unit_interval(score, f"normal_scores[{i}]")

    n = len(normal_scores)
    allowed = math.floor(max_escalation_rate * n)

    def count_escalated(width: float) -> int:
        low = max(0.0, threshold - width)
        high = min(1.0, threshold + width)
        return sum(1 for s in normal_scores if low <= s <= high)

    if count_escalated(0.0) > allowed:
        raise ValueError(
            f"cannot achieve max_escalation_rate={max_escalation_rate}: even the "
            f"tightest possible band (width 0) already escalates "
            f"{count_escalated(0.0)} of {n} normal scores, because at least one "
            f"sits exactly at threshold={threshold}. Choose a higher "
            f"max_escalation_rate or a different threshold."
        )

    # count_escalated is non-decreasing in width (a wider band only ever
    # includes more scores), so the largest feasible candidate can be
    # found by scanning ascending and stopping at the first violation.
    # Each raw |s - threshold| is clipped to 0.5: should_escalate() only
    # accepts band_half_width in [0, 0.5], and a threshold away from 0.5
    # (the normal case -- real thresholds are rarely exactly 0.5) can
    # otherwise push a candidate width past that, which every existing
    # test missed because it only ever used threshold=0.5, where
    # |s - 0.5| can never exceed 0.5 in the first place.
    candidates = sorted({0.0, 0.5, *(min(0.5, abs(s - threshold)) for s in normal_scores)})
    best = 0.0
    for width in candidates:
        if count_escalated(width) <= allowed:
            best = width
        else:
            break
    return best
