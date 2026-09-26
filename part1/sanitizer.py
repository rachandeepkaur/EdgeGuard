"""Part 1: build the summary that is allowed to leave the device.

Build plan section 7: "send a derived summary without raw CAN bytes."
This is an explicit allow-list, not a strip-list: SanitizedBundle names
every field that may leave, and StrictModel (extra="forbid") rejects
anything else. If DefenderOutput ever grows a new field -- for example
a raw window attached for debugging -- this file does not need to
remember to exclude it; sanitize() only ever copies the fields listed
below, on purpose.

DefenderOutput itself already contains no raw CAN data (see
shared/schemas.py), so this is defense in depth and an auditable
contract, not a fix for an active leak.
"""

from shared.schemas import DefenderOutput, StrictModel


class SanitizedBundle(StrictModel):
    """Everything a cloud second-opinion service is allowed to see."""

    window_id: str
    attack_score: float
    threshold: float
    decision: str
    evidence: str
    model_version: str


def sanitize(output: DefenderOutput) -> SanitizedBundle:
    """Build the bundle that may be sent off-device for a second opinion.

    Deliberately does not forward output.latency_ms: timing/hardware
    fingerprinting is not needed by a second opinion and is left out on
    the same allow-list principle, not because it is dangerous by itself.
    """
    return SanitizedBundle(
        window_id=output.window_id,
        attack_score=output.attack_score,
        threshold=output.threshold,
        decision=output.decision,
        evidence=output.evidence,
        model_version=output.model_version,
    )
