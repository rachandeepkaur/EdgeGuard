"""
Part 1: the "second FastAPI process" from build plan section 7.

    "For the short demo, a second FastAPI process can illustrate the
    policy. Clearly label it 'simulated cloud' if it runs locally...
    Do not describe a local process as actual cloud computing."

This file is that process. Every name in it says SIMULATED on purpose --
this is a stand-in for a second opinion, not a trained model, and the
demo/pitch must describe it that way. If real off-device deployment is
feasible later, document where it actually runs and re-measure the same
policy against that; do not just rename this file.

Run it directly for the demo:
    uvicorn part1.mock_cloud_endpoint:app --host 127.0.0.1 --port 9000

The client side (request_second_opinion) is what a Part 1/2 escalation
path actually calls. It NEVER raises on failure or timeout: build plan
section 7 requires "the immediate local decision remains available even
if the second service fails or is slow", so a failed or slow call comes
back as (None, elapsed_ms), not an exception the caller has to catch.
"""

import time
from typing import Literal, Optional

import httpx
from fastapi import FastAPI
from pydantic import ValidationError

from part1.sanitizer import SanitizedBundle
from shared.schemas import StrictModel

SIMULATED_CLOUD_LABEL = "simulated-cloud-not-a-real-cloud-deployment"

# A different, deliberately simple fixed threshold, standing in for "a
# second, independent opinion". This is NOT a trained model -- it exists
# so the demo can show a genuine occasional disagreement with the local
# decision (build plan section 7: "whether the second opinion corrects
# any errors"), without pretending to be more sophisticated than it is.
_SIMULATED_SECOND_THRESHOLD = 0.6


class SimulatedCloudAdvice(StrictModel):
    window_id: str
    advice: Literal["confirm", "override_to_attack", "override_to_accept"]
    note: str
    source: str = SIMULATED_CLOUD_LABEL


app = FastAPI(title="EdgeGuard simulated cloud (NOT a real cloud deployment)")


@app.post("/advise", response_model=SimulatedCloudAdvice)
def advise(bundle: SanitizedBundle) -> SimulatedCloudAdvice:
    """Simulated second opinion, using a different fixed threshold than
    the local Defender. Real cloud deployment would call an actual model
    here; this is an explicitly-labeled stand-in for the demo."""
    second_opinion_attack = bundle.attack_score >= _SIMULATED_SECOND_THRESHOLD
    local_attack = bundle.decision == "ATTACK"

    if second_opinion_attack == local_attack:
        return SimulatedCloudAdvice(
            window_id=bundle.window_id, advice="confirm",
            note=f"simulated second opinion agrees with the local decision "
                 f"(score {bundle.attack_score:.3f} vs stand-in threshold "
                 f"{_SIMULATED_SECOND_THRESHOLD})",
        )
    advice_value = "override_to_attack" if second_opinion_attack else "override_to_accept"
    return SimulatedCloudAdvice(
        window_id=bundle.window_id, advice=advice_value,
        note=f"simulated second opinion disagrees with the local decision "
             f"(score {bundle.attack_score:.3f} vs stand-in threshold "
             f"{_SIMULATED_SECOND_THRESHOLD})",
    )


def request_second_opinion(
    bundle: SanitizedBundle, *, base_url: str, timeout_s: float = 2.0,
) -> tuple[Optional[SimulatedCloudAdvice], float]:
    """POST the sanitized bundle to the simulated cloud service.

    Returns (advice, round_trip_ms). advice is None on any failure --
    connection error, timeout, bad response -- and this function never
    raises for those cases: the local decision must remain usable no
    matter what happens here. round_trip_ms is always returned, even on
    failure, so added latency can still be measured.
    """
    start = time.perf_counter()
    try:
        response = httpx.post(
            f"{base_url}/advise", json=bundle.model_dump(), timeout=timeout_s,
        )
        response.raise_for_status()
        advice = SimulatedCloudAdvice(**response.json())
    except (httpx.HTTPError, ValidationError, ValueError):
        return None, (time.perf_counter() - start) * 1000.0
    return advice, (time.perf_counter() - start) * 1000.0
