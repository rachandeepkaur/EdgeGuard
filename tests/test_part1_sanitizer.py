"""Tests for part1/sanitizer.py.

Run from EdgeGuard: python -m pytest tests/test_sanitizer.py -v
"""

import pytest
from pydantic import ValidationError

from part1.sanitizer import SanitizedBundle, sanitize
from shared.schemas import DefenderOutput


def test_sanitize_copies_the_allowed_fields():
    output = DefenderOutput(window_id="cap07_w0153", attack_score=0.83, threshold=0.70,
                            decision="ATTACK", evidence="Cross-signal mismatch",
                            model_version="v1", latency_ms=4.2)
    bundle = sanitize(output)
    assert bundle.window_id == "cap07_w0153"
    assert bundle.attack_score == 0.83
    assert bundle.decision == "ATTACK"


def test_sanitize_drops_latency():
    """latency_ms is intentionally not forwarded -- see sanitizer.py's
    docstring for why."""
    output = DefenderOutput(window_id="w1", attack_score=0.5, threshold=0.5,
                            decision="ATTACK", evidence="x", model_version="v1",
                            latency_ms=99.0)
    bundle = sanitize(output)
    assert not hasattr(bundle, "latency_ms")


def test_sanitized_bundle_rejects_extra_fields():
    """StrictModel (extra='forbid'): nothing beyond the declared fields
    can ever be attached to a bundle, even by accident."""
    with pytest.raises(ValidationError):
        SanitizedBundle(window_id="w1", attack_score=0.5, threshold=0.5,
                        decision="ATTACK", evidence="x", model_version="v1",
                        raw_can_payload="960C010204B10240")
