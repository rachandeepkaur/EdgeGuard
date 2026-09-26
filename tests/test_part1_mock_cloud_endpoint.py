"""Tests for part1/mock_cloud_endpoint.py.

Run from EdgeGuard: python -m pytest tests/test_mock_cloud_endpoint.py -v
"""

import pytest
from fastapi.testclient import TestClient

from part1.mock_cloud_endpoint import app, request_second_opinion
from part1.sanitizer import SanitizedBundle

client = TestClient(app)


def bundle(score, decision, threshold=0.5):
    return SanitizedBundle(window_id="cap07_w0001", attack_score=score,
                           threshold=threshold, decision=decision,
                           evidence="x", model_version="v1")


# ---------- the FastAPI route itself -------------------------------------
def test_confirm_when_both_sides_agree_attack():
    response = client.post("/advise", json=bundle(0.9, "ATTACK").model_dump())
    assert response.status_code == 200
    assert response.json()["advice"] == "confirm"


def test_confirm_when_both_sides_agree_accept():
    response = client.post("/advise", json=bundle(0.1, "ACCEPT").model_dump())
    assert response.json()["advice"] == "confirm"


def test_overrides_to_accept_when_local_attack_but_below_standin_threshold():
    """Local says ATTACK (score 0.55 >= its own threshold 0.5), but the
    stand-in second threshold (0.6) would call this ACCEPT."""
    response = client.post("/advise", json=bundle(0.55, "ATTACK").model_dump())
    assert response.json()["advice"] == "override_to_accept"


def test_overrides_to_attack_when_local_accept_but_above_standin_threshold():
    response = client.post("/advise", json=bundle(0.65, "ACCEPT", threshold=0.7).model_dump())
    assert response.json()["advice"] == "override_to_attack"


def test_response_is_clearly_labeled_simulated():
    response = client.post("/advise", json=bundle(0.9, "ATTACK").model_dump())
    assert "simulated" in response.json()["source"].lower()
    assert "cloud" in response.json()["source"].lower()


def test_rejects_a_bundle_with_extra_fields():
    """SanitizedBundle is a StrictModel: nothing beyond its declared fields
    should ever reach this endpoint, including by accident."""
    payload = bundle(0.9, "ATTACK").model_dump()
    payload["raw_can_payload"] = "960C010204B10240"
    response = client.post("/advise", json=payload)
    assert response.status_code == 422   # FastAPI's validation error


# ---------- the client function -------------------------------------------
def test_client_returns_none_and_elapsed_time_when_unreachable():
    advice, ms = request_second_opinion(
        bundle(0.9, "ATTACK"), base_url="http://127.0.0.1:9", timeout_s=0.5,
    )
    assert advice is None
    assert ms >= 0.0


def test_client_never_raises_on_failure():
    """The local decision must remain usable regardless of what happens
    here -- this function must not throw, even for a bad URL."""
    try:
        request_second_opinion(bundle(0.9, "ATTACK"), base_url="not-a-url", timeout_s=0.2)
    except Exception as exc:
        pytest.fail(f"request_second_opinion raised {exc!r}; it must never raise")
