"""
Decision threshold for the EdgeGuard Defender (Part 2).

Decision rule (must match the check in shared/schemas.py DefenderOutput):
    score >= threshold  ->  "ATTACK"
    score <  threshold  ->  "ACCEPT"

choose_threshold() must be given scores from NORMAL VALIDATION windows
only. Never use development or final-test data to choose the threshold.

There is NO default false-alarm rate. It is a team decision, so the
caller must always supply it.
"""

import json
import math
from pathlib import Path
from typing import List

ATTACK = "ATTACK"
ACCEPT = "ACCEPT"


def _check_unit_interval(value, name: str) -> None:
    """Value must be a real number from 0.0 to 1.0 (inclusive)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    if math.isnan(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")


def make_decision(score: float, threshold: float) -> str:
    """score >= threshold -> ATTACK, score < threshold -> ACCEPT."""
    _check_unit_interval(score, "score")
    _check_unit_interval(threshold, "threshold")
    return ATTACK if score >= threshold else ACCEPT


def choose_threshold(normal_scores: List[float], max_false_alarm_rate: float) -> float:
    """Return the LOWEST threshold that keeps
        false_alarms / total_normal_windows <= max_false_alarm_rate

    normal_scores: scores of NORMAL VALIDATION windows only.
    max_false_alarm_rate: e.g. 0.01 = at most 1 in 100 normal windows alarm.

    A false alarm is a normal window with score >= threshold.
    With max_false_alarm_rate = 0.0 the result is strictly greater than
    the highest normal score, so no normal window alarms.
    """
    if not isinstance(normal_scores, (list, tuple)):
        raise TypeError("normal_scores must be a list of numbers")
    if len(normal_scores) == 0:
        raise ValueError("normal_scores is empty: need normal validation scores")
    _check_unit_interval(max_false_alarm_rate, "max_false_alarm_rate")
    for i, score in enumerate(normal_scores):
        _check_unit_interval(score, f"normal_scores[{i}]")

    total = len(normal_scores)
    allowed_alarms = math.floor(max_false_alarm_rate * total)

    # Every window may alarm: the lowest possible threshold is 0.0.
    if allowed_alarms >= total:
        return 0.0

    ordered = sorted(normal_scores, reverse=True)   # highest score first
    # ordered[allowed_alarms] is the highest score that must NOT alarm.
    # The lowest threshold above it is the next representable number.
    # Scores tied with it are excluded too, so ties cannot add alarms.
    boundary = ordered[allowed_alarms]
    threshold = math.nextafter(boundary, math.inf)
    if threshold > 1.0:
        raise ValueError(
            "Cannot meet the target: normal validation windows score 1.0, so no "
            "threshold between 0 and 1 excludes them. Improve the detector or "
            "raise max_false_alarm_rate."
        )
    return threshold


def save_threshold(threshold: float, model_version: str, path) -> None:
    """Save the threshold together with the model version it belongs to."""
    _check_unit_interval(threshold, "threshold")
    if not isinstance(model_version, str) or not model_version:
        raise ValueError("model_version must be a non-empty string, e.g. 'v1'")
    record = {
        "threshold": threshold,
        "model_version": model_version,
        "rule": "score >= threshold -> ATTACK; score < threshold -> ACCEPT",
    }
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(record, indent=2), encoding="utf-8")


def load_threshold(path, expected_model_version: str) -> float:
    """Load a threshold; refuse it if it was saved for a different model."""
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(
            f"No threshold file at {file}. Run choose_threshold() and save_threshold() first."
        )
    record = json.loads(file.read_text(encoding="utf-8"))
    saved_version = record.get("model_version")
    if saved_version != expected_model_version:
        raise ValueError(
            f"Model version mismatch: threshold was saved for {saved_version!r}, "
            f"expected {expected_model_version!r}. Choose a new threshold for this model."
        )
    threshold = record.get("threshold")
    _check_unit_interval(threshold, "threshold")
    return threshold