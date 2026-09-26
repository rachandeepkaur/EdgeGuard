"""
Integration: Part 1 windows -> Part 2 Stage 1.

Proves the two parts fit together BEFORE real training happens: windows
produced by part1.windowing are accepted by defender.stage1, a model fitted
on them treats fresh normal traffic as normal, and the two loud attack
families Stage 1 is built for score higher than normal.

Run from the EdgeGuard folder:
    python -m pytest tests/test_part1_stage1_integration.py -v

Traffic here is SYNTHETIC, generated in ROAD's exact log format. These tests
check that the parts connect; they are not detection results on ROAD.
"""

import random

import pytest

from defender.stage1 import Stage1Model
from part1.windowing import make_windows

# Periodic IDs, as a real bus has: id -> broadcast period in seconds.
PERIODS = {"0D0": 0.010, "0F4": 0.020, "1A0": 0.050, "32D": 0.100}
BASE = 1110000000.0


def normal_lines(seconds, seed):
    rng = random.Random(seed)
    events = []
    for can_id, period in PERIODS.items():
        t = rng.uniform(0, period)
        while t < seconds:
            events.append((t + rng.gauss(0, period * 0.02), can_id))
            t += period
    events.sort()
    return [(t, cid, "0011223344556677") for t, cid in events if t >= 0]


def write_capture(path, events):
    events = sorted(events)
    path.write_text("\n".join(
        f"({BASE + t:.6f}) can0 {cid}#{payload}" for t, cid, payload in events
    ) + "\n")
    return str(path)


@pytest.fixture
def fitted(tmp_path):
    """Stage 1 fitted on Part 1 windows from two normal captures."""
    windows = []
    for i, seed in enumerate((1, 2), start=1):
        path = write_capture(tmp_path / f"n{i}.log", normal_lines(40, seed))
        windows += list(make_windows(path, f"cap{i:02d}"))
    return Stage1Model().fit(windows), tmp_path


def test_part1_windows_are_accepted_by_stage1(fitted):
    model, _ = fitted
    assert model.fitted
    assert set(model.known_ids) == set(PERIODS)


def test_fresh_normal_traffic_scores_as_normal(fitted):
    model, tmp_path = fitted
    path = write_capture(tmp_path / "fresh.log", normal_lines(20, seed=99))
    top = model.training_windows / (model.training_windows + 1)
    scores = [model.score(w).score for w in make_windows(path, "cap03")]
    # Nearly all fresh normal windows sit inside the normal training range.
    inside = sum(s <= top for s in scores)
    assert inside / len(scores) >= 0.9


def test_fuzzing_with_unknown_id_is_flagged(fitted):
    model, tmp_path = fitted
    events = normal_lines(20, seed=7)
    events += [(10.0 + k * 0.01, "7FF", "FFFFFFFFFFFFFFFF") for k in range(40)]
    path = write_capture(tmp_path / "fuzz.log", events)
    by_id = {w.window_id: model.score(w) for w in make_windows(path, "cap04")}
    attacked = by_id["cap04_w0010"]
    assert attacked.score > 0.95
    assert "unknown_ids" in attacked.evidence


def test_fabrication_at_double_rate_is_flagged(fitted):
    """Spoofing with message confliction: the real ECU keeps sending, so the
    target ID arrives at roughly twice its normal rate."""
    model, tmp_path = fitted
    events = normal_lines(20, seed=8)
    events += [(12.0 + k * 0.010 + 0.005, "0D0", "FFFF020080000000")
               for k in range(100)]
    path = write_capture(tmp_path / "fab.log", events)
    by_id = {w.window_id: model.score(w) for w in make_windows(path, "cap05")}
    normal_score = by_id["cap05_w0005"].score
    attacked = by_id["cap05_w0012"]
    assert attacked.score > normal_score
    assert attacked.score > 0.95


def test_elapsed_timestamps_work_with_stage1(fitted):
    """Part 1 emits elapsed seconds; Stage 1 only uses differences, so the
    epoch-vs-elapsed choice must not change behaviour."""
    model, tmp_path = fitted
    path = write_capture(tmp_path / "e.log", normal_lines(5, seed=3))
    first = next(make_windows(path, "cap06"))
    assert first.window_start == 0.0
    model.score(first)   # must not raise
