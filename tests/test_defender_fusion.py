"""
Tests for defender/fusion.py.

Run from the EdgeGuard folder with:
    python -m pytest tests/test_defender_fusion.py -v
"""

import pytest

from defender.fusion import fuse
from defender.stage1 import Stage1Result
from defender.stage2 import Stage2Result


def s1(score, evidence="MOCK stage1 evidence"):
    return Stage1Result(score=score, sub_scores={"unknown_ids": score, "rate_excess": 0.0,
                                                  "short_gaps": 0.0}, evidence=evidence)


def s2(score, evidence="MOCK stage2 evidence"):
    return Stage2Result(score=score, sub_scores={"out_of_range": score, "frozen_break": 0.0,
                                                  "large_jump": 0.0}, evidence=evidence)


def test_stage1_wins_when_higher():
    result = fuse(s1(0.9), s2(0.3))
    assert result.score == 0.9 and result.stage == "stage1"
    assert result.evidence == "[Stage 1] MOCK stage1 evidence"


def test_stage2_wins_when_higher():
    result = fuse(s1(0.3), s2(0.9))
    assert result.score == 0.9 and result.stage == "stage2"
    assert result.evidence == "[Stage 2] MOCK stage2 evidence"


def test_tie_goes_to_stage1():
    result = fuse(s1(0.5), s2(0.5))
    assert result.stage == "stage1"


def test_fused_score_is_never_below_stage1_alone():
    """v2 can only add detections, never remove a Stage 1 one."""
    for a, b in [(0.9, 0.1), (0.1, 0.9), (0.5, 0.5), (0.0, 0.0), (1.0, 1.0)]:
        assert fuse(s1(a), s2(b)).score >= a


def test_rejects_wrong_first_argument():
    with pytest.raises(TypeError, match="Stage1Result"):
        fuse("not a result", s2(0.5))


def test_rejects_wrong_second_argument():
    with pytest.raises(TypeError, match="Stage2Result"):
        fuse(s1(0.5), "not a result")


def test_result_is_read_only():
    result = fuse(s1(0.9), s2(0.3))
    with pytest.raises(Exception):
        result.score = 0.0