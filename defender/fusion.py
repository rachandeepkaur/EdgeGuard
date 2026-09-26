"""
Fusion of Stage 1 (timing) and Stage 2 (payload) into one Defender score.

v2 = v1's UNCHANGED Stage 1 model + a NEW Stage 2 model, fused. Stage 1
is never retrained for v2 (see train.py train_v2), so the only possible
difference between v1 and v2 is Stage 2. This keeps the evidence-gate
claim exact: "an attack v1 missed, v2 catches" can only be explained by
the new Stage 2, never by a changed Stage 1.

Fusion rule: take the HIGHER of the two stage scores ("max fusion").
Consequence: fused_score >= stage1_score always. So at the SAME
threshold, anything Stage 1 alone would flag, the fused score still
flags. Stage 2 can only ADD detections, never remove one Stage 1 made.

Evidence names which stage decided, so a result is explainable without
opening logs: "[Stage 1] ..." or "[Stage 2] ...".
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from defender.stage1 import Stage1Result
from defender.stage2 import Stage2Result


class FusionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    score: float = Field(ge=0.0, le=1.0)
    stage: Literal["stage1", "stage2"]
    evidence: str


def fuse(stage1_result: Stage1Result, stage2_result: Stage2Result) -> FusionResult:
    """Combine Stage 1 and Stage 2 results: the higher score wins.
    A tie is attributed to Stage 1, since Stage 1 already ran in v1."""
    if not isinstance(stage1_result, Stage1Result):
        raise TypeError(f"fuse() needs a Stage1Result, got {type(stage1_result).__name__}")
    if not isinstance(stage2_result, Stage2Result):
        raise TypeError(f"fuse() needs a Stage2Result, got {type(stage2_result).__name__}")

    if stage1_result.score >= stage2_result.score:
        return FusionResult(score=stage1_result.score, stage="stage1",
                            evidence=f"[Stage 1] {stage1_result.evidence}")
    return FusionResult(score=stage2_result.score, stage="stage2",
                        evidence=f"[Stage 2] {stage2_result.evidence}")