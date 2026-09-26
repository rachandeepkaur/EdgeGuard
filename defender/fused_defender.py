"""
Fused v2 + TimingCNN Defender (added 2026-09-25): ATTACK if EITHER the
already-shipped v2 (Stage1+Stage2, percentile-rank fusion) or TimingCNN
(causal Conv1d over 100ms timing bins, defender/timing_cnn.py) would call
ATTACK on its own. v2 is used exactly as already shipped -- never refit.

Real result, 2-fold DEVELOPMENT cross-validation
(results/crossval_fused_timing_cnn.json): v2 alone 63.3% recall / 1 false
alarm (out of 237 normal windows); TimingCNN alone 48.5% recall / 0 false
alarms; fused 76.4% recall / still just 1 false alarm. TimingCNN adds
recall almost entirely on v2's weakest families (reverse_light_off 1.6% ->
50%, reverse_light_on 32.2% -> 49.2%) without adding a single new false
alarm, while v2's own fuzzing/masquerade coverage (Stage1's unknown-ID
check, Stage2's payload checks) is preserved because v2 is OR'd with a
second detector, not replaced.

WHY an OR of two decisions, not a single fused score: v2's attack_score and
TimingCNN's live on two incomparable scales from two independent
calibrations (see defender/crossval_fused_timing_cnn.py's module docstring
for the full reasoning). The "attack_score" this class reports is NOT a
third calibration -- it exists only so DefenderOutput's own
score>=threshold consistency check holds. It expresses each sub-model's
score as a ratio to ITS OWN threshold (>=1.0 exactly when that model alone
would say ATTACK) and reports the higher of the two ratios, capped at 1.0,
against a fixed combined threshold of 1.0. Read it as "which of the two
fired, and how firmly" -- not as a calibrated probability.
"""

import time
from typing import Tuple

from defender.defender import Defender
from defender.threshold import ATTACK, make_decision
from defender.timing_cnn import TimingCNNModel, bin_timing_features
from part1.pipeline import preprocess
from shared.schemas import DefenderOutput, TrafficWindow


class FusedV2TimingCNNDefender:
    """Same score_window(window) -> DefenderOutput interface as
    defender.defender.Defender, so it drops into the exact same evaluation
    machinery (part3/evaluator.py, integration/run_final_evaluation.py, the
    dashboard, run_demo.py) with no changes to any of it."""

    def __init__(self, v2: Defender, cnn: TimingCNNModel,
                watch_ids: Tuple[str, ...] = ("0D0", "6E0"),
                model_version: str = "fused_v2_timing_cnn_v1"):
        if not isinstance(v2, Defender):
            raise ValueError("FusedV2TimingCNNDefender needs a Defender for v2")
        if not cnn.fitted:
            raise ValueError("FusedV2TimingCNNDefender needs a fitted TimingCNNModel")
        self.v2 = v2
        self.cnn = cnn
        self.watch_ids = watch_ids
        self.model_version = model_version

    def score_window(self, window: TrafficWindow) -> DefenderOutput:
        start = time.perf_counter()
        processed = preprocess(window)

        v2_out = self.v2.score_window(processed)
        v2_attack = v2_out.decision == ATTACK
        v2_ratio = v2_out.attack_score / v2_out.threshold if v2_out.threshold > 0 else 1.0

        features = bin_timing_features(processed, self.watch_ids)
        cnn_score = self.cnn.predict_proba_attack([features])[0]
        cnn_attack = cnn_score >= self.cnn.threshold
        cnn_ratio = cnn_score / self.cnn.threshold if self.cnn.threshold > 0 else 1.0

        combined_score = min(max(v2_ratio, cnn_ratio, float(v2_attack), float(cnn_attack)), 1.0)
        threshold = 1.0
        decision = make_decision(combined_score, threshold)
        latency_ms = (time.perf_counter() - start) * 1000

        fired = []
        if v2_attack:
            fired.append(f"v2 ({v2_out.evidence})")
        if cnn_attack:
            fired.append(f"TimingCNN (score {cnn_score:.4f} >= threshold {self.cnn.threshold:.4f})")
        evidence = "; ".join(fired) if fired else "Neither v2 nor TimingCNN alarmed"

        return DefenderOutput(
            window_id=window.window_id, attack_score=combined_score,
            threshold=threshold, decision=decision, evidence=evidence,
            model_version=self.model_version, latency_ms=latency_ms,
        )
