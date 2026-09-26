"""
Main Defender interface for EdgeGuard (Part 2).

This is the ONE function other parts call:

    score_window(window) -> DefenderOutput

Input:  a TrafficWindow (shared/schemas.py), already preprocessed by Part 1.
        A plain dict is also accepted; it is validated as a TrafficWindow
        first, so any label field (is_attack, family, ...) is rejected.
Output: DefenderOutput with window_id, attack_score, threshold, decision,
        evidence, model_version and latency_ms.

Versions:
    v1  Stage 1 (timing) only.
    v2  Stage 1 (v1's UNCHANGED model) + Stage 2 (payload), combined by
        defender.fusion.fuse(). Stage2 is optional in the constructor;
        Defender.load() auto-detects it from whether a stage2 file exists
        for that model version, so v1 keeps loading exactly as before.

THRESHOLD WIRING (important):
DefenderOutput checks that decision matches score >= threshold. The
threshold used here MUST be the one chosen by threshold.py (v1) or
train_v2 (v2) for THIS model version, loaded with load_threshold(path,
model_version). Never type a threshold by hand here. Defender.load()
does this for you, and refuses files saved for a different model version.

latency_ms measures Defender inference only (scoring one window, both
stages when present). Window collection time is NOT included; measure
it separately.
"""

import json
import math
import time
from pathlib import Path
from typing import Optional, Union

from defender.fusion import fuse
from defender.stage1 import Stage1Model
from defender.stage2 import Stage2Model
from defender.threshold import load_threshold, make_decision
from shared.schemas import DefenderOutput, TrafficWindow


def stage1_path(model_dir, model_version: str) -> Path:
    return Path(model_dir) / f"stage1_{model_version}.json"


def stage2_path(model_dir, model_version: str) -> Path:
    return Path(model_dir) / f"stage2_{model_version}.json"


def threshold_path(model_dir, model_version: str) -> Path:
    return Path(model_dir) / f"threshold_{model_version}.json"


def train_info_file(model_dir, model_version: str) -> Path:
    return Path(model_dir) / f"train_info_{model_version}.json"


class Defender:
    def __init__(self, stage1: Stage1Model, threshold: float, model_version: str,
                stage2: Optional[Stage2Model] = None, window_s: Optional[float] = None):
        """window_s: the window length the models were trained on. If given,
        score_window() refuses windows of any other length, so a model can
        never silently score windows cut differently from its training."""
        if not isinstance(stage1, Stage1Model) or not stage1.fitted:
            raise ValueError("Defender needs a trained Stage1Model")
        if stage2 is not None and (not isinstance(stage2, Stage2Model) or not stage2.fitted):
            raise ValueError("stage2, if given, must be a trained Stage2Model")
        make_decision(0.0, threshold)          # validates threshold is in [0, 1]
        if not isinstance(model_version, str) or not model_version:
            raise ValueError("model_version must be a non-empty string, e.g. 'v1'")
        if window_s is not None and window_s <= 0:
            raise ValueError(f"window_s must be positive, got {window_s}")
        self.stage1 = stage1
        self.stage2 = stage2
        self.threshold = threshold
        self.model_version = model_version
        self.window_s = window_s

    @classmethod
    def load(cls, model_dir, model_version: str) -> "Defender":
        """Load Stage 1 and its threshold for this model version. If a
        Stage 2 file also exists for this version, it is loaded too and
        the Defender fuses both (v2 behaviour); otherwise it behaves
        exactly like v1 (Stage 1 only). The training window length is read
        from train_info_<version>.json when that file exists."""
        stage1 = Stage1Model.load(stage1_path(model_dir, model_version), model_version)
        threshold = load_threshold(threshold_path(model_dir, model_version), model_version)
        stage2_file = stage2_path(model_dir, model_version)
        stage2 = Stage2Model.load(stage2_file, model_version) if stage2_file.exists() else None
        info_file = train_info_file(model_dir, model_version)
        window_s = json.loads(info_file.read_text(encoding="utf-8"))["window_s"] \
            if info_file.exists() else None
        return cls(stage1, threshold, model_version, stage2=stage2, window_s=window_s)

    def score_window(self, window: Union[TrafficWindow, dict]) -> DefenderOutput:
        """Score one window. Never modifies the window."""
        if isinstance(window, dict):
            window = TrafficWindow.model_validate(window)   # rejects label fields
        elif not isinstance(window, TrafficWindow):
            raise TypeError(
                f"score_window() needs a TrafficWindow or dict, got {type(window).__name__}"
            )
        if self.window_s is not None:
            duration = window.window_end - window.window_start
            if not math.isclose(duration, self.window_s, abs_tol=1e-6):
                raise ValueError(
                    f"window {window.window_id} is {duration} s long, but model "
                    f"{self.model_version} was trained on {self.window_s} s windows. "
                    f"Cut windows with part1 using the same window length."
                )

        started = time.perf_counter()
        stage1_result = self.stage1.score(window)
        if self.stage2 is not None:
            stage2_result = self.stage2.score(window)
            fusion_result = fuse(stage1_result, stage2_result)
            attack_score, evidence = fusion_result.score, fusion_result.evidence
        else:
            attack_score, evidence = stage1_result.score, f"[Stage 1] {stage1_result.evidence}"
        decision = make_decision(attack_score, self.threshold)
        latency_ms = (time.perf_counter() - started) * 1000.0

        return DefenderOutput(
            window_id=window.window_id,
            attack_score=attack_score,
            threshold=self.threshold,
            decision=decision,
            evidence=evidence,
            model_version=self.model_version,
            latency_ms=latency_ms,
        )


# ---------------------------------------------------------------------
# Module-level interface promised to Part 3:  score_window(window)
# ---------------------------------------------------------------------
_active_defender: Optional[Defender] = None


def set_active_defender(defender: Defender) -> None:
    """Choose which trained Defender score_window() uses."""
    global _active_defender
    if not isinstance(defender, Defender):
        raise TypeError("set_active_defender() needs a Defender")
    _active_defender = defender


def score_window(window: Union[TrafficWindow, dict]) -> DefenderOutput:
    if _active_defender is None:
        raise RuntimeError(
            "No Defender loaded. Call set_active_defender(Defender.load(model_dir, "
            "model_version)) first."
        )
    return _active_defender.score_window(window)