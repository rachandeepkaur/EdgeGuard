"""
Supervised ML baseline (Part 2 extension, comparison only -- added 2026-09-25).

Answers a question raised on review: does a LEARNED classifier beat
EdgeGuard's percentile-rank anomaly fusion (defender/fusion.py), given the
exact same six raw signals Stage 1 and Stage 2 already compute? This isolates
"supervised learning vs. percentile-rank scoring" as the only variable --
feature engineering is held identical to the shipped v1/v2 models, so any
difference in the result comes from HOW the six numbers are combined, not
from seeing more of the bus.

NOT part of the shipped detector. v1/v2 deliberately avoid any ML runtime
dependency (plain Python, no numpy) to support the "fits a gateway ECU"
footprint claim in the README; this baseline uses scikit-learn and is a
comparison experiment only. Nothing in defender/defender.py, run_demo.py or
run_training.py imports this module.

Unlike v1/v2, which never see a single attack example during training (by
design -- see defender/stage1.py and stage2.py), this model IS a genuine
supervised classifier: it is fit on real attack labels from `development`
captures. It reuses the ALREADY-FROZEN, ALREADY-FITTED v2 stage1/stage2
models purely as a feature extractor (their fit() methods are never called
again here) -- so every number is directly comparable to v2's own fusion of
the same six signals, on the same watch-list.
"""

import json
import math
import time
from pathlib import Path
from typing import List, Tuple

from sklearn.linear_model import LogisticRegression

from defender.stage1 import Stage1Model
from defender.stage2 import Stage2Model
from defender.threshold import choose_threshold, make_decision
from shared.schemas import DefenderOutput, TrafficWindow

FEATURES: Tuple[str, ...] = (
    "unknown_ids", "rate_excess", "short_gaps",
    "out_of_range", "frozen_break", "large_jump",
)


def extract_features(stage1: Stage1Model, stage2: Stage2Model,
                     window: TrafficWindow) -> List[float]:
    """The exact six raw check values Stage 1 / Stage 2 already compute for
    this window (the same values defender.stage1._unusualness turns into
    percentile-rank scores). No new signal is added here."""
    raw1, _ = stage1._raw_values(window)
    raw2, _ = stage2._raw_values(window)
    return [raw1["unknown_ids"], raw1["rate_excess"], raw1["short_gaps"],
            raw2["out_of_range"], raw2["frozen_break"], raw2["large_jump"]]


class MLBaselineModel:
    """A thin, plain-JSON-serializable wrapper around a scikit-learn
    LogisticRegression, so this stays inspectable -- a learned coefficient
    per raw signal, the same spirit as v1/v2's reference distributions,
    just a learned linear combination instead of a percentile-rank one."""

    def __init__(self):
        self.fitted = False
        self.coef: List[float] = []
        self.intercept: float = 0.0
        self.threshold: float = 1.0
        self.training_windows = 0
        self.training_attack_windows = 0

    def fit(self, feature_rows: List[List[float]], labels: List[bool]) -> "MLBaselineModel":
        if len(feature_rows) != len(labels):
            raise ValueError("feature_rows and labels must be the same length")
        if len(feature_rows) < 2:
            raise ValueError("fit() needs at least 2 examples")
        if len(set(labels)) < 2:
            raise ValueError("fit() needs at least one normal and one attack example")
        clf = LogisticRegression(class_weight="balanced", max_iter=2000)
        clf.fit(feature_rows, [1 if lbl else 0 for lbl in labels])
        self.coef = clf.coef_[0].tolist()
        self.intercept = float(clf.intercept_[0])
        self.training_windows = len(feature_rows)
        self.training_attack_windows = sum(1 for lbl in labels if lbl)
        self.fitted = True
        return self

    def _proba_one(self, row: List[float]) -> float:
        z = self.intercept + sum(c * x for c, x in zip(self.coef, row))
        # Numerically-stable sigmoid: avoids OverflowError on a very negative z.
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        e = math.exp(z)
        return e / (1.0 + e)

    def predict_proba_attack(self, feature_rows: List[List[float]]) -> List[float]:
        if not self.fitted:
            raise RuntimeError("MLBaselineModel is not trained. Call fit() first.")
        return [self._proba_one(row) for row in feature_rows]

    def calibrate_threshold(self, normal_feature_rows: List[List[float]],
                            max_false_alarm_rate: float) -> float:
        """Same discipline as v1/v2 (defender/threshold.py): the threshold is
        chosen on NORMAL validation scores only, never on attack data."""
        scores = self.predict_proba_attack(normal_feature_rows)
        self.threshold = choose_threshold(scores, max_false_alarm_rate)
        return self.threshold

    def save(self, path) -> None:
        if not self.fitted:
            raise RuntimeError("Cannot save an untrained MLBaselineModel")
        record = {
            "features": list(FEATURES),
            "coef": self.coef,
            "intercept": self.intercept,
            "threshold": self.threshold,
            "training_windows": self.training_windows,
            "training_attack_windows": self.training_attack_windows,
        }
        file = Path(path)
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(json.dumps(record, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "MLBaselineModel":
        file = Path(path)
        if not file.exists():
            raise FileNotFoundError(f"No ML baseline model at {file}")
        record = json.loads(file.read_text(encoding="utf-8"))
        if tuple(record["features"]) != FEATURES:
            raise ValueError(
                f"Saved feature order {record['features']} does not match "
                f"FEATURES {FEATURES} -- retrain"
            )
        model = cls()
        model.coef = record["coef"]
        model.intercept = record["intercept"]
        model.threshold = record["threshold"]
        model.training_windows = record["training_windows"]
        model.training_attack_windows = record["training_attack_windows"]
        model.fitted = True
        return model


class MLBaselineDefender:
    """Same score_window(window) -> DefenderOutput interface as
    defender.defender.Defender, so it drops into the exact same evaluation
    machinery (part3/evaluator.py, part3/metrics.py, the one-time final_test
    loop in integration/run_final_evaluation.py) with no changes to any of
    it. stage1/stage2 here are v2's ALREADY-FROZEN models, borrowed purely
    as a feature extractor -- never refit."""

    def __init__(self, stage1: Stage1Model, stage2: Stage2Model,
                model: MLBaselineModel, model_version: str = "ml_baseline_v1"):
        if not (isinstance(stage1, Stage1Model) and stage1.fitted):
            raise ValueError("MLBaselineDefender needs a fitted Stage1Model")
        if not (isinstance(stage2, Stage2Model) and stage2.fitted):
            raise ValueError("MLBaselineDefender needs a fitted Stage2Model")
        if not model.fitted:
            raise ValueError("MLBaselineDefender needs a fitted MLBaselineModel")
        self.stage1 = stage1
        self.stage2 = stage2
        self.model = model
        self.model_version = model_version

    def score_window(self, window: TrafficWindow) -> DefenderOutput:
        start = time.perf_counter()
        features = extract_features(self.stage1, self.stage2, window)
        score = self.model.predict_proba_attack([features])[0]
        latency_ms = (time.perf_counter() - start) * 1000
        decision = make_decision(score, self.model.threshold)
        contributions = [abs(c * x) for c, x in zip(self.model.coef, features)]
        top = FEATURES[max(range(len(FEATURES)), key=lambda i: contributions[i])]
        evidence = (f"ML baseline (logistic regression over {', '.join(FEATURES)}); "
                   f"top contributor: {top}")
        return DefenderOutput(
            window_id=window.window_id, attack_score=score,
            threshold=self.model.threshold, decision=decision,
            evidence=evidence, model_version=self.model_version,
            latency_ms=latency_ms,
        )
