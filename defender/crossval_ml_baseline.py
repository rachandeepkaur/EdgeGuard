"""
Two-fold, DEVELOPMENT-only cross-validation for the supervised ML baseline
(defender/ml_baseline.py) -- mirrors defender/crossval.py's attack_cv()
protocol exactly (same folds, same masquerade-dedup rule, same train/
validation/development split usage), so the numbers are directly comparable
to v1/v2's own development cross-validation, with only the model swapped:
percentile-rank fusion -> a learned logistic regression over the same six
raw signals (defender/ml_baseline.py's FEATURES).

Never reads final_test / separate: RoadData here is never constructed with
final_evaluation=True, and the manifest refuses those groups otherwise.
The one-time final_test comparison, if the team accepts that risk, is a
separate, explicit step in integration/run_final_evaluation.py (the ONLY
place in this codebase allowed to read final_test).

Usage (from the EdgeGuard folder, .venv active):
    python -m defender.crossval_ml_baseline --data-dir data/road \\
        --max-false-alarm-rate 0.01 --v2-model-dir models
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from defender.defender import Defender
from defender.ml_baseline import FEATURES, MLBaselineModel, extract_features
from part1.pipeline import RoadData
from part1.split_manifest import MANIFEST_PATH, load_manifest


def _examples(road: RoadData, v2: Defender, names: List[str]
             ) -> Tuple[List[List[float]], List[bool], List[dict]]:
    """features, is_attack labels and per-window metadata for every window of
    the given captures. A masquerade capture is byte-identical to its
    fabrication twin outside the injection interval, so (exactly as
    defender/crossval.py's attack_cv() does) only ITS ATTACKED windows are
    kept -- normal windows are counted once, from the fabrication capture."""
    features: List[List[float]] = []
    labels: List[bool] = []
    meta: List[dict] = []
    for name in names:
        entry = road.manifest.entry(name)
        for window, label in road.labelled_windows(name):
            if entry.kind == "masquerade" and not label.is_attack:
                continue
            features.append(extract_features(v2.stage1, v2.stage2, window))
            labels.append(bool(label.is_attack))
            meta.append({"capture_id": entry.capture_id, "family": entry.family,
                        "kind": entry.kind})
    return features, labels, meta


def _macro_recall_by_family(rows: List[dict]) -> Dict[str, float]:
    families = sorted({r["family"] for r in rows if r["attacked"] > 0})
    recalls = {}
    for family in families:
        chosen = [r for r in rows if r["family"] == family and r["attacked"] > 0]
        attacked = sum(r["attacked"] for r in chosen)
        detected = sum(r["detected"] for r in chosen)
        if attacked:
            recalls[family] = round(detected / attacked, 4)
    return recalls


def cross_validate(road: RoadData, v2: Defender, max_false_alarm_rate: float,
                   log=print) -> dict:
    folds = road.manifest.development_folds()
    if len(folds) < 2:
        raise ValueError(f"ML baseline CV needs at least 2 development folds, got {sorted(folds)}")

    train_names = road.manifest.names("train")
    val_names = road.manifest.names("validation")
    log(f"Reading {len(train_names)} train + {len(val_names)} validation normal drives...")
    normal_features, normal_labels, _ = _examples(road, v2, train_names)
    val_features, val_labels, _ = _examples(road, v2, val_names)
    if any(normal_labels) or any(val_labels):
        raise ValueError("train/validation captures must be all-normal (is_attack=False)")

    rows = []
    for k, held_out in folds.items():
        other = [n for f, names in folds.items() if f != k for n in names]
        other_features, other_labels, _ = _examples(road, v2, other)

        model = MLBaselineModel().fit(normal_features + other_features,
                                      normal_labels + other_labels)
        model.calibrate_threshold(val_features, max_false_alarm_rate)
        log(f"Fold {k}: trained on {model.training_windows} windows "
           f"({model.training_attack_windows} attack), threshold={model.threshold:.4f}")

        for name in held_out:
            entry = road.manifest.entry(name)
            f_capture, l_capture, _ = _examples(road, v2, [name])
            scores = model.predict_proba_attack(f_capture) if f_capture else []
            detected = sum(1 for s, y in zip(scores, l_capture) if y and s >= model.threshold)
            false_alarms = sum(1 for s, y in zip(scores, l_capture) if not y and s >= model.threshold)
            attacked = sum(1 for y in l_capture if y)
            normal = sum(1 for y in l_capture if not y)
            rows.append({"capture_id": entry.capture_id, "family": entry.family,
                        "kind": entry.kind, "fold": k, "attacked": attacked,
                        "normal": normal, "detected": detected, "false_alarms": false_alarms})
            log(f"  {entry.capture_id} ({entry.family}/{entry.kind}): "
               f"{detected}/{attacked} detected, {false_alarms}/{normal} false alarms")

    total_normal = sum(r["normal"] for r in rows)
    total_alarms = sum(r["false_alarms"] for r in rows)
    total_attacked = sum(r["attacked"] for r in rows)
    total_detected = sum(r["detected"] for r in rows)
    return {
        "protocol": "2-fold CV over development attack recordings (fold = index), "
                    "ML baseline (logistic regression over Stage1+Stage2's raw signals)",
        "features": list(FEATURES),
        "max_false_alarm_rate": max_false_alarm_rate,
        "per_capture": rows,
        "macro_recall_by_family": _macro_recall_by_family(rows),
        "overall_recall": round(total_detected / total_attacked, 4) if total_attacked else None,
        "false_alarms_in_attack_captures": {
            "alarms": total_alarms, "windows": total_normal,
            "false_alarm_rate": round(total_alarms / total_normal, 6) if total_normal else None,
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="ML baseline dev-only cross-validation (never reads final_test)."
    )
    parser.add_argument("--data-dir", default="data/road")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--max-false-alarm-rate", type=float, default=0.01)
    parser.add_argument("--v2-model-dir", default="models",
                        help="where v2's ALREADY-FITTED stage1/stage2 live -- borrowed "
                             "here purely as a feature extractor, never refit")
    parser.add_argument("--out", default="results/crossval_ml_baseline.json")
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    road = RoadData(args.data_dir, manifest=manifest)
    v2 = Defender.load(args.v2_model_dir, "v2")

    report = cross_validate(road, v2, args.max_false_alarm_rate)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nOverall recall: {report['overall_recall']}")
    print(f"Macro recall by family: {report['macro_recall_by_family']}")
    print(f"False alarms (attack-capture normal windows): "
         f"{report['false_alarms_in_attack_captures']}")
    print(f"Saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
