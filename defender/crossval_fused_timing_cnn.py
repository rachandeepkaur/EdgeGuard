"""
Two-fold, DEVELOPMENT-only cross-validation of a FUSED detector: the
ALREADY-SHIPPED, already-fitted v2 (Stage1+Stage2) OR'd with a newly-trained
TimingCNN (defender/timing_cnn.py). Mirrors defender/crossval_ml_baseline.py
and crossval_timing_cnn.py's protocol exactly (same folds, same masquerade
dedup, never reads final_test).

WHY an OR of two independently-calibrated decisions, not a single fused
score: v2's attack_score lives on Stage1/Stage2's shared percentile-rank
scale (threshold ~0.9988); TimingCNN's lives on a sigmoid probability scale
from a completely different fit (threshold ~0.6 here). They are not
comparable numbers, so max-fusing the RAW scores the way Stage1/Stage2
fuse (defender/fusion.py) would be meaningless. A decision-level OR keeps
each detector's own calibrated operating point intact: ATTACK if EITHER
v2 or TimingCNN would call it ATTACK on its own.

v2 itself is NOT refit per fold here -- it is the real, already-shipped,
already-frozen model (models/stage{1,2}_v2.json), loaded once. Only
TimingCNN is trained per fold, exactly as in crossval_timing_cnn.py. This
matches the real deployment question: "if we ADD TimingCNN alongside the
v2 we already have, does detection improve?" -- not "what if we retrained
v2 too."

Needs PyTorch -- run from the machine that has it.

Usage:
    .venv/bin/python3 -m defender.crossval_fused_timing_cnn \\
        --data-dir data/road --max-false-alarm-rate 0.01 --v2-model-dir models
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from defender.defender import Defender
from defender.threshold import ATTACK
from defender.timing_cnn import TimingCNNModel, bin_timing_features
from part1.pipeline import RoadData, preprocess
from part1.split_manifest import MANIFEST_PATH, load_manifest

DEFAULT_WATCH_IDS = ("0D0", "6E0")


def _examples(road: RoadData, watch_ids, names: List[str]
             ) -> Tuple[List[List[List[float]]], List[bool], List[dict]]:
    """TimingCNN feature tensors, is_attack labels and metadata for every
    window of the given captures. Same masquerade-dedup rule as the other
    crossval scripts: a masquerade capture contributes only its attacked
    windows."""
    features: List[List[List[float]]] = []
    labels: List[bool] = []
    meta: List[dict] = []
    for name in names:
        entry = road.manifest.entry(name)
        for window, label in road.labelled_windows(name):
            if entry.kind == "masquerade" and not label.is_attack:
                continue
            features.append(bin_timing_features(window, watch_ids))
            labels.append(bool(label.is_attack))
            meta.append({"capture_id": entry.capture_id, "family": entry.family,
                        "kind": entry.kind, "window": window})
    return features, labels, meta


def _macro_recall_by_family(rows: List[dict], key: str) -> Dict[str, float]:
    families = sorted({r["family"] for r in rows if r["attacked"] > 0})
    recalls = {}
    for family in families:
        chosen = [r for r in rows if r["family"] == family and r["attacked"] > 0]
        attacked = sum(r["attacked"] for r in chosen)
        detected = sum(r[key] for r in chosen)
        if attacked:
            recalls[family] = round(detected / attacked, 4)
    return recalls


def cross_validate(road: RoadData, v2: Defender, watch_ids,
                   max_false_alarm_rate: float, epochs: int = 40, log=print) -> dict:
    folds = road.manifest.development_folds()
    if len(folds) < 2:
        raise ValueError(f"Fused CV needs at least 2 development folds, got {sorted(folds)}")

    train_names = road.manifest.names("train")
    val_names = road.manifest.names("validation")
    log(f"Reading {len(train_names)} train + {len(val_names)} validation normal drives...")
    normal_features, normal_labels, _ = _examples(road, watch_ids, train_names)
    val_features, val_labels, _ = _examples(road, watch_ids, val_names)
    if any(normal_labels) or any(val_labels):
        raise ValueError("train/validation captures must be all-normal (is_attack=False)")

    rows = []
    for k, held_out in folds.items():
        other = [n for f, names in folds.items() if f != k for n in names]
        other_features, other_labels, _ = _examples(road, watch_ids, other)

        cnn = TimingCNNModel()
        cnn.fit(normal_features + other_features, normal_labels + other_labels,
               epochs=epochs, seed=k)
        cnn.calibrate_threshold(val_features, max_false_alarm_rate)
        log(f"Fold {k}: TimingCNN trained on {cnn.training_windows} windows "
           f"({cnn.training_attack_windows} attack), threshold={cnn.threshold:.4f}")

        for name in held_out:
            entry = road.manifest.entry(name)
            f_capture, l_capture, meta_capture = _examples(road, watch_ids, [name])
            cnn_scores = cnn.predict_proba_attack(f_capture) if f_capture else []

            v2_detected = cnn_detected = fused_detected = 0
            v2_false = cnn_false = fused_false = 0
            attacked = normal = 0
            for is_attack, cnn_score, m in zip(l_capture, cnn_scores, meta_capture):
                v2_out = v2.score_window(preprocess(m["window"]))
                v2_attack = v2_out.decision == ATTACK
                cnn_attack = cnn_score >= cnn.threshold
                fused_attack = v2_attack or cnn_attack
                if is_attack:
                    attacked += 1
                    v2_detected += v2_attack
                    cnn_detected += cnn_attack
                    fused_detected += fused_attack
                else:
                    normal += 1
                    v2_false += v2_attack
                    cnn_false += cnn_attack
                    fused_false += fused_attack

            rows.append({"capture_id": entry.capture_id, "family": entry.family,
                        "kind": entry.kind, "fold": k, "attacked": attacked, "normal": normal,
                        "v2_detected": v2_detected, "v2_false": v2_false,
                        "cnn_detected": cnn_detected, "cnn_false": cnn_false,
                        "fused_detected": fused_detected, "fused_false": fused_false})
            log(f"  {entry.capture_id} ({entry.family}/{entry.kind}): "
               f"v2 {v2_detected}/{attacked}, cnn {cnn_detected}/{attacked}, "
               f"fused {fused_detected}/{attacked} detected; "
               f"false alarms v2={v2_false} cnn={cnn_false} fused={fused_false} / {normal}")

    total_normal = sum(r["normal"] for r in rows)
    total_attacked = sum(r["attacked"] for r in rows)
    summary = {}
    for model_name in ("v2", "cnn", "fused"):
        total_detected = sum(r[f"{model_name}_detected"] for r in rows)
        total_false = sum(r[f"{model_name}_false"] for r in rows)
        summary[model_name] = {
            "overall_recall": round(total_detected / total_attacked, 4) if total_attacked else None,
            "macro_recall_by_family": _macro_recall_by_family(rows, f"{model_name}_detected"),
            "false_alarms": {"alarms": total_false, "windows": total_normal,
                            "false_alarm_rate": round(total_false / total_normal, 6)
                            if total_normal else None},
        }
    return {
        "protocol": "2-fold CV over development attack recordings (fold = index), "
                    "v2 (already-shipped, not refit) OR TimingCNN (fold-trained)",
        "watch_ids": list(watch_ids), "epochs": epochs,
        "max_false_alarm_rate": max_false_alarm_rate,
        "per_capture": [{k: v for k, v in r.items()} for r in rows],
        "summary": summary,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="v2 OR TimingCNN dev-only cross-validation (never reads final_test)."
    )
    parser.add_argument("--data-dir", default="data/road")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--max-false-alarm-rate", type=float, default=0.01)
    parser.add_argument("--watch-ids", default=",".join(DEFAULT_WATCH_IDS))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--v2-model-dir", default="models")
    parser.add_argument("--out", default="results/crossval_fused_timing_cnn.json")
    args = parser.parse_args(argv)

    watch_ids = tuple(i.strip().upper() for i in args.watch_ids.split(",") if i.strip())
    manifest = load_manifest(args.manifest)
    road = RoadData(args.data_dir, manifest=manifest)
    v2 = Defender.load(args.v2_model_dir, "v2")

    report = cross_validate(road, v2, watch_ids, args.max_false_alarm_rate, args.epochs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for name in ("v2", "cnn", "fused"):
        s = report["summary"][name]
        print(f"\n{name}: overall recall {s['overall_recall']}, "
             f"macro recall by family {s['macro_recall_by_family']}")
        print(f"  false alarms: {s['false_alarms']}")
    print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
