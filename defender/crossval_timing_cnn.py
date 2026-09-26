"""
Two-fold, DEVELOPMENT-only cross-validation for TimingCNN
(defender/timing_cnn.py) -- mirrors defender/crossval_ml_baseline.py and
defender/crossval.py's attack_cv() protocol exactly (same folds, same
masquerade-dedup rule, same train/validation/development split usage), so
the numbers are directly comparable to v1/v2's own development
cross-validation and to the ml_baseline comparison, with only the model
swapped: percentile-rank fusion / logistic regression -> a small causal
Conv1d over per-window timing bins.

Never reads final_test / separate: RoadData here is never constructed with
final_evaluation=True.

Needs PyTorch -- run from the machine that has it (see tests/test_timing_cnn.py).

Usage (from the EdgeGuard folder):
    .venv/bin/python3 -m defender.crossval_timing_cnn --data-dir data/road \\
        --max-false-alarm-rate 0.01
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from defender.timing_cnn import N_BINS, N_CHANNELS, TimingCNNModel, bin_timing_features
from part1.pipeline import RoadData
from part1.split_manifest import MANIFEST_PATH, load_manifest

DEFAULT_WATCH_IDS = ("0D0", "6E0")


def _examples(road: RoadData, watch_ids, names: List[str]
             ) -> Tuple[List[List[List[float]]], List[bool], List[dict]]:
    """(N_CHANNELS, N_BINS) feature tensors, is_attack labels and metadata
    for every window of the given captures. Same masquerade-dedup rule as
    defender/crossval.py's attack_cv() and crossval_ml_baseline.py's
    _examples(): a masquerade capture contributes only its attacked
    windows (byte-identical to its fabrication twin outside the injection
    interval)."""
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


def cross_validate(road: RoadData, watch_ids, max_false_alarm_rate: float,
                   epochs: int = 40, log=print) -> dict:
    folds = road.manifest.development_folds()
    if len(folds) < 2:
        raise ValueError(f"TimingCNN CV needs at least 2 development folds, got {sorted(folds)}")

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

        model = TimingCNNModel()
        model.fit(normal_features + other_features, normal_labels + other_labels,
                  epochs=epochs, seed=k)
        model.calibrate_threshold(val_features, max_false_alarm_rate)
        log(f"Fold {k}: trained on {model.training_windows} windows "
           f"({model.training_attack_windows} attack), "
           f"final_train_loss={model.final_train_loss:.4f}, threshold={model.threshold:.4f}")

        for name in held_out:
            entry = road.manifest.entry(name)
            f_capture, l_capture, _ = _examples(road, watch_ids, [name])
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
                    "TimingCNN (causal Conv1d over 100ms timing bins)",
        "n_bins": N_BINS, "n_channels": N_CHANNELS, "watch_ids": list(watch_ids),
        "epochs": epochs, "max_false_alarm_rate": max_false_alarm_rate,
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
        description="TimingCNN dev-only cross-validation (never reads final_test)."
    )
    parser.add_argument("--data-dir", default="data/road")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--max-false-alarm-rate", type=float, default=0.01)
    parser.add_argument("--watch-ids", default=",".join(DEFAULT_WATCH_IDS))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--out", default="results/crossval_timing_cnn.json")
    args = parser.parse_args(argv)

    watch_ids = tuple(i.strip().upper() for i in args.watch_ids.split(",") if i.strip())
    manifest = load_manifest(args.manifest)
    road = RoadData(args.data_dir, manifest=manifest)

    report = cross_validate(road, watch_ids, args.max_false_alarm_rate, args.epochs)
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
