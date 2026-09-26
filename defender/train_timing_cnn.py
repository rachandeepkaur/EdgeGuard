"""
Final, single TimingCNN fit for deployment/evaluation (as opposed to
defender/crossval_timing_cnn.py's per-fold fits, which exist only to
produce an honest cross-validated estimate). Mirrors how v1/v2 themselves
are finalized: fit on ALL of train (normal) + validation (normal) +
development (real, labelled attacks), threshold calibrated on validation
alone -- the same discipline as defender/threshold.py's choose_threshold().

This is the model integration/run_final_evaluation_fused.py loads to build
the one-time final_test comparison. It never reads final_test itself.

Usage:
    .venv/bin/python3 -m defender.train_timing_cnn --data-dir data/road \\
        --max-false-alarm-rate 0.01 --model-dir models/timing_cnn_v1
"""

import argparse
from pathlib import Path
from typing import List, Tuple

from defender.timing_cnn import TimingCNNModel, bin_timing_features
from part1.pipeline import RoadData
from part1.split_manifest import MANIFEST_PATH, load_manifest

DEFAULT_WATCH_IDS = ("0D0", "6E0")


def _examples(road: RoadData, watch_ids, names: List[str]
             ) -> Tuple[List[List[List[float]]], List[bool]]:
    features: List[List[List[float]]] = []
    labels: List[bool] = []
    for name in names:
        entry = road.manifest.entry(name)
        for window, label in road.labelled_windows(name):
            if entry.kind == "masquerade" and not label.is_attack:
                continue
            features.append(bin_timing_features(window, watch_ids))
            labels.append(bool(label.is_attack))
    return features, labels


def train_final(road: RoadData, watch_ids, max_false_alarm_rate: float,
                epochs: int = 60, seed: int = 0, log=print) -> TimingCNNModel:
    train_names = road.manifest.names("train")
    val_names = road.manifest.names("validation")
    dev_names = road.manifest.names("development")

    log(f"Reading {len(train_names)} train + {len(val_names)} validation "
       f"+ {len(dev_names)} development captures...")
    normal_features, normal_labels = _examples(road, watch_ids, train_names)
    val_features, val_labels = _examples(road, watch_ids, val_names)
    dev_features, dev_labels = _examples(road, watch_ids, dev_names)
    if any(normal_labels) or any(val_labels):
        raise ValueError("train/validation captures must be all-normal (is_attack=False)")

    model = TimingCNNModel()
    model.fit(normal_features + dev_features, normal_labels + dev_labels,
             epochs=epochs, seed=seed)
    model.calibrate_threshold(val_features, max_false_alarm_rate)
    log(f"Trained on {model.training_windows} windows "
       f"({model.training_attack_windows} attack), "
       f"final_train_loss={model.final_train_loss:.4f}, threshold={model.threshold:.4f}")
    return model


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Final TimingCNN fit (train+validation+development; never final_test)."
    )
    parser.add_argument("--data-dir", default="data/road")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--max-false-alarm-rate", type=float, default=0.01)
    parser.add_argument("--watch-ids", default=",".join(DEFAULT_WATCH_IDS))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-dir", default="models/timing_cnn_v1")
    args = parser.parse_args(argv)

    watch_ids = tuple(i.strip().upper() for i in args.watch_ids.split(",") if i.strip())
    manifest = load_manifest(args.manifest)
    road = RoadData(args.data_dir, manifest=manifest)

    model = train_final(road, watch_ids, args.max_false_alarm_rate, args.epochs, args.seed)
    model.save(args.model_dir)
    print(f"Saved to {args.model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
