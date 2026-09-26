"""
DEVELOPMENT-only evaluation for DefenderLLM (defender/defender_llm.py) --
same captures, same masquerade-dedup rule, same train/validation/development
usage as defender/crossval.py's attack_cv() and defender/crossval_ml_baseline.py,
so the numbers are directly comparable to v1/v2/ml_baseline/TimingCNN's own
development numbers, with only the model swapped.

UNLIKE ml_baseline/TimingCNN, DefenderLLM has no fit() step: it is a frozen,
pretrained, zero-shot model -- the same weights score every window
regardless of which development fold that window happens to fall in. So
there is nothing to retrain per fold here. "Fold" below is kept ONLY as a
reporting partition, purely so this script's output has the same shape
(folds / per_capture / macro_recall_by_family) as every other crossval
script for side-by-side comparison -- it does not mean a different model
was used per fold, because there is only ever one DefenderLLMModel, scored
once, calibrated once (on validation, same discipline as everywhere else).

This makes a real run cheaper than it looks: ONE calibration pass over
validation, then ONE scoring pass over all of development -- not a
per-fold retrain-and-rescore like ml_baseline/TimingCNN need.

Real LLM inference is slow relative to everything else in this codebase
(a forward pass per window, not a lookup). Use --max-windows-per-capture
for a fast smoke-scale dry run before committing to the full development
set -- see --help.

Never reads final_test / separate: RoadData here is never constructed with
final_evaluation=True.

Usage (from the EdgeGuard folder, .venv active, transformers+torch installed):
    # fast smoke-scale dry run first:
    python -m defender.crossval_defender_llm --data-dir data/road \\
        --max-false-alarm-rate 0.01 --v2-model-dir models --max-windows-per-capture 5
    # then the real run:
    python -m defender.crossval_defender_llm --data-dir data/road \\
        --max-false-alarm-rate 0.01 --v2-model-dir models
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

from defender.defender import Defender
from defender.defender_llm import DEFAULT_MODEL_NAME, DefenderLLMModel
from defender.ml_baseline import FEATURES, extract_features
from part1.pipeline import RoadData
from part1.split_manifest import MANIFEST_PATH, load_manifest


def _examples(road: RoadData, v2: Defender, names: List[str], max_windows: int
             ) -> Tuple[List[List[float]], List[bool], List[dict]]:
    features: List[List[float]] = []
    labels: List[bool] = []
    meta: List[dict] = []
    for name in names:
        entry = road.manifest.entry(name)
        step = road.keep_every(name, max_windows)
        for window, label in road.labelled_windows(name, keep_every=step):
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


def cross_validate(road: RoadData, v2: Defender, model: DefenderLLMModel,
                   max_false_alarm_rate: float, max_windows_per_capture: int = 100_000,
                   log=print) -> dict:
    folds = road.manifest.development_folds()
    if len(folds) < 2:
        raise ValueError(f"DefenderLLM eval needs at least 2 development folds, got {sorted(folds)}")

    val_names = road.manifest.names("validation")
    log(f"Reading {len(val_names)} validation normal windows for threshold calibration...")
    val_features, val_labels, _ = _examples(road, v2, val_names, max_windows_per_capture)
    if any(val_labels):
        raise ValueError("validation captures must be all-normal (is_attack=False)")

    t0 = time.time()
    model.calibrate_threshold(val_features, max_false_alarm_rate)
    log(f"Calibrated on {len(val_features)} validation windows in {time.time() - t0:.1f}s, "
       f"threshold={model.threshold:.4f}")

    rows = []
    for k, held_out in folds.items():
        log(f"Fold {k} (reporting partition only -- same frozen model, not retrained):")
        for name in held_out:
            entry = road.manifest.entry(name)
            f_capture, l_capture, _ = _examples(road, v2, [name], max_windows_per_capture)
            t0 = time.time()
            scores = model.predict_proba_attack(f_capture) if f_capture else []
            elapsed = time.time() - t0
            detected = sum(1 for s, y in zip(scores, l_capture) if y and s >= model.threshold)
            false_alarms = sum(1 for s, y in zip(scores, l_capture) if not y and s >= model.threshold)
            attacked = sum(1 for y in l_capture if y)
            normal = sum(1 for y in l_capture if not y)
            rows.append({"capture_id": entry.capture_id, "family": entry.family,
                        "kind": entry.kind, "fold": k, "attacked": attacked,
                        "normal": normal, "detected": detected, "false_alarms": false_alarms})
            log(f"  {entry.capture_id} ({entry.family}/{entry.kind}): "
               f"{detected}/{attacked} detected, {false_alarms}/{normal} false alarms "
               f"({elapsed:.1f}s for {len(f_capture)} windows)")

    total_normal = sum(r["normal"] for r in rows)
    total_alarms = sum(r["false_alarms"] for r in rows)
    total_attacked = sum(r["attacked"] for r in rows)
    total_detected = sum(r["detected"] for r in rows)
    return {
        "protocol": "DEVELOPMENT-only evaluation, DefenderLLM (frozen zero-shot local LLM "
                    "over Stage1+Stage2's raw signals; 'fold' is a reporting partition only)",
        "model_name": model.model_name,
        "features": list(FEATURES),
        "max_false_alarm_rate": max_false_alarm_rate,
        "max_windows_per_capture": max_windows_per_capture,
        "calibration_windows": model.calibration_windows,
        "threshold": model.threshold,
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
        description="DefenderLLM dev-only evaluation (never reads final_test)."
    )
    parser.add_argument("--data-dir", default="data/road")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--max-false-alarm-rate", type=float, default=0.01)
    parser.add_argument("--v2-model-dir", default="models",
                        help="where v2's ALREADY-FITTED stage1/stage2 live -- borrowed "
                             "here purely as a feature extractor, never refit")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME,
                        help="local Hugging Face causal LM to load")
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--max-windows-per-capture", type=int, default=100_000,
                        help="cap windows sampled per capture -- use a small number "
                             "(e.g. 5) for a fast smoke-scale dry run before the full set")
    parser.add_argument("--out", default="results/crossval_defender_llm.json")
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    road = RoadData(args.data_dir, manifest=manifest)
    v2 = Defender.load(args.v2_model_dir, "v2")

    print(f"Loading {args.model_name} locally (first run downloads the weights)...")
    model = DefenderLLMModel.load_local(args.model_name, args.max_new_tokens)

    report = cross_validate(road, v2, model, args.max_false_alarm_rate,
                            args.max_windows_per_capture)
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
