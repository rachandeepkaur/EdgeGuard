"""
Block 5 evidence, fused-detector variant: the ONE-TIME held-out comparison
of the already-shipped v2 against FusedV2TimingCNNDefender (v2 OR
TimingCNN) on final_test.

This is a SECOND entry point into integration.run_final_evaluation's
run_final_evaluation() -- the exact same function already used for the
v1-vs-v2 report -- never a reimplementation of it. That module's docstring
invariant ("every call to RoadData with final_evaluation=True in this whole
codebase lives HERE and nowhere else") is about the call site inside
run_final_evaluation() itself; this script never touches RoadData directly,
so the invariant still holds with this script added.

v2 (defender.defender.Defender) and FusedV2TimingCNNDefender both expose
the same score_window(window) -> DefenderOutput contract, so
run_final_evaluation(baseline=v2, updated=fused, ...) works with NO changes
to that function -- baseline/updated were never typed as "must literally be
class Defender", only as "anything with .score_window()".

Only run this once real dev-CV evidence supports it.
results/crossval_fused_timing_cnn.json already shows the fused detector at
76.4% recall vs v2 alone at 63.3%, with the SAME false-alarm count (1/237)
-- gains concentrated on v2's weakest families (reverse_light_off,
reverse_light_on), v2's own fuzzing/masquerade coverage fully preserved
because it's OR'd in, not replaced. Both v2 and the TimingCNN checkpoint
this script loads are already frozen (defender/train_timing_cnn.py's
one-time fit on train+validation+development) -- nothing here re-trains or
re-thresholds anything.

Usage:
    .venv/bin/python3 -m integration.run_final_evaluation_fused \
        --data-dir data/road --model-dir models \
        --cnn-model-dir models/timing_cnn_v1
"""

import argparse
import json
from pathlib import Path

from defender.defender import Defender
from defender.fused_defender import FusedV2TimingCNNDefender
from defender.timing_cnn import TimingCNNModel
from integration.run_final_evaluation import run_final_evaluation
from part1.split_manifest import MANIFEST_PATH


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Block 5 (fused variant): the ONE-TIME v2-vs-fused comparison on final_test."
    )
    parser.add_argument("--data-dir", default="data/road",
                        help="ROAD folder with ambient/ attacks/ (default data/road)")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--model-dir", default="models",
                        help="directory holding the frozen v2 Stage1/Stage2 models")
    parser.add_argument("--cnn-model-dir", default="models/timing_cnn_v1",
                        help="directory holding the frozen TimingCNN checkpoint "
                             "(defender/train_timing_cnn.py's output)")
    parser.add_argument("--watch-ids", default="0D0,6E0",
                        help="must match the watch-ids TimingCNN was trained with")
    parser.add_argument("--output", default="results/final_evaluation_fused.json")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing final evaluation report")
    args = parser.parse_args(argv)

    output_path = Path(args.output)
    if output_path.exists() and not args.force:
        parser.error(
            f"{output_path} already exists -- this is the ONE-TIME final comparison "
            "(build plan S6/S9). If you really mean to redo it, pass --force and say "
            "why in your commit message."
        )

    v2 = Defender.load(args.model_dir, "v2")
    cnn = TimingCNNModel.load(args.cnn_model_dir)
    watch_ids = tuple(i.strip().upper() for i in args.watch_ids.split(",") if i.strip())
    fused = FusedV2TimingCNNDefender(v2, cnn, watch_ids=watch_ids)

    report = run_final_evaluation(v2, fused, args.data_dir, args.manifest)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")

    print("=== Block 5 (fused variant): FINAL evaluation (final_test, read once) ===")
    print(f"Captures ({len(report.final_test_captures)}): "
         f"{', '.join(report.final_test_captures)}")
    print(f"Windows: {report.windows}")
    print(f"\n{report.baseline_version} (already-shipped v2, baseline): {report.baseline}")
    print(f"  recall by family: {report.baseline_recall_by_family}")
    if report.baseline_detection_delay:
        print(f"  detection delay: {report.baseline_detection_delay}")
    print(f"\n{report.updated_version} (v2 OR TimingCNN, updated): {report.updated}")
    print(f"  recall by family: {report.updated_recall_by_family}")
    if report.updated_detection_delay:
        print(f"  detection delay: {report.updated_detection_delay}")
    if report.updated_per_capture:
        print("\nPer-capture (fused model):")
        for capture_id, m in report.updated_per_capture.items():
            print(f"  {capture_id}: tp={m['tp']} fp={m['fp']} tn={m['tn']} fn={m['fn']}")
    print(f"\nSaved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
