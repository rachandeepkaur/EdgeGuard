"""
Builds dashboard/src/data/realRun.json: a real EdgeGuard v2 Defender replay
for the dashboard's "live" data source (see dashboard/src/data/liveFeed.js
and README.md's "Connecting the real feed").

This is REAL ROAD data, REAL v2 inference, REAL evidence text and REAL
per-window latency -- not synthetic mock data. It concatenates four real
DEVELOPMENT attack captures (never final_test/separate) in their own
chronological order, one after another, purely to give the demo replay a
few distinct attack bursts the way defender/dev_check.py's own numbers show
they naturally occur (each capture already mixes normal driving with an
injected attack in the middle).

The last capture is a deliberate HARD CASE: reverse_light_off_attack_1,
which v2 misses entirely (0 / 8 attack windows, see CLAUDE.md's development
check). Showing a known weakness is more honest than a replay of only the
attacks v2 catches -- but a missed attack looks exactly like normal traffic
from the Defender's output alone, so each window also carries a separate
"truth" field ("attack" / "normal") taken from Part 1's private
GroundTruthLabel. That field is EVALUATOR-side only: it is read after
score_window() returns, is never passed to the Defender, and is not part of
shared/schemas.py's DefenderOutput. The dashboard uses it only to mark
misses and false alarms, never to change a decision.

The highway drives (the other obvious hard case, v2's false alarms) are
final_test captures, which RoadData refuses outside the one-time final
evaluation, so they cannot appear here.

Usage (from the EdgeGuard repo root):
    python -m dashboard.scripts.build_real_run --data-dir ~/Downloads/road
"""

import argparse
import json
import platform
import sys
from pathlib import Path

# Run from the repo root; add it to sys.path so `defender`/`part1` import
# whether this is invoked as -m or as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from defender.defender import Defender
from integration.run_demo import calibrate_escalation_band
from part1.pipeline import RoadData
from part1.split_manifest import MANIFEST_PATH, load_manifest

# Same budget used in the README/CLAUDE.md's real escalation runs and in
# integration.run_demo's --max-escalation-rate examples.
MAX_ESCALATION_RATE = 0.3

# Real development captures, in the order they'll play. Each already mixes
# normal driving with one injected attack burst (see defender/dev_check.py's
# own real-data numbers: 22/6, 25/63, 38/34, 8/20 attacked/normal windows).
# The last one is the hard case v2 misses (see the module docstring).
CAPTURES = [
    "correlated_signal_attack_2",
    "max_speedometer_attack_1",
    "reverse_light_on_attack_2",
    "reverse_light_off_attack_1",
]
MODEL_VERSION = "v2"


def build(data_dir, manifest_path, model_dir):
    defender = Defender.load(model_dir, MODEL_VERSION)
    road = RoadData(data_dir, manifest=load_manifest(manifest_path), window_s=defender.window_s or 1.0)

    # Real band, calibrated on validation scores only -- never on the
    # development windows this script replays (see part1.escalation_policy
    # and integration.run_demo.calibrate_escalation_band). This lets the
    # dashboard show, per window, whether it would fall inside the real
    # escalation band, without duplicating the calibration logic.
    band_half_width = calibrate_escalation_band(
        defender, data_dir, manifest_path, MAX_ESCALATION_RATE)

    names = road.manifest.names("development")
    missing = [c for c in CAPTURES if c not in names]
    if missing:
        raise ValueError(f"not development captures per the manifest: {missing}")

    run = []
    for name in CAPTURES:
        for window, label in road.labelled_windows(name):
            output = defender.score_window(window)   # the label never reaches the Defender
            run.append({
                "window_id": output.window_id,
                "attack_score": round(output.attack_score, 4),
                "threshold": output.threshold,
                "decision": output.decision,
                "evidence": output.evidence,
                "model_version": output.model_version,
                "latency_ms": round(output.latency_ms, 2),
                # Evaluator-side only, added after scoring (see module docstring).
                "truth": "attack" if label.is_attack else "normal",
            })

    meta = {
        "captureId": road.capture_id(CAPTURES[0]),
        "captures": [road.capture_id(c) for c in CAPTURES],
        "modelVersion": MODEL_VERSION,
        "threshold": defender.threshold,
        "escalationBandHalfWidth": band_half_width,
        "maxEscalationRate": MAX_ESCALATION_RATE,
        # latency_ms is inference time on whichever machine ran this script.
        "latencyMeasuredOn": f"{platform.node()} ({platform.machine()})",
        "source": "real ROAD development captures, real v2 Defender inference "
                  "(build_real_run.py) -- not mock data",
    }
    return run, meta


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/road")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--out", default="dashboard/src/data/realRun.json")
    args = parser.parse_args(argv)

    run, meta = build(args.data_dir, args.manifest, args.model_dir)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"run": run, "meta": meta}, indent=2), encoding="utf-8")

    attacked = sum(1 for r in run if r["decision"] == "ATTACK")
    missed = sum(1 for r in run if r["truth"] == "attack" and r["decision"] == "ACCEPT")
    false_alarms = sum(1 for r in run if r["truth"] == "normal" and r["decision"] == "ATTACK")
    print(f"Wrote {len(run)} real windows ({attacked} flagged ATTACK, {missed} missed "
         f"attacks, {false_alarms} false alarms) from {', '.join(CAPTURES)} to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
