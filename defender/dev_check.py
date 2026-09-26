"""
DEVELOPMENT sanity check: v1 vs v2 (vs v3, if trained) on the development
attack captures of the frozen split manifest.

NOT the evidence gate. Final-test captures are never read here. Part 3's
evaluator and evidence gate produce the official results.

Every window of every development attack capture is scored (no sampling),
and labelled by Part 1 (part1.labels): attacked = it holds at least one
injected frame. A masquerade capture is byte-identical to its fabrication
twin outside the injection interval, so it contributes only its attacked
windows; normal windows are counted once, from the fabrication capture.

It also re-creates Part 3's confirmed development miss (all 0D0 frames of
the first ambient_dyno_reverse window frozen at 3A710460F5000000) as a
regression check.

Usage (from the EdgeGuard folder):
    python -m defender.dev_check --data-dir ~/Downloads/road/dataset
"""

import argparse

from defender.defender import Defender, stage1_path
from part1.pipeline import RoadData
from part1.split_manifest import MANIFEST_PATH, load_manifest
from shared.schemas import TrafficWindow

# Part 3's confirmed development miss: 0D0 frozen (counter + checksum).
PART3_FREEZE_CAPTURE = "ambient_dyno_reverse"
PART3_FREEZE_PAYLOAD = "3A710460F5000000"


def freeze_id(window: TrafficWindow, can_id: str, payload: str) -> TrafficWindow:
    """COPY of the window with every frame of can_id set to one payload."""
    data = window.model_dump()
    data["window_id"] = window.window_id + "_v01"
    for frame in data["frames"]:
        if frame["can_id"].upper() == can_id:
            frame["payload"] = payload
    return TrafficWindow(**data)


def check_capture(road: RoadData, name: str, defenders) -> dict:
    """Counts per defender: attacked windows detected, normal windows flagged."""
    masquerade = road.manifest.entry(name).kind == "masquerade"
    counts = {v: {"attacked": 0, "detected": 0, "normal": 0, "false_alarms": 0}
              for v in defenders}
    for window, label in road.labelled_windows(name):
        if masquerade and not label.is_attack:
            continue      # byte-identical to the fabrication twin: counted there
        for version, defender in defenders.items():
            flagged = defender.score_window(window).decision == "ATTACK"
            c = counts[version]
            if label.is_attack:
                c["attacked"] += 1
                c["detected"] += flagged
            else:
                c["normal"] += 1
                c["false_alarms"] += flagged
    return counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="DEVELOPMENT sanity check: v1 vs v2 (vs v3).")
    parser.add_argument("--data-dir", default="data/road")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args(argv)

    versions = [v for v in ("v1", "v2", "v3") if stage1_path(args.model_dir, v).exists()]
    print(f"Loading {', '.join(versions)} ...", flush=True)
    defenders = {v: Defender.load(args.model_dir, v) for v in versions}
    window_s = defenders[versions[0]].window_s or 1.0
    road = RoadData(args.data_dir, manifest=load_manifest(args.manifest), window_s=window_s)

    rows = []
    for name in road.manifest.names("development", folder="attacks"):
        print(f"  checking {name} ...", flush=True)
        rows.append((name, check_capture(road, name, defenders)))

    print("\n=== DEVELOPMENT sanity check (NOT the evidence gate) ===")
    header = f"{'capture':42s} {'attacked':>8s}"
    for v in versions:
        header += f" {v + ' det':>7s}"
    header += f" {'normal':>7s}"
    for v in versions:
        header += f" {v + ' FA':>6s}"
    print(header)
    totals = {v: [0, 0, 0, 0] for v in versions}
    for name, c in rows:
        first = c[versions[0]]
        line = f"{name:42s} {first['attacked']:8d}"
        for v in versions:
            line += f" {c[v]['detected']:7d}"
        line += f" {first['normal']:7d}"
        for v in versions:
            line += f" {c[v]['false_alarms']:6d}"
        print(line)
        for v in versions:
            t = totals[v]
            t[0] += c[v]["detected"]; t[1] += c[v]["attacked"]
            t[2] += c[v]["false_alarms"]; t[3] += c[v]["normal"]
    print("Totals:")
    for v in versions:
        d, a, f, n = totals[v]
        print(f"  {v}: detected {d}/{a} attacked windows, false alarms {f}/{n} normal windows")

    # Regression check: Part 3's confirmed development miss.
    original = next(road.windows(PART3_FREEZE_CAPTURE))
    frozen = freeze_id(original, "0D0", PART3_FREEZE_PAYLOAD)
    print(f"\n=== Part 3 freeze repro (0D0 frozen at {PART3_FREEZE_PAYLOAD}) ===")
    for v, defender in defenders.items():
        a = defender.score_window(original)
        b = defender.score_window(frozen)
        print(f"  {v}: original {a.decision} ({a.attack_score:.6f})  "
              f"frozen {b.decision} ({b.attack_score:.6f})  {b.evidence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
