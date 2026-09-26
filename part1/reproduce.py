"""
Part 1 -- reproducibility check (build plan Block 1 completion check:
"normal windows and private labels reproducible").

For each capture it runs the whole Part 1 pipeline (read -> clean -> window
-> label) and hashes the result: every window (ids, bounds, frames) and every
label. The same data and code must always give the same hashes, on any
machine.

    # write a fingerprint
    python -m part1.reproduce --data-dir ~/Downloads/road/dataset \\
        --out results/part1_fingerprint.json
    # a teammate checks their copy against it
    python -m part1.reproduce --data-dir <their road> --check results/part1_fingerprint.json

The fingerprint holds only neutral ids and hashes, never names or labels, so
it is safe to commit. final_test / separate captures are skipped unless
--final-evaluation is passed.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

from part1.cleaning import CleaningReport
from part1.pipeline import RoadData


def capture_fingerprint(road: RoadData, name: str) -> dict:
    """Hashes of one capture's windows and labels, plus counts."""
    windows = hashlib.sha256()
    labels = hashlib.sha256()
    report = CleaningReport()
    n = attacks = 0
    for window, label in road.labelled_windows(name, report=report):
        windows.update(window.model_dump_json().encode("utf-8"))
        labels.update(label.model_dump_json().encode("utf-8"))
        n += 1
        attacks += label.is_attack
    return {"capture_id": road.capture_id(name), "windows": n, "attack_windows": attacks,
            "frames_kept": report.frames_kept, "duplicates_dropped": report.duplicates_dropped,
            "skipped_lines": report.skipped_lines,
            "windows_sha256": windows.hexdigest(), "labels_sha256": labels.hexdigest()}


def fingerprint(road: RoadData, names: Optional[Iterable[str]] = None, log=print) -> dict:
    if names is None:
        groups = ["train", "validation", "development"]
        if road.final_evaluation:
            groups += ["final_test", "separate"]
        names = road.manifest.names(*groups)
    rows = []
    for name in names:
        rows.append(capture_fingerprint(road, name))
        log(f"  {rows[-1]['capture_id']}: {rows[-1]['windows']} windows, "
            f"{rows[-1]['attack_windows']} attack")
    rows.sort(key=lambda r: r["capture_id"])
    return {"window_s": road.window_s, "vehicle_id": road.vehicle_id, "captures": rows}


def compare(expected: dict, actual: dict) -> list:
    """Differences between two fingerprints, as readable lines (empty = identical)."""
    diffs = []
    for key in ("window_s", "vehicle_id"):
        if expected.get(key) != actual.get(key):
            diffs.append(f"{key}: expected {expected.get(key)}, got {actual.get(key)}")
    want = {r["capture_id"]: r for r in expected["captures"]}
    got = {r["capture_id"]: r for r in actual["captures"]}
    for cid in sorted(set(want) | set(got)):
        if cid not in got:
            diffs.append(f"{cid}: missing")
        elif cid not in want:
            diffs.append(f"{cid}: not in the expected fingerprint")
        else:
            for key, value in want[cid].items():
                if got[cid].get(key) != value:
                    diffs.append(f"{cid} {key}: expected {value}, got {got[cid].get(key)}")
    return diffs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Part 1 reproducibility fingerprint.")
    parser.add_argument("--data-dir", required=True, help="ROAD folder with ambient/ attacks/")
    parser.add_argument("--out", help="write the fingerprint here")
    parser.add_argument("--check", help="compare against this fingerprint file")
    parser.add_argument("--capture", action="append", help="limit to these capture names")
    parser.add_argument("--final-evaluation", action="store_true",
                        help="also fingerprint final_test / separate captures")
    args = parser.parse_args(argv)
    if not args.out and not args.check:
        parser.error("pass --out, --check, or both")

    road = RoadData(args.data_dir, final_evaluation=args.final_evaluation)
    result = fingerprint(road, args.capture)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.out} ({len(result['captures'])} captures)")
    if args.check:
        expected = json.loads(Path(args.check).read_text(encoding="utf-8"))
        if args.capture:
            keep = {r["capture_id"] for r in result["captures"]}
            expected = {**expected, "captures": [r for r in expected["captures"]
                                                 if r["capture_id"] in keep]}
        diffs = compare(expected, result)
        if diffs:
            print("NOT reproducible:")
            for line in diffs:
                print(f"  {line}")
            return 1
        print(f"Reproducible: {len(result['captures'])} captures match {args.check}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
