"""Turn a real evasion log into a real HardeningSet, and report which Stage 2
watch-list gaps the confirmed misses expose.

Architecture doc section on adversarial testing and hardening: "Fleet window
-> Red Team proposal -> constrained injector -> Blue Team score -> ground
truth comparison -> confirmed misses -> candidate update -> held-out
evaluation." part3.evasion_log and part3.hardening_set implement "confirmed
misses", but until this module nothing outside hardening_set's own test file
ever called build_hardening_set() on a real evasion log, so the loop from
"confirmed misses" to "candidate update" had never actually been closed.

This module closes it: it loads a real evasion log written by
integration.run_demo's test path, builds the real HardeningSet, and breaks
the misses down by family and by whether Stage 2 was even watching the
target CAN ID -- the split that separates a genuine detector weakness from
a Stage 2 coverage gap defender/run_training.py's --stage2-watch can close.

Usage (from the EdgeGuard folder, after a real evasion log exists):
    python -m part3.hardening_report /tmp/evasions_v2.jsonl \\
        --data-dir ~/Downloads/road --baseline-version v2 \\
        --watch-ids 0D0,6E0
"""

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

from integration.run_demo import ATTACK_FAMILIES
from part1.pipeline import RoadData
from part1.split_manifest import MANIFEST_PATH, load_manifest
from part3.evasion_log import EvasionEntry
from part3.hardening_set import HardeningSet, build_hardening_set


def load_evasions(path: Union[str, Path]) -> List[EvasionEntry]:
    """Read a JSONL evasion log written by integration.run_demo's test path."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [EvasionEntry(**json.loads(line)) for line in lines if line.strip()]


def development_window_variant_ids(road: RoadData,
                                    families: Sequence[str] = ATTACK_FAMILIES) -> List[str]:
    """The allow-list build_hardening_set() checks evasions against.

    integration.run_demo's test path assigns variant_index by
    enumerate(families, start=1) and part3.attack_injector.inject() suffixes
    each attacked window's id with "_v{variant_index:02d}" -- so a raw
    development window_id is never what appears in the evasion log; its
    per-family variant ids are. Reconstruct the same suffixes here rather
    than guessing them, so a change to ATTACK_FAMILIES's order is caught by
    build_hardening_set()'s own "not in the development manifest" error
    instead of silently mis-matching.
    """
    ids = []
    for name in road.manifest.names("development"):
        for window in road.windows(name):
            for variant_index in range(1, len(families) + 1):
                ids.append(f"{window.window_id}_v{variant_index:02d}")
    return ids


def build_real_hardening_set(evasions: Iterable[EvasionEntry], road: RoadData, *,
                             baseline_model_version: str,
                             families: Sequence[str] = ATTACK_FAMILIES) -> HardeningSet:
    """build_hardening_set(), wired to a real evasion log and a real manifest."""
    allow_list = development_window_variant_ids(road, families)
    return build_hardening_set(evasions, allow_list, baseline_model_version=baseline_model_version)


def diagnose(evasions: Sequence[EvasionEntry], watch_ids: Optional[Sequence[str]]) -> dict:
    """Break confirmed misses down by family and by Stage 2 watch coverage.

    watch_ids=None means Stage 2 watched every CAN ID (--stage2-watch all):
    every miss then counts as "watched", so this only ever attributes misses
    to a genuine detector-sensitivity gap, never a coverage gap.
    """
    watch = None if watch_ids is None else {w.upper() for w in watch_ids}
    by_family = Counter(e.family for e in evasions)
    by_target = Counter(e.target_can_id for e in evasions)
    watched = [e for e in evasions if watch is None or e.target_can_id in watch]
    unwatched = [e for e in evasions if watch is not None and e.target_can_id not in watch]
    return {
        "total_misses": len(evasions),
        "distinct_windows": len({e.window_id for e in evasions}),
        "by_family": dict(by_family),
        "by_target_can_id": dict(by_target),
        "watch_ids": sorted(watch) if watch is not None else "all",
        "coverage_gap_misses": len(unwatched),
        "coverage_gap_targets": sorted({e.target_can_id for e in unwatched}),
        "sensitivity_gap_misses": len(watched),
        "sensitivity_gap_by_family": dict(Counter(e.family for e in watched)),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evasion_log", help="JSONL path from --evasion-log")
    parser.add_argument("--data-dir", default="data/road")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--baseline-version", required=True,
                        help="model_version the evasion log was recorded against")
    parser.add_argument("--watch-ids", default=None,
                        help='comma-separated Stage 2 watch-list the baseline model used, '
                             'e.g. "0D0,6E0", or omit for "all"')
    parser.add_argument("--out", help="optional JSON report path")
    args = parser.parse_args(argv)

    evasions = load_evasions(args.evasion_log)
    wrong_version = sorted({e.model_version for e in evasions} - {args.baseline_version})
    if wrong_version:
        parser.error(f"evasion log has model_version(s) {wrong_version}, expected "
                    f"only {args.baseline_version!r}")

    road = RoadData(args.data_dir, manifest=load_manifest(args.manifest))
    hardening_set = build_real_hardening_set(
        evasions, road, baseline_model_version=args.baseline_version)
    watch_ids = args.watch_ids.split(",") if args.watch_ids else None
    report = diagnose(evasions, watch_ids)
    report["hardening_set"] = asdict(hardening_set)

    print(f"HardeningSet: baseline={hardening_set.baseline_model_version}, "
          f"{len(hardening_set.window_ids)} distinct missed windows")
    print(f"By family: {report['by_family']}")
    print(f"By target CAN ID: {report['by_target_can_id']}")
    print(f"Coverage gap (target not on watch-list {report['watch_ids']}): "
          f"{report['coverage_gap_misses']} misses on {report['coverage_gap_targets']}")
    print(f"Sensitivity gap (target WAS watched, still missed): "
          f"{report['sensitivity_gap_misses']} misses, by family "
          f"{report['sensitivity_gap_by_family']}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
