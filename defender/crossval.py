"""
Cross-validation for the EdgeGuard Defender (Part 2).

Two protocols, both grouped by CAPTURE, never by window (neighbouring windows
of one drive are near-copies, so a window-level split would leak). Captures,
groups and labels all come from Part 1: the frozen split manifest and
part1.pipeline.RoadData.

1. ambient -- leave-one-capture-out (LOCO) false-alarm estimate.
   Pool = train + validation normal drives. For each held-out capture H:
     a. inner LOCO over the rest: fit on (rest minus J), score J. This gives
        OUT-OF-FOLD scores for every capture in the rest, and the threshold
        is chosen on those (choose_threshold, same false-alarm target).
        Never on in-sample scores: held-out drives score higher than the
        drives a model was fitted on.
     b. fit on all of the rest, score H, count false alarms at that threshold.
   It answers: "trained and calibrated our way, how often does the Defender
   false-alarm on a drive it has never seen?" v1 (Stage 1) and v2 (Stage 1 +
   Stage 2, fused) are both measured from the same fits.

2. attacks -- 2-fold CV over the DEVELOPMENT attack recordings. The manifest
   gives each one a fold (its index: _1 or _2). The Stage 2 watch-list comes
   from the target IDs of the OTHER fold's attacks only. Stage 1 and Stage 2
   are fitted on train drives and the threshold on validation drives, exactly
   as run_training does. Every window of the held-out fold is scored and
   labelled by Part 1 (attacked = holds an injected frame).
   A masquerade capture is byte-identical to its fabrication twin outside the
   injection interval, so it contributes only its attacked windows; normal
   windows are counted once, from the fabrication capture.

What this is NOT: a replacement for the frozen-model final-test number. It
estimates the PROCEDURE (each fold has its own models and threshold). Use it
to choose settings, report it beside the final-test number, and never tune
on the final test. final_test / separate captures are never read: RoadData
refuses them.

Usage (from the EdgeGuard folder, .venv active):
    python -m defender.crossval ambient --max-false-alarm-rate 0.01 \\
        --data-dir ~/Downloads/road/dataset
    python -m defender.crossval attacks --max-false-alarm-rate 0.01 \\
        --data-dir ~/Downloads/road/dataset
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

from defender.fusion import fuse
from defender.stage1 import Stage1Model
from defender.stage2 import Stage2Model, normalize_can_id
from defender.threshold import choose_threshold
from defender.run_training import DEVELOPMENT_WATCH_IDS
from part1.labels import injection_rule
from part1.pipeline import RoadData
from part1.split_manifest import MANIFEST_PATH, load_manifest

VERSIONS = ("v1", "v2")   # v1 = Stage 1 score, v2 = Stage 1 + Stage 2 fused
CONDITIONS = ("fabrication", "masquerade", "fuzzing")


# ---------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------
def binomial_upper_95(k: int, n: int) -> float:
    """Exact one-sided 95 % upper bound (Clopper-Pearson) on a rate k / n.
    Windows of one capture are correlated, so treat it as optimistic."""
    if n <= 0 or k >= n:
        return 1.0

    def cdf(p):   # P(X <= k), X ~ Binomial(n, p), in log space
        log_p, log_q = math.log(p), math.log1p(-p)
        return sum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                            + i * log_p + (n - i) * log_q) for i in range(k + 1))

    lo, hi = k / n, 1.0 - 1e-12
    for _ in range(100):
        mid = (lo + hi) / 2
        if cdf(mid) > 0.05:
            lo = mid
        else:
            hi = mid
    return hi


def _summary(alarms: int, windows: int, target: float, window_s: float) -> dict:
    hours = windows * window_s / 3600
    return {"alarms": alarms, "windows": windows,
            "false_alarm_rate": round(alarms / windows, 6) if windows else None,
            "false_alarm_rate_95_upper": round(binomial_upper_95(alarms, windows), 6),
            "normal_hours": round(hours, 4),
            "false_alarms_per_hour": round(alarms / hours, 3) if hours else None,
            "target": target}


# ---------------------------------------------------------------------
# Data loading (each capture read once, reused by every fold)
# ---------------------------------------------------------------------
def load_ambient(road: RoadData, names, max_windows, stats_watch, log=print):
    """Sampled windows and Stage 2 full-capture range stats per capture."""
    probe = Stage2Model(watch_ids=stats_watch)
    windows, stats = {}, {}
    for name in names:
        step = road.keep_every(name, max_windows)
        windows[name] = list(road.windows(name, keep_every=step))
        stats[name] = probe.capture_range_stats(road.frames(name))
        log(f"  {road.capture_id(name)}: {len(windows[name])} windows (every {step}th)")
    return windows, stats


def _only_watched(stats, watch):
    """Restrict range stats to a watch-list (keys are 'ID|field')."""
    if watch is None:
        return stats
    keep = set(watch)
    return tuple({k: v for k, v in part.items() if k.split("|", 1)[0] in keep}
                 for part in stats)


def fit_models(names, windows, stats, watch, frozen_mode, bound_percentile=None):
    """Stage 1 and Stage 2 fitted on the named captures only."""
    train = [w for n in names for w in windows[n]]
    stage1 = Stage1Model().fit(train)
    stage2 = Stage2Model(watch_ids=watch, frozen_mode=frozen_mode,
                         bound_percentile=bound_percentile).fit(
        train, range_stats=[_only_watched(stats[n], watch) for n in names])
    return stage1, stage2


def score_both(stage1, stage2, windows) -> Dict[str, List[float]]:
    """v1 (Stage 1) and v2 (fused) scores for each window."""
    scores = {"v1": [], "v2": []}
    for window in windows:
        first = stage1.score(window)
        scores["v1"].append(first.score)
        scores["v2"].append(fuse(first, stage2.score(window)).score)
    return scores


# ---------------------------------------------------------------------
# Protocol 1: leave-one-capture-out over normal drives
# ---------------------------------------------------------------------
def ambient_loco(names, windows, stats, max_false_alarm_rate, watch, frozen_mode="width",
                 bound_percentile=None, window_s=1.0, log=print) -> dict:
    """Nested LOCO: out-of-fold threshold on the rest, false alarms on the held-out one."""
    if len(names) < 3:
        raise ValueError(f"leave-one-out needs at least 3 captures, got {len(names)}")
    rows = []
    for held_out in names:
        rest = [n for n in names if n != held_out]
        oof = {v: [] for v in VERSIONS}
        for inner_out in rest:
            models = fit_models([n for n in rest if n != inner_out], windows, stats,
                                watch, frozen_mode, bound_percentile)
            for v, s in score_both(*models, windows[inner_out]).items():
                oof[v].extend(s)
        thresholds = {v: choose_threshold(oof[v], max_false_alarm_rate) for v in VERSIONS}
        scores = score_both(*fit_models(rest, windows, stats, watch, frozen_mode, bound_percentile),
                            windows[held_out])
        row = {"capture_id": windows[held_out][0].capture_id if windows[held_out] else None,
               "windows": len(windows[held_out])}
        for v in VERSIONS:
            row[f"{v}_threshold"] = thresholds[v]
            row[f"{v}_alarms"] = sum(1 for s in scores[v] if s >= thresholds[v])
        rows.append(row)
        log(f"  held out {row['capture_id']}: v1 {row['v1_alarms']}/{row['windows']}, "
            f"v2 {row['v2_alarms']}/{row['windows']} false alarms")

    total = sum(r["windows"] for r in rows)
    summary = {}
    for v in VERSIONS:
        alarms = sum(r[f"{v}_alarms"] for r in rows)
        summary[v] = {**_summary(alarms, total, max_false_alarm_rate, window_s),
                      "captures_over_target": sum(
                          1 for r in rows
                          if r["windows"] and r[f"{v}_alarms"] / r["windows"] > max_false_alarm_rate)}
    return {"protocol": "normal-drive leave-one-capture-out, out-of-fold threshold",
            "captures": len(rows), "per_capture": rows, "summary": summary}


# ---------------------------------------------------------------------
# Protocol 2: 2-fold CV over development attack recordings
# ---------------------------------------------------------------------
def fold_watch_ids(road: RoadData, train_attacks) -> List[str]:
    """Stage 2 watch-list: target IDs of the TRAINING fold's attacks only
    (fuzzing and accelerator captures have no single target)."""
    ids = {injection_rule(road.manifest.entry(n), road.road_metadata).target
           for n in train_attacks}
    ids.discard(None)
    if not ids:
        raise ValueError("training fold has no targeted attacks: cannot build a watch-list")
    return sorted(ids)


def attack_cv(road: RoadData, max_false_alarm_rate, max_windows, frozen_mode="width",
              bound_percentile=None, log=print) -> dict:
    folds = road.manifest.development_folds()
    if len(folds) < 2:
        raise ValueError(f"attack CV needs at least 2 development folds, got {sorted(folds)}")
    watches = {k: fold_watch_ids(road, [n for f, c in folds.items() if f != k for n in c])
               for k in folds}
    union = sorted({i for w in watches.values() for i in w})

    train_names = road.manifest.names("train")
    val_names = road.manifest.names("validation")
    log("Reading train + validation normal drives:")
    windows, stats = load_ambient(road, train_names + val_names, max_windows, union, log)
    validation = [w for n in val_names for w in windows[n]]

    fold_rows, capture_rows = [], []
    for k, held_out in folds.items():
        stage1, stage2 = fit_models(train_names, windows, stats, watches[k], frozen_mode,
                                    bound_percentile)
        val_scores = score_both(stage1, stage2, validation)
        thresholds = {v: choose_threshold(val_scores[v], max_false_alarm_rate) for v in VERSIONS}
        fold_rows.append({"fold": k, "watch_ids": watches[k], "held_out": len(held_out),
                          **{f"{v}_threshold": thresholds[v] for v in VERSIONS}})
        log(f"Fold {k}: watch-list {watches[k]} (from the other fold), "
            f"holding out {len(held_out)} captures")
        for name in held_out:
            entry = road.manifest.entry(name)
            row = {"capture_id": entry.capture_id, "fold": k, "family": entry.family,
                   "condition": entry.kind, "attacked": 0, "normal": 0,
                   **{f"{v}_{c}": 0 for v in VERSIONS for c in ("detected", "false_alarms")}}
            for window, label in road.labelled_windows(name):
                if entry.kind == "masquerade" and not label.is_attack:
                    continue      # byte-identical to the fabrication twin: count once
                scores = score_both(stage1, stage2, [window])
                row["attacked" if label.is_attack else "normal"] += 1
                for v in VERSIONS:
                    if scores[v][0] >= thresholds[v]:
                        row[f"{v}_detected" if label.is_attack else f"{v}_false_alarms"] += 1
            capture_rows.append(row)
            log(f"  {name}: v1 {row['v1_detected']}/{row['attacked']}, "
                f"v2 {row['v2_detected']}/{row['attacked']} attacked windows detected")

    return {"protocol": "2-fold CV over development attack recordings (fold = index)",
            "folds": fold_rows, "per_capture": capture_rows,
            "summary": _attack_summary(capture_rows, max_false_alarm_rate, road.window_s)}


def _attack_summary(rows, target, window_s) -> dict:
    """Per-family recall, macro-averaged over families (never pooled: one
    family would dominate a pooled number), plus false alarms on normal windows."""
    out = {}
    for v in VERSIONS:
        per_condition = {}
        for condition in CONDITIONS + ("all",):
            recalls = {}
            for family in sorted({r["family"] for r in rows}):
                chosen = [r for r in rows if r["family"] == family
                          and (condition == "all" or r["condition"] == condition)]
                attacked = sum(r["attacked"] for r in chosen)
                if attacked:
                    recalls[family] = round(sum(r[f"{v}_detected"] for r in chosen) / attacked, 4)
            per_condition[condition] = {
                "per_family_recall": recalls,
                "macro_recall": round(sum(recalls.values()) / len(recalls), 4) if recalls else None}
        normal = sum(r["normal"] for r in rows)
        alarms = sum(r[f"{v}_false_alarms"] for r in rows)
        out[v] = {**per_condition,
                  "normal_windows_in_attack_captures": _summary(alarms, normal, target, window_s)}
    return out


# ---------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Defender cross-validation (never reads test).")
    parser.add_argument("protocol", choices=("ambient", "attacks"))
    parser.add_argument("--max-false-alarm-rate", type=float, required=True,
                        help="team decision, e.g. 0.01 (no default)")
    parser.add_argument("--data-dir", default="data/road", help="ROAD folder with ambient/ attacks/")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH), help="frozen split manifest")
    parser.add_argument("--window-s", type=float, default=1.0)
    parser.add_argument("--max-windows-per-capture", type=int, default=100,
                        help="normal-drive windows per capture, same default as run_training")
    parser.add_argument("--stage2-watch", default=",".join(DEVELOPMENT_WATCH_IDS),
                        help='ambient protocol only: watch-list, e.g. "0D0,6E0", or "all"')
    parser.add_argument("--frozen-mode", default="width", choices=("width", "rate"))
    parser.add_argument("--bound-percentile", type=float, default=None,
                        help="Stage 2 out-of-range bounds as a percentile, e.g. 1.0 to "
                             "trim the extreme 1%% off each side (default: strict min/max)")
    parser.add_argument("--out", help="JSON report path (default results/crossval_<protocol>.json)")
    args = parser.parse_args(argv)
    if args.max_windows_per_capture < 1:
        parser.error("--max-windows-per-capture must be at least 1")

    road = RoadData(args.data_dir, manifest=load_manifest(args.manifest), window_s=args.window_s)
    settings = {"manifest": str(args.manifest), "window_s": args.window_s,
                "max_windows_per_capture": args.max_windows_per_capture,
                "max_false_alarm_rate": args.max_false_alarm_rate,
                "frozen_mode": args.frozen_mode,
                "bound_percentile": args.bound_percentile}

    if args.protocol == "ambient":
        watch = None if args.stage2_watch.strip().lower() == "all" else \
            sorted({normalize_can_id(i) for i in args.stage2_watch.split(",") if i.strip()})
        names = road.manifest.names("train", "validation")
        print(f"Reading {len(names)} normal drives:")
        windows, stats = load_ambient(road, names, args.max_windows_per_capture, watch)
        print(f"Leave-one-capture-out ({len(names)} outer folds, "
              f"{len(names) * (len(names) - 1)} inner fits):")
        report = ambient_loco(names, windows, stats, args.max_false_alarm_rate, watch,
                              args.frozen_mode, args.bound_percentile, args.window_s)
        settings["stage2_watch_ids"] = watch
    else:
        report = attack_cv(road, args.max_false_alarm_rate, args.max_windows_per_capture,
                           args.frozen_mode, args.bound_percentile)

    report = {"settings": settings, **report,
              "note": "Procedure-level estimate: each fold has its own models and threshold. "
                      "Use it to choose settings; report it beside the frozen-model "
                      "final-test number, never instead of it."}
    out = Path(args.out) if args.out else Path("results") / f"crossval_{args.protocol}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== Cross-validation summary (NOT the evidence gate) ===")
    for v in VERSIONS:
        s = report["summary"][v]
        if args.protocol == "ambient":
            print(f"  {v}: false alarms {s['alarms']}/{s['windows']} = {s['false_alarm_rate']} "
                  f"(95% upper {s['false_alarm_rate_95_upper']}, target {s['target']}), "
                  f"{s['false_alarms_per_hour']}/hour, "
                  f"{s['captures_over_target']}/{report['captures']} captures over target")
        else:
            fa = s["normal_windows_in_attack_captures"]
            print(f"  {v}: macro recall fabrication {s['fabrication']['macro_recall']}, "
                  f"masquerade {s['masquerade']['macro_recall']}, "
                  f"fuzzing {s['fuzzing']['macro_recall']}; "
                  f"false alarms {fa['alarms']}/{fa['windows']}")
    print(f"Report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
