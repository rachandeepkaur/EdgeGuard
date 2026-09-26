"""
Train Defender v1 and v2 on REAL ROAD normal drives (Part 2).

Usage (from the EdgeGuard folder, .venv active):
    python -m defender.run_training --max-false-alarm-rate 0.01 \\
        --data-dir ~/Downloads/road/dataset

What it does:
  1. Reads the FROZEN split manifest (part1/split_manifest.json): the train
     group fits the models, the validation group chooses the threshold.
  2. Gets every window from Part 1 (part1.pipeline.RoadData): cleaned frames,
     fixed non-overlapping 1 s windows, neutral capture and vehicle ids.
     At most --max-windows-per-capture evenly spaced windows are kept per
     capture so memory stays reasonable (about 1 MB per real window).
  3. Trains v1 (Stage 1) and v2 (v1's Stage 1 + new Stage 2) and saves them in
     --model-dir, then prints a summary. Stage 2 learns field ranges from
     EVERY cleaned frame of the train captures (streamed, low memory), because
     sampled windows miss rare-but-normal changes.

It NEVER reads attack captures and NEVER reads development or final_test
captures: only the train and validation normal drives.

--harden-v3 trains ONLY v3 (v1's Stage 1 + Stage 2 with the "rate" frozen
check), from the same captures and settings, and leaves v1 and v2 untouched.

--harden-v4 trains ONLY v4 (v1's Stage 1 + Stage 2 with percentile
out-of-range bounds instead of strict min/max, "width" frozen check
otherwise unchanged from v2), isolating bound_percentile as the one
difference from v2. Requires --bound-percentile.

--harden-v5 trains ONLY v5 (v1's Stage 1 + Stage 2, otherwise identical to
v2, with a wider --stage2-watch), closing the red-vs-blue hardening loop:
a real Red Team run against v2 on all real development captures
(integration.run_demo --mode test) logged 1293 confirmed misses
(part3.evasion_log); part3.hardening_set.build_hardening_set() turned
those into a real HardeningSet, and the 9 distinct target CAN IDs those
misses named -- almost all outside v2's {0D0, 6E0} watch-list -- are
DEVELOPMENT_HARDENED_WATCH_IDS below (00E excluded: it alone raised
ambient false alarms on ambient_dyno_drive_extended_short in a
single-split check, with no matching recall benefit). Requires
--stage2-watch to differ from DEVELOPMENT_WATCH_IDS.

The false-alarm rate has NO default: it is a team decision.
"""

import argparse

from defender.train import train_defender, train_v2
from part1.pipeline import RoadData
from part1.split_manifest import MANIFEST_PATH, load_manifest

# Stage 2 watch-list: CAN IDs targeted by the DEVELOPMENT attacks
# (max_speedometer, reverse_light_off/on -> 0D0; correlated_signal -> 6E0).
# Chosen from development only, never final_test. 4E7 (coolant, final_test
# only) is deliberately NOT included: an honest test of an unwatched ID.
DEVELOPMENT_WATCH_IDS = ["0D0", "6E0"]

# Candidate watch-list for --harden-v5: DEVELOPMENT_WATCH_IDS plus every CAN
# ID a real, confirmed development miss (part3 Red Team freeze/offset attacks
# against v2, /tmp/evasions_v2.jsonl -> part3.hardening_set.build_hardening_set)
# actually targeted, EXCEPT 00E (raised ambient false alarms in a
# single-split check -- see the module docstring). Still development-only
# and still excludes 4E7 (coolant, final_test only).
DEVELOPMENT_HARDENED_WATCH_IDS = ["006", "033", "0A7", "0D0", "125", "354", "5E1", "6E0"]


def load_windows(road, names, max_windows, log=print):
    windows = []
    for name in names:
        step = road.keep_every(name, max_windows)
        got = list(road.windows(name, keep_every=step))
        windows.extend(got)
        log(f"  {road.capture_id(name)}: {len(got)} windows (every {step}th)")
    return windows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train Defender v1 and v2 on real ROAD data.")
    parser.add_argument("--max-false-alarm-rate", type=float, required=True,
                        help="team decision, e.g. 0.01 (no default)")
    parser.add_argument("--data-dir", default="data/road", help="ROAD folder (default data/road)")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH), help="frozen split manifest")
    parser.add_argument("--model-dir", default="models", help="where to save models")
    parser.add_argument("--window-s", type=float, default=1.0,
                        help="window length; windows never overlap (build plan)")
    parser.add_argument("--max-windows-per-capture", type=int, default=100,
                        help="memory limit, about 1 MB per window (default 100)")
    parser.add_argument("--stage2-watch", default=",".join(DEVELOPMENT_WATCH_IDS),
                        help='Stage 2 watch-list, e.g. "0D0,6E0" (quote it in PowerShell), '
                             'or "all" to watch every ID')
    parser.add_argument("--harden-v3", action="store_true",
                        help="train only v3 (rate frozen check); v1 must already exist")
    parser.add_argument("--harden-v4", action="store_true",
                        help="train only v4 (percentile out-of-range bounds); v1 must "
                             "already exist; requires --bound-percentile")
    parser.add_argument("--harden-v5", action="store_true",
                        help="train only v5 (wider Stage 2 watch-list, from confirmed "
                             "development misses); v1 must already exist; requires "
                             "--stage2-watch to differ from DEVELOPMENT_WATCH_IDS "
                             "(see DEVELOPMENT_HARDENED_WATCH_IDS above)")
    parser.add_argument("--bound-percentile", type=float, default=None,
                        help="Stage 2 out-of-range bounds as a percentile, e.g. 1.0 to "
                             "trim the extreme 1%% off each side (default: strict min/max, "
                             "used only with --harden-v4)")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace existing v1/v2 files (only if never evaluated)")
    args = parser.parse_args(argv)

    if args.max_windows_per_capture < 1:
        parser.error("--max-windows-per-capture must be at least 1")
    if args.harden_v4 and args.bound_percentile is None:
        parser.error("--harden-v4 requires --bound-percentile")
    if sum([args.harden_v3, args.harden_v4, args.harden_v5]) > 1:
        parser.error("pass only one of --harden-v3 / --harden-v4 / --harden-v5")
    if args.harden_v5 and args.stage2_watch.strip() == ",".join(DEVELOPMENT_WATCH_IDS):
        parser.error("--harden-v5 requires --stage2-watch to differ from "
                     "DEVELOPMENT_WATCH_IDS, e.g. "
                     '"' + ",".join(["006", "033", "0A7", "0D0", "125", "354", "5E1", "6E0"]) + '"')

    watch = None if args.stage2_watch.strip().lower() == "all" else \
        [i for i in args.stage2_watch.split(",") if i.strip()]

    road = RoadData(args.data_dir, manifest=load_manifest(args.manifest), window_s=args.window_s)
    train_names = road.manifest.names("train")
    validation_names = road.manifest.names("validation")
    missing = [n for n in train_names + validation_names if not road.path(n).exists()]
    if missing:
        parser.error(f"missing captures in {road.data_dir}: {missing}")

    print("Reading TRAIN captures:")
    train = load_windows(road, train_names, args.max_windows_per_capture)
    print("Reading VALIDATION captures:")
    validation = load_windows(road, validation_names, args.max_windows_per_capture)

    full_train_frames = [road.frames(name) for name in train_names]
    stride_s = args.window_s          # non-overlapping windows
    if args.harden_v3:
        print("Training v3 (v1 Stage 1 + Stage 2 with 'rate' frozen check;"
              " this can take several minutes)...")
        v3 = train_v2(args.model_dir, "v1", train, validation, "v3", args.model_dir,
                      args.window_s, stride_s, overwrite=args.overwrite,
                      stage2_range_captures=full_train_frames, stage2_watch_ids=watch,
                      stage2_frozen_mode="rate")
        reports = (v3,)
    elif args.harden_v4:
        print(f"Training v4 (v1 Stage 1 + Stage 2 with {args.bound_percentile}%"
              " percentile out-of-range bounds; this can take several minutes)...")
        v4 = train_v2(args.model_dir, "v1", train, validation, "v4", args.model_dir,
                      args.window_s, stride_s, overwrite=args.overwrite,
                      stage2_range_captures=full_train_frames, stage2_watch_ids=watch,
                      stage2_bound_percentile=args.bound_percentile)
        reports = (v4,)
    elif args.harden_v5:
        print(f"Training v5 (v1 Stage 1 + Stage 2, watch-list widened to {watch}; "
              "this can take several minutes)...")
        v5 = train_v2(args.model_dir, "v1", train, validation, "v5", args.model_dir,
                      args.window_s, stride_s, overwrite=args.overwrite,
                      stage2_range_captures=full_train_frames, stage2_watch_ids=watch)
        reports = (v5,)
    else:
        print("Training v1 (Stage 1)...")
        v1 = train_defender(train, validation, args.max_false_alarm_rate, "v1", args.model_dir,
                            args.window_s, stride_s, overwrite=args.overwrite)
        print("Training v2 (v1 Stage 1 + new Stage 2, ranges from all train frames;"
              " this can take several minutes)...")
        v2 = train_v2(args.model_dir, "v1", train, validation, "v2", args.model_dir,
                      args.window_s, stride_s, overwrite=args.overwrite,
                      stage2_range_captures=full_train_frames, stage2_watch_ids=watch)
        reports = (v1, v2)

    print("\n=== Training summary (frozen split manifest) ===")
    for report in reports:
        print(f"{report.model_version}: {report.train_windows} train windows, "
              f"{report.validation_windows} validation windows, "
              f"threshold {report.threshold:.6f}, "
              f"validation false-alarm rate {report.validation_false_alarm_rate:.4f} "
              f"(target {report.max_false_alarm_rate})")
    print(f"Stage 2 watch-list: {reports[-1].stage2_watch_ids or 'all IDs'}"
          f"  |  frozen check: {reports[-1].stage2_frozen_mode}"
          f"  |  bound percentile: {reports[-1].stage2_bound_percentile}")
    print(f"Models saved in {args.model_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
