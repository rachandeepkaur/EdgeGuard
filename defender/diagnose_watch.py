"""
Diagnostic (read-only): would a Stage 2 WATCH-LIST stop the false alarms?

Uses the saved v2 Stage 2 field ranges (learned from ALL train frames) and
counts how many NORMAL validation windows have a watched-ID field outside
its training range, or a jump larger than seen in training.

Usage:  python -m defender.diagnose_watch --watch 0D0,6E0
"""
import argparse
from collections import Counter

from part1.pipeline import RoadData
from defender.stage2 import Stage2Model, _field_values


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", required=True, help="comma-separated hex IDs, e.g. 0D0,6E0")
    args = parser.parse_args(argv)
    watch = {i.strip().upper().replace("0X", "").zfill(3) for i in args.watch.split(",")}

    model = Stage2Model.load("models/stage2_v2.json", "v2")
    missing = [i for i in watch if not any(k.startswith(i + "|") for k in model.field_range)]
    print(f"Watch-list: {sorted(watch)}   (never seen in training: {missing or 'none'})")

    road = RoadData("data/road")
    total, flagged, reasons = 0, 0, Counter()
    for name in road.manifest.names("validation"):
        for window in road.windows(name, keep_every=road.keep_every(name, 100)):
            total += 1
            hit = None
            for (can_id, field), values in _field_values(window).items():
                if can_id not in watch:
                    continue
                key = f"{can_id}|{field}"
                if key not in model.field_range:
                    continue
                low, high = model.field_range[key]
                if min(values) < low or max(values) > high:
                    hit = f"{key} out of range [{low},{high}]: saw {min(values)}..{max(values)}"
                    break
                if len(values) >= 2:
                    jump = max(abs(b - a) for a, b in zip(values, values[1:]))
                    if jump > model.field_max_jump.get(key, 0):
                        hit = f"{key} jump {jump} > normal {model.field_max_jump.get(key, 0)}"
                        break
            if hit:
                flagged += 1
                reasons[hit.split(" ")[0]] += 1
    print(f"Normal validation windows: {total}")
    print(f"Flagged on watched IDs:    {flagged}")
    for field, count in reasons.most_common(5):
        print(f"  {field}: {count} windows")


if __name__ == "__main__":
    main()