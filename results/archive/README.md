# Archived results

These cross-validation reports were generated **before** the windowing fix
(commit `cc42046`, "updated part 1") that unified Part 1 and Part 2 on
non-overlapping 1.0s windows and the frozen split manifest.

Each file still shows `"stride_s": 0.5` in its `settings` block, which is
the old, superseded windowing convention — `defender/road_reader.py` (the
module that produced 0.5s-overlapping windows) has since been deleted.

Kept for the record, not for citing in the demo or the final report. The
current, correct reports are `results/crossval_ambient.json` and
`results/crossval_attacks.json`.

| Archived file | Superseded by |
|---|---|
| `crossval_ambient_provisional.json` | `../crossval_ambient.json` |
| `crossval_ambient_splitsjson.json` | `../crossval_ambient.json` |
| `crossval_attacks_splitsjson.json` | `../crossval_attacks.json` |
