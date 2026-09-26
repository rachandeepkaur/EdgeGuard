# CLAUDE.md

Guidance for Claude Code and teammates working in this repo: what exists, what is still open, and the rules that keep the results honest.

## What this is

EdgeGuard is an on-vehicle CAN bus intrusion detector, built on the ROAD dataset (real CAN recordings from one car, with injected attacks). The team splits the work into parts:

| Part | Owner area | Folder |
|---|---|---|
| Part 1 | Reading captures, cleaning, windowing, labels, decoded signals, the split | `part1/` |
| Part 2 | The Defender: detection models, training, threshold, cross-validation | `defender/` |
| Part 3 | Red team, injector, evaluator, evidence gate | `part3/` |
| Member 4 | Dashboard and demo UI | `dashboard/` |
| Shared | Data contracts between parts | `shared/schemas.py` |

**Deadline:** originally Fri 25 Sep 2026, 8pm; **extended** (new date to be confirmed).

**Current branch:** `part1-data-pipeline`. It holds all the Part 1 work below and is **not committed or pushed yet**.

## Source of truth: the revised build plan

`EdgeGuard_Combined_Build_Plan_revised.pdf` (23 Sep 2026) is the team's plan. Where this file or the code disagrees with it, the plan wins. The parts that constrain the code:

- **Windows:** fixed, **non-overlapping 1-second** windows as the first implementation. Preserve message order and timestamps. Each window has `vehicle_id`, `capture_id`, `window_start`, and frames (`timestamp`, `can_id`, raw payload).
- **Labels are private:** labels, attack details and source capture names live in a separate evaluator record, never in the Defender request.
- **Split:** freeze train, validation and final test *before* tuning, and keep related source and derived captures together.
  - Train v1 on train.
  - Develop attacks and v2 candidates on **development**.
  - Compare v1 and v2 **once** on untouched test captures.
- **Threshold:** chosen on validation captures, never on the final test. Section 3 says "validation" while section 6 says "tune thresholds using development data". We read it as: threshold from normal validation drives, development for attacks and v2 misses. **The team still needs to confirm this.**
- **Defender:** Layer 0 rules (frequency, out-of-range decoded values, flatlines, abrupt changes) plus one lightweight anomaly model, combined by a fixed, documented rule. Measure what each contributes. **Every evaluation window is scored.** Decoded signals are optional features, and a missing signal is handled explicitly, never guessed.
- **Deferred:** timing CNN and cross-signal models (unless supported by data and an ablation), large LLM analyst, five attack families, OTA/registry, full fleet dashboard.
- **Metrics:** recall by attack family and overall; **false alarms per hour** of normal traffic; **detection delay** (time to first alert); edge latency (median and p95); end-to-end delay, with window collection separate from processing; v1 vs v2 on the same untouched test groups, including regressions.
- **Part 1 = Block 1 "Data":** load ROAD with the source preserved, create the split manifest, build 1-second windows. Completion check: **normal windows and private labels are reproducible.**

## Where the data is

The ROAD dataset is **not** in this repo and must never be committed. `.gitignore` blocks `data/`, `*.log` and `*.csv`.

- On this machine: `~/Downloads/road/dataset/` (`ambient/`, `attacks/`, both `capture_metadata.json` files, `signal_extractions/DBC/anonymized.dbc`).
- The code defaults to `data/road/`, which does not exist. Either pass `--data-dir ~/Downloads/road/dataset`, or link it once: `mkdir -p data && ln -s ~/Downloads/road/dataset data/road`.

`~/Downloads/road/` is also a **separate preprocessing project** (`edgeguard/` package, `splits.json`, `docs/00–06`). Part 1 here reuses its measured findings, labelling rules and decoder layout rules. It is **not a git repo** and exists only on this laptop.

## Commands

Run from the repo root. The environment is a local `.venv` (gitignored): `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`. Below, `<road>` = `~/Downloads/road/dataset`.

| Task | Command |
|---|---|
| All tests (335, about 20 s) | `.venv/bin/python -m pytest -q` |
| Show the split | `python -m part1.split_manifest show` |
| Rebuild the split (only if the team changes it) | `python -m part1.split_manifest build --splits ~/Downloads/road/splits.json --data-dir <road>` |
| Reproducibility check | `python -m part1.reproduce --data-dir <road> --check results/part1_fingerprint.json` |
| Train v1 + v2 | `python -m defender.run_training --max-false-alarm-rate 0.01 --data-dir <road>` |
| Development check | `python -m defender.dev_check --data-dir <road>` |
| Leave-one-out false alarms (normal drives) | `python -m defender.crossval ambient --max-false-alarm-rate 0.01 --data-dir <road>` |
| 2-fold attack CV (development) | `python -m defender.crossval attacks --max-false-alarm-rate 0.01 --data-dir <road>` |
| Latency benchmark (mock traffic) | `python -m defender.nano_runner --mock --output results/nano_benchmark_MOCK.json` |
| Latency benchmark (real traffic) | `python -m defender.nano_runner --model-version v1 --data-dir <road> --group validation --output results/nano_benchmark_REAL_v1.json` (also v2; `--group` is train/validation/development, never final_test/separate) |
| Integration demo (Block 4, one command) | `python -m integration.run_demo --model-version v2 --data-dir <road>` (both paths; `--mode ordinary`/`--mode test` for just one, `--test-families freeze,offset`, `--evasion-log <path>`) |
| **Final evaluation (Block 5, run ONCE)** | `python -m integration.run_final_evaluation --baseline-version v1 --updated-version v2 --data-dir <road>` -- the only place in the codebase allowed to touch `final_test`. Refuses to overwrite `results/final_evaluation.json` without `--force`. |
| Dashboard | `cd dashboard && npm install && npm run dev` |

The false-alarm rate has no default on purpose: it is a team decision (0.01 has been used so far).

## How data flows (Part 1)

```
ROAD .log  ->  fleet_simulator.iter_raw   (elapsed integer microseconds, malformed lines counted)
           ->  cleaning.clean             (exact duplicates dropped, 0xFFF filler counted, 0.1 % gate)
           ->  windowing.raw_windows      (1 s, non-overlapping, empty windows kept, partial last dropped)
           ->  TrafficWindow              (neutral capture_id + vehicle_id; goes to the Defender)
           +   labels.window_label        (GroundTruthLabel; evaluator only, never to the Defender)
preprocess(window, decoder)               (fills decoded_signals AFTER any injection, on a copy)
```

Everyone gets data through **`part1.pipeline.RoadData`**: `windows()`, `labelled_windows()`, `frames()`, `metadata()`, `keep_every()`. It reads the split from `part1/split_manifest.json` and **refuses final_test / separate captures** unless created with `final_evaluation=True`.

## What is done

### Part 1: Block 1 steps 1–6, all done (branch `part1-data-pipeline`)

| Step | What was built | File(s) | Checked by |
|---|---|---|---|
| 1. Frozen split manifest | 45 captures, 5 groups, one neutral-ID scheme (same as `discover_captures`). Twins share a group; development attacks carry a fold (`_1`/`_2`). Checked on load. | `part1/split_manifest.py`, `part1/split_manifest.json` | `tests/test_part1_split_manifest.py` (including the committed file) |
| 2. One windowing function | 1 s non-overlapping, integer-µs boundaries, `vehicle_id`, `keep_every` sampling for training only, cleaning built in. `defender/road_reader.py` **deleted**. | `part1/windowing.py`, `part1/fleet_simulator.py` | `tests/test_part1_windowing.py` |
| 3. Metadata and private labels | Frame rules: masquerade, fabrication (with payload mask), fuzzing (whole capture), accelerator (none). Closed interval compared in integer µs. A window is an attack if it holds ≥ 1 injected frame; labels store `injected_frames` and `interval_overlap_s`. | `part1/labels.py`, `part1/pipeline.py` (`CaptureMetadata`) | `tests/test_part1_labels.py`, `tests/test_part1_pipeline.py` |
| 4. Cleaning | Exact duplicates dropped, `0xFFF` filler counted not removed, malformed lines counted, out-of-order frames raise. A capture fails if more than 0.1 % would be removed. | `part1/cleaning.py` | `tests/test_part1_cleaning.py` |
| 5. Decoded signals | Pure-Python DBC decoder (Intel and Motorola, signed, NaN for short payloads, error for watched IDs missing from the DBC). `preprocess()` fills `decoded_signals` as `"0D0:Unknown_4" -> values`. | `part1/decode.py`, `part1/pipeline.py` | `tests/test_part1_decode.py` |
| 6. Reproducibility check | Hashes every window and label per capture. Neutral IDs only, safe to commit. | `part1/reproduce.py`, `results/part1_fingerprint.json` | `tests/test_part1_pipeline.py` |

**Checked on the real dataset:**
- **Decoder:** bit-exact against ROAD's signal CSVs. 0 mismatches over 115,287 frames of `ambient_dyno_reverse` and 197,664 frames of `max_speedometer_attack_1_masquerade`, across all 664 signals.
- **Labels:** each of the 8 development fabrication captures has exactly as many injected frames as its masquerade twin (e.g. 2,445 for `max_speedometer_attack_1`, matching the other project's figure).
- **Reproducibility:** 27 captures (train, validation, development) run twice with identical windows and labels.

**Shared schema changes** (`shared/schemas.py`, all optional fields, so older code still works):
- `TrafficWindow.vehicle_id`, checked to be neutral.
- `GroundTruthLabel.injected_frames` and `interval_overlap_s`. A normal label may not have them.

### Part 2 (`defender/`): switched over to Part 1
- `run_training`, `dev_check`, `crossval`, `diagnose`, `diagnose_watch` now read the manifest and get windows and labels from `RoadData`. `PROVISIONAL_SPLIT`, `neutral_ids()` and the time-overlap labelling are gone.
- `Defender.load()` reads the window length from `train_info_<version>.json`, and `score_window()` **refuses windows of any other length**.
- `stage2.py`: full-capture range stats are computed once per capture and reused across folds (`capture_range_stats`, `merge_range_stats`), giving an identical model.
- **v1 and v2 were retrained** on the manifest (`models/*_v1.json`, `*_v2.json` overwritten; they had never been final-evaluated).
- `nano_runner.py`: latency benchmark, mock traffic **or real ROAD windows** (`--model-version`, see the Commands table).

### Dashboard (`dashboard/`): wired to real v2 Defender output, 2026-09-24
React + Vite, KPI tab, live architecture diagram, fleet view. `DATA_SOURCE = "live"`
(`dashboard/src/App.jsx`) by default: `dashboard/scripts/build_real_run.py` scores four real
ROAD development captures end to end with the real trained v2 Defender and writes
`dashboard/src/data/realRun.json` (216 windows, 86 flagged ATTACK, real evidence text and
latency) -- see `dashboard/README.md`'s "Data sources" section. **Updated 2026-09-25:** the
fourth capture, `reverse_light_off_attack_1`, is a deliberate hard case v2 misses (0/8), and
each window now carries an evaluator-side `truth` field (added after scoring, never given to
the Defender) so the dashboard can mark misses and false alarms in amber (8 missed, 1 false
alarm). Highway drives were not used: they are final_test. This is a
REPLAY of a precomputed run, not a live socket to a running Nano; there is still no live
streaming inference process anywhere in this repo. The mock generator is kept as a fallback
(`DATA_SOURCE = "mock"`).

**Found and fixed in the same pass:** the root `.gitignore`'s bare `data/` pattern (meant only
to keep the real ROAD dataset out of git) also matched `dashboard/src/data/`, so
`mockRun.js`/`schema.js` had **never actually been committed** -- they existed on disk but
were absent from every commit in `git log`. Fixed by anchoring the pattern to `/data/` (repo
root only); both files are now tracked for the first time, alongside the new `liveFeed.js`/
`realRun.json`/`build_real_run.py`.

### Part 3 (`part3/`): red team attacker + injector, both attack families
Ported forward from the unmerged `part3-red-team` branch (`fb98314`), which had drifted too
far from current `main` to merge (it predated `part1.pipeline`/`RoadData` entirely). The
design (evaluator, metrics, evasion log, hardening set, evidence gate) is unchanged from that
branch; `attack_spec.py`/`attack_injector.py`/`attack_validator.py`/`red_team_agent.py` were
updated for current interfaces and gained the **second attack family** the build plan calls
for:
- **freeze** (original) -- stops a changing CAN payload.
- **offset** (new) -- pushes one payload byte toward an extreme by a bounded signed delta,
  clamped to 0-255. Targets exactly what `stage2.py` already checks per byte (`out_of_range`).

`red_team_agent.propose(window, family)` and `attack_injector.inject(window, spec)` dispatch
by family. Confirmed on real ROAD data: a targeted offset attack on the watched ID `0D0` flips
v2 from ACCEPT (score 0.96) to ATTACK (score 0.999, `out_of_range` evidence).

### Integration (`integration/run_demo.py`): Block 4, one command, both paths
- **Ordinary path**: replay (`RoadData` via `nano_runner.real_windows`) -> preprocess ->
  `Defender.score_window` -> `SimulatedConsumer` decision (ALERT/ISOLATION or FORWARD).
- **Test path**: replay of DEVELOPMENT captures only (Red Team's own rule) -> `propose` an
  attack -> `inject` it into a copy -> score both the attacked window and its unmodified
  pair (a negative control) -> `part3.evaluator` + `part3.metrics` report real recall
  (overall and by family) and false-alarm rate, since the label is known for certain here.
  Optional `--evasion-log` appends confirmed misses via `part3.evasion_log`.
- Confirmed on real ROAD data (development captures, 837 windows x 2 families): v1 (Stage 1
  only) recall 2.6%; v2 (Stage 1 + Stage 2) recall 22.8% overall (15.8% freeze, 29.7% offset),
  false-alarm rate 29.7% on paired normal controls. That false-alarm rate is high and not yet
  investigated -- these are generic Red Team proposals, not targeted at the `0D0`/`6E0`
  watch-list, so this is a rougher, broader test than the hand-targeted example above, closer
  to a stress test than a calibrated benchmark.

### Cloud-escalation path (build plan section 7): wired end-to-end, 2026-09-25
This was previously listed as "still a stub" -- that undersold what already existed:
`part1/escalation_policy.py`, `part1/sanitizer.py`, `part1/egress_metrics.py` and
`part1/mock_cloud_endpoint.py` were all real and individually tested, but **nothing in
`integration/run_demo.py` ever called them together** -- the ordinary path never actually
escalated anything. This closes that gap: the HP Edge AI SJSUHack brief (see
`docs/Edge_AI_SJSUHack_for_Students_-_ZGX_Nano_GB10.pdf`) requires "an explicit, defensible,
measurable decision about when to escalate to the cloud" as a judged deliverable, not an
optional extra -- this section is that decision, made real.

- **`integration/run_demo.py --escalate --max-escalation-rate <rate>`** (new): for every
  ordinary-path window, after the local `Defender.score_window()` decision is made (and
  logged to `SimulatedConsumer` exactly as before -- **escalation never touches the local
  decision or action**), `should_escalate()` checks whether the score falls in an
  uncertainty band around the threshold. If so, the window is sanitized
  (`part1.sanitizer.sanitize`, an allow-list -- no raw CAN bytes ever leave) and POSTed to a
  running simulated-cloud process (`part1.mock_cloud_endpoint`, launch separately with
  `python -m uvicorn part1.mock_cloud_endpoint:app --host 127.0.0.1 --port 9000` --
  the plain `uvicorn` command may not be on PATH depending on how it was installed; `python
  -m uvicorn ...` always works). A failed or unreachable call is recorded as a failure
  (`part1.egress_metrics`), never as an exception -- verified by actually killing the cloud
  process and confirming the ordinary-path decisions and actions were byte-identical to a
  non-escalating run.
- **`calibrate_escalation_band()`** (new, in `run_demo.py`): the band half-width is
  calibrated on VALIDATION scores only (`choose_band_half_width`, same "team decision,
  no default" contract as `--max-false-alarm-rate`) -- never on the windows the ordinary
  path is about to score, and never on development or final_test.
- **Found and fixed a real bug while wiring this up**: `choose_band_half_width()` could
  return a width outside the `[0, 0.5]` range `should_escalate()` itself enforces, whenever
  the threshold was far from 0.5 -- which real Defender thresholds always are (v2's is
  0.9988). Every existing test for it happened to use `threshold=0.5` exactly, where
  `|score - 0.5|` can never exceed 0.5, so the bug never showed up until this was run against
  a real model. Fixed by clipping each candidate width to 0.5; added a regression test using
  an off-center threshold.
- **Real run against real ROAD data, live cloud process** (v2, validation captures,
  815 windows, `--max-escalation-rate 0.3`): band = threshold +/- 0.095; **240 / 815 (29.5%)
  escalated**, 0 failed calls, mean added latency ~101 ms. **235 / 240 (97.9%) came back as a
  "correction"** -- but this number is inflated by a real mismatch, not a sign the cloud
  second opinion is unusually valuable: `mock_cloud_endpoint`'s stand-in threshold (0.6,
  fixed) sits nowhere near v2's real threshold (0.9988), so almost every escalated score
  disagrees with the stand-in by construction. The mechanism (calibration, sanitization,
  graceful degradation, egress metrics) is real and correctly wired; the **correction-rate
  number is not yet a meaningful measurement** and should not be quoted as one without
  recalibrating the stand-in threshold relative to each model's own operating point first.
- Calibrated tests: 9 new tests in `tests/test_integration_run_demo.py` (calibration bounds,
  cloud-unreachable resilience, a real FastAPI `TestClient`-backed second opinion, CLI
  validation) plus 1 regression test in `tests/test_part1_escalation_policy.py`; 521 tests
  passing repo-wide.

### Dashboard: fixed a stale "cloud not wired up" panel, 2026-09-25

`RoutingPanel` still said "No cloud escalation path exists in the pipeline yet" and "CLOUD --
not wired up" -- true when it was written, false since the escalation path was wired end to end
(see the Cloud-escalation section above). Anyone opening the dashboard, including a judge, would
have been told the opposite of what the repo actually does on its single most rubric-relevant
feature. Fixed:

- `dashboard/scripts/build_real_run.py` now calls `integration.run_demo.calibrate_escalation_band`
  (real budget 0.3, same as everywhere else) and writes `escalationBandHalfWidth` into
  `realRun.json`'s meta, alongside the existing real v2 inference output. Regenerated
  `realRun.json` for real -- 95 of 188 replayed windows fall in the real band, so the badge
  below actually changes state during playback, not a static claim.
- `dashboard/src/utils/escalation.js`: a small client-side mirror of
  `part1/escalation_policy.should_escalate()`, display-only -- the real decision is always made
  in Python.
- `RoutingPanel` now shows, per replayed window, whether it falls inside the real calibrated band
  ("IN ESCALATION BAND -- would call cloud" / "CONFIDENT -- no escalation"), plus the real
  measured escalation-quality numbers (precision 48.1%, all false alarms escalated, 183/1293
  misses escalated, ~241 bytes/0 raw CAN bytes per escalation) from
  `results/escalation_quality_v2/demo_test.json`, copied into `dashboard/src/data
  /escalationQuality.js` with a comment pointing at the source file.
- `mockRun.js`'s `CAPTURE_META` also carries a band (the same real calibrated value) so mock mode
  exercises the identical display logic without crashing.

Verified with `npm run build` (clean) and by computing escalation counts against the regenerated
`realRun.json` directly in Node.

### Escalation quality measured against known labels, 2026-09-25 (`part1/escalation_quality.py`)

Replaces the simulated cloud's "correction rate" as the escalation metric. That rate only measured
how often the stand-in cloud (fixed threshold 0.6) disagreed with v2 (threshold 0.9988), so it said
nothing about whether escalation targets real mistakes. `compute_escalation_quality()` scores the
band directly against the Red Team test path's known labels, reusing `part3.evaluator.evaluate_batch`
for the join so it can never disagree with the detection metrics. No cloud call is involved.

`integration/run_demo.py --escalate` now also applies to the test path (previously rejected with
`--mode test`); in `--mode both` the test path reuses the ordinary path's band. The ordinary path's
printout now labels the stand-in's disagreement rate as "NOT a quality metric".

Real result, v2, band +/- 0.0950 (30% budget on validation normal scores), 2,511 development test
windows (`results/escalation_quality_v2/demo_test.json`):

| | |
|---|---|
| Escalated | 898 / 2511 (35.8%) |
| Local mistakes | 1542 (249 false alarms, 1293 misses) |
| Escalation precision | 48.1% |
| Escalation recall | 28.0% -- all 249 false alarms, but only 183 / 1293 misses |
| Error rate outside the band | 68.8% (higher than inside) |
| Accuracy local -> ceiling with a perfect second opinion | 38.6% -> 55.8% |
| Bundle size | 241 bytes mean, 269 max; 0 raw CAN bytes (allow-list pinned by a test) |

Reading: the band's top is clipped at 1.0 (v2's threshold is 0.9988), so every ATTACK decision is
escalated -- escalation reviews every false alarm. Misses are mostly confident (scored below the
band) and a score-based second opinion never sees them; the 1293 misses are the same 1293 analysed
in the hardening section (70% coverage gap). Caveat: the test mix is two attacked windows per
normal one, so accuracy figures reflect that mix, not real attack rates.

### Stage 2 hardening attempts (v3, bound_percentile): two tried, neither fixes the known gaps
- **v3** (`--harden-v3`, "rate" frozen check) was already coded before this session; trained and
  dev-checked on real data for the first time here. Result: **byte-identical to v2** on every
  development capture (247/390 detected, 1/237 false alarms) -- reverse_light_off/on are still
  0/8 and 0/21 detected. The Part 3 freeze repro's frozen score rises slightly (0.983 vs v2's
  0.959) but not past v3's own threshold (0.999), so the decision doesn't change either.
  **"Rate" mode does not fix the reverse-light gap.** That needs actual decoded-signal (bit-level)
  checking (`part1/decode.py` exists and is tested, but nothing calls `preprocess(window, Decoder)`
  before scoring yet) -- still open, not built this session.
- **bound_percentile** (`Stage2Model(bound_percentile=...)`, `--harden-v4` / `--bound-percentile`
  on `run_training`/`crossval`): a NEW option, this session -- out-of-range bounds become the
  p-th/(100-p)-th percentile of training values instead of strict min/max, leaving frozen_break
  and large_jump untouched. Backward compatible (old v1/v2/v3 model files load unchanged; 19 new
  tests, 500 total passing). **A targeted check against `ambient_dyno_drive_extended_short`** (the
  known worst leave-one-out drive, 29/90 false alarms per the table below) **found it does not
  help -- false alarms stayed at 29-30/90 across percentiles 0.1 to 5.0, slightly WORSE, not
  better.** Diagnosis: every false alarm is `out_of_range` on `0D0 byte7`, whose training range
  was `[0, 129]` from the other 8 drives; `extended_short` pushes it past that. This looks like a
  genuine **coverage gap** (this byte's true range -- quite possibly a rolling counter or
  checksum -- extends further than 8 drives happened to show), not a rare outlier inflating the
  bound. Percentile trimming fixes the opposite failure mode (a bound made too WIDE by one freak
  training value); here the bound was already too NARROW, so trimming it further only made things
  worse. **Do not train a production v4 from this without addressing the coverage question**
  (wider training regime coverage, or excluding likely counter/checksum fields from out_of_range,
  or a margin that EXPANDS rather than trims the learned range) -- the feature is implemented and
  tested, but this specific use of it is a documented dead end, not a fix.

### Hardening loop closed: v5, watch-list widened from real confirmed misses, 2026-09-24
This closes the gap the architecture doc calls the "red-vs-blue hardening loop" ("Fleet window ->
Red Team proposal -> constrained injector -> Blue Team score -> ground truth comparison ->
confirmed misses -> candidate update -> held-out evaluation"): `part3.hardening_set.build_hardening_set()`
existed and was tested, but nothing outside its own test file had ever called it on a real evasion
log -- v2 was never actually hardened against Part 3's own confirmed misses. **v3 and v4 above were
not that**; both are independent Stage 2 variants, not built from a `HardeningSet`.

- **Step 1, real confirmed misses:** reran the Red Team test path (`integration.run_demo --mode test
  --model-version v2`) on all 18 real development captures, reproducing the Integration section's
  numbers exactly (2511 windows, recall 22.8% overall) and writing every miss to
  `/tmp/evasions_v2.jsonl` (1293 confirmed false negatives; not committed -- it's a byte-for-byte
  reproducible artifact of a one-line command, not a durable one).
- **Step 2, `part3/hardening_report.py` (new)**: wires `build_hardening_set()` to a real evasion
  log for the first time, and breaks the misses down by family and by whether Stage 2's watch-list
  was covering the target CAN ID (`results/hardening_report_v2.json`):
  - **910 / 1293 (70%) were a coverage gap**: the target CAN ID (`006`, `00E`, `033`, `0A7`, `125`,
    `354`, `5E1`) was never on Stage 2's `{0D0, 6E0}` watch-list, so Stage 2 could not have flagged
    them regardless of severity. **All 588 offset misses target `006` alone** -- `part3.red_team_agent`'s
    `propose_offset()` always picks the most STABLE byte in the window, and `006` is apparently that
    byte in nearly every development capture, which also makes it an easy `out_of_range` catch once
    watched (a byte that never moves, pushed to an extreme).
  - **383 / 1293 (30%) were a sensitivity gap**: all `freeze`, all already on the watched `0D0`/`6E0`
    IDs, still missed -- `frozen_break` itself isn't catching them, independent of watch coverage.
- **Step 3, candidate (`--harden-v5`)**: widened the watch-list to every CAN ID a real miss named,
  **except `00E`** -- a single-split ambient check (`ambient_dyno_drive_extended_short`, the known
  worst leave-one-out drive) found `00E` alone raises false alarms 29/90 -> 36/90 with no matching
  recall benefit (checked one ID at a time; the other 6 additions were each neutral, and combined
  they reproduce the 29/90 baseline exactly -- not the full nested LOCO, same caveat as the
  `bound_percentile` check above). `DEVELOPMENT_HARDENED_WATCH_IDS` in `defender/run_training.py`
  records this list; `models/*_v5.json` are the real trained files.
- **Result, real Red Team test on development data (never final_test)**:

  | | v2 (baseline) | v5 (candidate) |
  |---|---|---|
  | Recall overall | 22.8% | **58.1%** |
  | Recall, freeze | 15.8% | 16.1% (unchanged) |
  | Recall, offset | 29.7% | **100%** |
  | False alarms (paired controls) | 29.7% | 29.9% (unchanged) |

  **Offset is effectively solved** by watch-list coverage alone, at no false-alarm cost. **Freeze is
  not** -- `results/hardening_report_v5.json` shows the same 383-ish freeze misses persist even
  though every one of their targets is now watched (only 22 residual misses, all on the deliberately
  excluded `00E`), confirming this is a `frozen_break` sensitivity problem, not a coverage problem,
  and matching the "rate" mode finding above: neither frozen-check variant nor watch-list coverage
  fixes freeze detection. That needs its own investigation (how a freeze this short registers within
  a 1 s window), separate from this fix.
- **v5 FAILS the evidence gate -- do not ship it.** Run 2026-09-24 with explicit sign-off
  (`results/final_evaluation_v2_vs_v5.json`, Block 5 below): on the real `final_test` attacks
  (`correlated_signal`, `max_speedometer`, `reverse_light_on/off`, `max_engine_coolant_temp`,
  `fuzzing` -- none of which target the CAN IDs `--harden-v5` added), **recall is byte-identical to
  v2** (tp 168, fn 91, every family's recall unchanged to the digit) -- the widened watch-list never
  once caught a real attack v2 didn't already catch -- **while false alarms roughly triple** (446 ->
  1390 windows, 9.07% -> 28.3% of final_test, worse than the already-flagged v2 problem) and mean
  inference time nearly triples (2.2 ms -> 6.4 ms). The single-split `extended_short` ambient check
  above was not enough to catch this: final_test's 14 captures include 2 highway drives never seen
  in training (Known problems and limits #4) and the real coolant/reverse-light attack captures,
  and evidently at least one of the newly-watched IDs (`033`, `0A7`, `125`, `354`, or `5E1`) is far
  noisier across that population than it was on the one drive checked.
  **Lesson: a watch-list built from Part 3's own synthetic Red Team attack choices does not
  transfer to the real ROAD final_test attacks** -- `propose_offset`'s "attack the most stable byte"
  heuristic and the real attackers' actual targets are different distributions, so a candidate that
  looks like a clean win against Red Team (see the recall table above) can still be a real
  regression on genuinely held-out driving data. Before trying watch-list widening again: validate
  with the FULL nested ambient LOCO (`defender/crossval.py ambient`, all 9 train+validation
  captures, not one held-out drive) as a minimum bar, not a single-split spot check.
- v1-vs-v2 and v2-vs-v5 are both spent now; a v6 attempt needs a real fix (freeze sensitivity, or a
  more conservative field selection for offset) and its own justified use of `--force`.
- Calibrated tests: `tests/test_hardening_report.py` (9 tests) and 3 new tests in
  `tests/test_defender_run_training.py` for `--harden-v5`; 512 tests passing repo-wide.

### Detection delay added, real re-run, 2026-09-25 (`part3/detection_delay.py`)

Closes half of item 7 below (false alarms/hour was already done; detection delay was the other
open piece). Measurement only -- v1 and v2 are the same frozen models, no retraining, no new
decision made from this. Reuses the exact same one-time final_test scoring pass
`integration/run_final_evaluation.py` already does; just also keeps window_start/window_end and
feeds them to `compute_detection_delay()`, which finds, per capture with a real continuous
injection interval (fabrication/masquerade only -- fuzzing has no single onset), the first
ATTACK-decision window at or after the interval's start and reports its window_end minus the
interval start (window collection time is real detection cost; inference is ~1-2ms and reported
separately, per the plan's own metric definition).

**Real bug caught building this, before it ever reached results/final_evaluation.json:** the
first implementation kept every scored TrafficWindow (including its ~2,400 raw CAN frames) alive
for the whole final_test run just to read three numbers off each one later -- roughly 12 million
Frame objects at once across 5,178 windows, which OOM-killed the script (`exit 137`, empty
stdout, no traceback -- SIGKILL doesn't allow one) on every attempt until diagnosed. Fixed by
extracting a lightweight `_WindowTiming(capture_id, window_start, window_end)` per window instead
of retaining the window itself; peak RSS dropped to 52MB. Worth remembering: never accumulate a
list of TrafficWindow objects across a whole capture set for later use -- extract only the fields
actually needed, immediately, or the raw frames make it unnecessarily heavy.

Verified before trusting the re-run: `results/final_evaluation.json`'s tp/fp/tn/fn/recall/
precision/f1/recall_by_family for both v1 and v2 are byte-identical to the pre-existing report
(only `mean_inference_ms` moved, as expected -- it's a timing measurement, not a frozen result).

Real result, 11 final_test captures with a real interval:

|                          | v1    | v2       |
|--------------------------|-------|----------|
| captures ever detected   | 5/11  | 6/11     |
| mean delay (detected)    | 28.2s | 0.57s    |
| p95 delay (detected)     | 39.5s | 0.93s    |

v1's rare detections come from Stage 1 timing drifting far enough to cross threshold, which takes
a long time; v2 catches what it catches almost immediately. Read alongside recall, not instead of
it -- the "detected" subsets differ in size and identity between the two models (5 vs 6 of the
same 11 captures), so this is a fair but small-n comparison. README's "Real results" section has
this table; `results/final_evaluation.json`'s `baseline_detection_delay` / `updated_detection_delay`
have the full numbers.

### Per-capture false-alarm breakdown added, real re-run, 2026-09-25 (`part3/per_capture_metrics.py`)

Closes item 8 from the review list (analyze highway/generalization separately) and directly
answers `part3/metrics.py`'s own docstring TODO ("Overlapping windows are correlated; report
capture-level results separately"). Measurement only -- groups the exact same `evaluate_batch()`
results already computed for the blended final_test report by `capture_id`, via the same
`window_id -> capture_id` map used for detection delay (`_WindowTiming`, never the full
`TrafficWindow`, same memory reason as above). No model change, no new scoring pass.

Real result, v2's 446 final_test false alarms broken out by the 3 normal captures:

| capture | windows | false alarms | rate |
|---|---|---|---|
| ambient_highway_street_driving_long (cap12) | 3,764 | 433 | 11.5% |
| ambient_highway_street_driving_diagnostics (cap11) | 469 | 9 | 1.9% |
| ambient_dyno_drive_basic_short (cap02) | 444 | 4 | 0.9% |

433/446 (97.1%) of every v2 false alarm on final_test comes from one capture; both highway
captures together are 442/446 (99.1%). This replaces a bug in our own earlier write-up: the
README and item 15 below had compared final_test's blended rate against
`results/crossval_attacks.json`'s "15.19/hour" figure, which is `normal_windows_in_attack_captures`
-- just 1 alarm over ~4 minutes of normal driving embedded in attack captures, far too small a
sample to be a baseline. The correct, robust comparison is `results/crossval_ambient.json`'s proper
9-capture leave-one-out estimate: 30 alarms / 763 windows = 141.5/hour, which the per-capture
breakdown is consistent with once highway driving (present in final_test, absent from
train/development entirely) is accounted for. Calibrated tests: `tests/test_part3_per_capture_metrics.py`
(5 tests, all passing); 553 tests passing repo-wide.

### Freeze-sensitivity root cause investigated, 2026-09-25 (read-only diagnostic, no code change)

Closes item 6 from the review list. Freeze-family recall has been stuck around 16% since the
hardening work (`sensitivity_gap_by_family: {"freeze": 383}` in `results/hardening_report_v2.json`
-- all 383 sensitivity-gap misses, as opposed to coverage-gap misses on unwatched IDs, are freeze).
Investigated with two live diagnostic scripts against the real trained v2 model (not saved to a
file -- read-only investigation):

1. Built real freeze attacks with `part3.red_team_agent.propose()` + `part3.attack_injector.inject()`
   against real development windows on the watched IDs (`0D0`, `6E0`), scored with the real v2
   `Defender`, and printed `stage2._raw_values()`. The `frozen_break` check fires at its maximum
   possible raw value (65409.0, field `0D0`'s `pair6_be` full training-range width) on every single
   attempt -- the check itself works. But the final fused decision stayed ACCEPT: `attack_score=0.6134`
   against `threshold=0.9988`.
2. Loaded `Defender.load("models", "v2").stage2.reference["frozen_break"]` directly (581 training
   windows) and counted values: 43520.0 appears 270 times, **65409.0 appears 224 times** (38.6% of
   all normal training windows), 129.0 appears 47 times, 65330.0 appears 30 times. `bisect_left(ref,
   65409.0) = 357`, so `score = 357/582 = 0.6134` -- exactly matching the live attack's score.

**Root cause confirmed:** `_unusualness()` (`defender/stage1.py`) scores `frozen_break` as a
percentile rank against every normal training window's raw value, and `frozen_break`'s raw value in
`frozen_mode="width"` (the default) is a *fixed per-field constant* -- the field's total training
range width -- not a measure of how extreme this specific freeze event is. Because `0D0`'s watched
field legitimately freezes at that exact width value in 224 of 581 (38.6%) real normal training
windows (this vehicle's dyno-based driving genuinely holds this signal still a lot), a synthetic
freeze attack that produces the identical value only ranks at the 61st percentile of "normal" --
nowhere near the ~99.88th-percentile threshold -- even though the raw check fires perfectly every
time. This is a genuine sensitivity limit of percentile-rank scoring for this dataset's driving
conditions, not a code bug, and not something a threshold change alone can fix (the check output
itself doesn't distinguish a real freeze attack from this vehicle's normal held-still readings).
A real fix would need a different raw signal for `frozen_break` (e.g. how long the current value has
already been held, rather than which value it is) -- out of scope this close to the deadline; see
"Known problems and limits" and README's "What EdgeGuard does not yet catch" for the disclosed
write-up.

### Final evaluation (`integration/run_final_evaluation.py`): Block 5, RUN on 2026-09-24 -- `results/final_evaluation.json`
- The one-time, held-out v1-vs-v2 comparison on `final_test` (build plan S9). `run_final_evaluation()`
  is the only place in the codebase that constructs `RoadData(..., final_evaluation=True)`; it uses
  `evidence_gate.compare_final_models()` for the overlap/consistency checks. This has now been spent --
  do not rerun without `--force` and a stated reason; a real code or model change gets a new version to
  compare, not a redo of v1-vs-v2.
- 14 final_test captures, 5178 windows.

| | v1 (baseline) | v2 (updated) |
|---|---|---|
| Recall | 0.39 % (1 / 259) | **64.9 %** (168 / 259) |
| False alarms | 20 / 4919 = 0.41 % | **446 / 4919 = 9.07 %** |
| Precision | 4.8 % | 27.4 % |
| F1 | 0.7 % | 38.5 % |
| Mean inference | 0.31 ms | 1.80 ms |

Recall by family:

| | v1 | v2 |
|---|---|---|
| correlated_signal | 0 % | **100 %** |
| max_speedometer | 0 % | **100 %** |
| fuzzing | 100 % | 100 % |
| reverse_light_on | 0 % | 39.6 % |
| reverse_light_off | 0 % | **0 %** |
| max_engine_coolant_temp | 0 % | **0 %** |

- **The headline v1-vs-v2 story holds on the real final test**: v2's Stage 2 payload check is the
  reason it catches anything beyond fuzzing, exactly as claimed everywhere else in this doc.
- **But v2's false-alarm rate on final_test (9.07 %) is far above the 1 % target** used for training
  and validation, and roughly 3x the 3.9 % leave-one-out figure -- worse than any single ambient
  drive examined during cross-validation. Don't quote v2 as "meets the false-alarm target" without
  this caveat.
- **max_engine_coolant_temp is 0 % for both models.** This is exactly the risk flagged in "Known
  problems and limits" #5: coolant (`4E7`) is the one final-test attack target that never appears in
  development, and the result confirms the watch-list (`0D0`, `6E0`) does not generalize to it. v2's
  gains are real but concentrated on IDs it was effectively already tuned for.
- **reverse_light_off stays at 0 % for v2**, consistent with the development-set miss noted above (the
  reverse-light bit-flip needs decoded-signal checking that Stage 2 doesn't have yet).
- Calibrated tests: `tests/test_run_final_evaluation.py` (7 tests, fake data only, all passing).

### Second evidence-gate run: v2 vs v5, RUN on 2026-09-24 -- `results/final_evaluation_v2_vs_v5.json`
- `v5` is the hardening-loop candidate above (Stage 2 watch-list widened from Part 3's real confirmed
  misses). Run with explicit sign-off, same 14 final_test captures, 5178 windows.

| | v2 (baseline) | v5 (updated) |
|---|---|---|
| Recall | 64.9 % (168 / 259) | 64.9 % (168 / 259) -- **identical** |
| False alarms | 446 / 4919 = 9.07 % | **1390 / 4919 = 28.3 %** |
| Precision | 27.4 % | 10.8 % |
| F1 | 38.5 % | 18.5 % |
| Mean inference | 2.21 ms | 6.39 ms |

- **v5 fails the evidence gate.** Every family's recall is unchanged to the digit (the widened
  watch-list never caught a real final_test attack v2 didn't already catch -- none of the real
  attacks target the CAN IDs it added), while false alarms roughly triple. See the Hardening loop
  section above for the full analysis and the lesson (a watch-list tuned to Part 3's own synthetic
  Red Team attack choices does not transfer to real ROAD attacks). **v2 remains the model to ship.**
  `v5`'s model files stay in `models/` as a documented, real, negative result -- not a candidate for
  the demo.

## Results (frozen manifest, Part 1 windows and labels)

All from train, validation and development captures. **None of this is the final test.**

**Training:**

| | v1 | v2 |
|---|---|---|
| Threshold | 0.958763 | 0.998808 |
| Validation false alarms (target 1 %) | 1 / 182 = 0.55 % | 1 / 182 = 0.55 % |

Training used 581 windows from 7 drives; validation used 182 windows from 2 drives.

**Development check** (`dev_check`, every window of the 18 development attack captures):

| | v1 | v2 |
|---|---|---|
| Attacked windows detected | 10 / 390 | **247 / 390** |
| False alarms on normal windows | 6 / 237 | 1 / 237 |

- **v2 catches:** correlated_signal and max_speedometer (100 %), fuzzing (100 %), and fabricated reverse_light_on_2 (38 / 38).
- **v2 misses:** reverse_light_off, reverse_light_on_1, all reverse-light masquerades, and Part 3's frozen-`0D0` window.

**2-fold attack CV** (`results/crossval_attacks.json`), average recall over attack types:

| | v1 | v2 |
|---|---|---|
| Fabrication | 1 % | 67 % |
| Masquerade | 1 % | 50 % |
| Fuzzing | 100 % | 100 % |
| False alarms on normal windows | 6 / 237 | 1 / 237 |

**Leave-one-out false alarms on normal drives** (`results/crossval_ambient.json`; 9 drives, each held out in turn, out-of-fold threshold, target 1 %):

| | v1 | v2 |
|---|---|---|
| False alarms | 5 / 763 = 0.66 % (95 % upper 1.4 %) | **30 / 763 = 3.9 %** (95 % upper 5.3 %) |
| Per hour of normal driving | 24 / h | 142 / h |
| Drives over the 1 % target | 5 / 9 (0 or 1 alarm each, on ~50–100 windows) | 2 / 9 |

- 29 of v2's 30 false alarms come from **one drive, `extended_short`** (29 / 90 windows).
- Every other drive has 0 or 1 false alarm.
- The per-hour figures use sampled windows (at most 100 per drive), so treat them as rough.
- Validation alone (0.55 %) hides this: it is the same coverage problem seen before the manifest.

Results from before the manifest (provisional split, 0.5 s overlapping windows) are superseded, and are kept only in `defender/README.md` for the record.

## Decisions the team must confirm (before merging)

1. **Shared schema change:** `vehicle_id`, `injected_frames`, `interval_overlap_s`. `schemas.py` says fields change only with team agreement, and Part 3 must know about them.
2. **Part 1's code was rewritten** by Part 2. The Part 1 owner should review `part1/`.
3. **Fuzzing `_1`/`_2` are in development.** `splits.json` puts all fuzzing in test, because that project trains Stage 1 on synthetic fuzzing. This one doesn't.
4. **No highway driving in training.** Both highway drives are final test (as in `splits.json`), so expect more false alarms on them at the final test. Accept this consciously, or move one highway drive to train and rebuild the manifest.
5. **Threshold on validation vs development** (the build plan contradiction above).

## Known problems and limits

1. **Stage 2 false alarms depend on which drives are in training.** In leave-one-out, `extended_short` alone produces most of v2's false alarms (29 / 90) when it's held out. Its value ranges aren't covered by the other drives.
2. **Validation is small:** 2 drives, 182 windows. Use the leave-one-out number beside it.
3. **Decoded signals exist, but the Defender doesn't use them yet.** The reverse-light misses (one bit inside a byte) need Stage 2 to check decoded signals. Until then that recall gain is a hypothesis.
4. **The frozen-`0D0` repro is weaker.** Its drive (`ambient_dyno_reverse`) is now in train, so the repro window may be in-sample.
5. **The final test mostly repeats known attacks.** The watch-list (`0D0`, `6E0`) comes from development attacks that target the same IDs as the final-test `_3` attacks. Only coolant (`4E7`) tests an unseen target. Report the two separately.
6. **Short attacks with 1 s windows.** The shortest attack (fuzzing_3, 0.65 s) often falls across a window boundary. Use `interval_overlap_s` to report partly covered windows separately.
7. **Freeze-family recall caps around 16%, root cause confirmed 2026-09-25.** `frozen_break`'s raw score is a fixed per-field constant (the field's full training-range width), scored by percentile rank; watched field `0D0` legitimately hits that exact value in 224/581 (38.6%) of normal training windows, so any freeze attack producing the same value only reaches the 61st percentile against a ~99.9th-percentile threshold. See the dedicated section above. Not fixable by a threshold change; needs a different raw signal (e.g. how-long-held, not which-value) -- out of scope now.

## What to do next (in order)

1. **Finish the branch.**
   - Superseded `results/crossval_ambient_provisional.json`, `crossval_ambient_splitsjson.json` and `crossval_attacks_splitsjson.json` have been moved to `results/archive/` (see `results/archive/README.md`), kept for the record but not for citing in the demo or final report. Current reports: `results/crossval_ambient.json`, `results/crossval_attacks.json`.
   - Run the tests (335 passed, reproducibility check passed against a fresh copy of the ROAD data, confirmed 2026-09-24).
   - Review `git diff`, with extra care on `shared/schemas.py` and `part1/`.
2. **Get agreement** on the five decisions above (Part 1 owner, Part 3 owner).
3. **Commit, push the branch, open a PR.** Mention in the commit message that the models were retrained.
4. **Save the other project in git.** Copy `~/Downloads/road/{edgeguard/, splits.json, docs/, tests/}` (not `dataset/` or `build/`) into this repo or its own repo.
5. **Stage 2 on decoded signals** (Part 2).
   - Out-of-range and flatline checks per decoded signal of the watch-list IDs, using `preprocess(window, Decoder)` at training and scoring time.
   - Percentile bounds instead of min/max, chosen by cross-validation (the other project found min/max bounds too loose).
   - Measure with `crossval attacks` and `crossval ambient`; keep only if recall rises without more false alarms.
6. **Reduce false-alarm sensitivity to training coverage.** Target `extended_short`'s 29 / 90. Percentile out-of-range bounds were tried and built (`bound_percentile`, see the Stage 2 hardening section above) but do NOT help -- the cause is a coverage gap (0D0 byte7's true range exceeds what 8 drives showed), not an outlier, so trimming the bound only hurts. Try wider training regime coverage (a manifest change, team decision) or excluding counter/checksum-like fields from out_of_range instead. (This is a different false-alarm problem from item 13's watch-list widening -- that one didn't move `extended_short` either way.)
7. **Report the plan's metrics.** Done: false alarms per hour (cross-validation reports, and the
   final_test-vs-crossval comparison in item 15 below), detection delay per attack (see the
   Detection delay section above). Still open: recall for fully vs partly covered windows
   (`interval_overlap_s` is already recorded on every GroundTruthLabel; nothing consumes it yet).
8. **Part 3 integration:** the injector works on a COPY of `RoadData` windows, re-windows with `windowing.windows_from_frames`, calls `preprocess()`, and the evaluator joins `DefenderOutput` with `GroundTruthLabel` by `window_id`.
9. **Done, 2026-09-24:** the one-time final_test v1-vs-v2 comparison has been run (see the Final evaluation section above, `results/final_evaluation.json`). Headline: v2 recall 64.9% vs v1 0.39%, but v2's false-alarm rate on final_test is 9.07% (well above the 1% target) and it still misses max_engine_coolant_temp (the one unseen-target attack) and reverse_light_off entirely. Worth digging into before presenting this as a clean win.
10. **Done:** real-traffic Nano benchmark (`nano_runner --model-version`, real ROAD windows via `RoadData`). Confirmed on this laptop: v1 mean 0.35 ms/window (~2,870 windows/s), v2 mean 1.74 ms/window (~575 windows/s), both on 815 real validation windows -- still needs the actual Nano hardware to confirm, not just this laptop.
11. **Done:** Block 3 (`part3/`, freeze + offset) and Block 4 integration (`integration/run_demo.py`, one command runs both the ordinary and test paths). See the Part 3 / Integration sections above for real-data numbers, including the unexplained 29.7% false-alarm rate on generic (non-watch-list-targeted) Red Team attacks, which is worth digging into before the demo.
12. **Done, 2026-09-24:** dashboard connected to real v2 Defender output (see the Dashboard section above); also fixed a `.gitignore` bug that had kept the dashboard's own data-handling source files out of git entirely.
13. **Done, 2026-09-24:** closed the red-vs-blue hardening loop end to end (see the Hardening loop
    and second evidence-gate sections above): `build_hardening_set()` is now actually called on a real
    evasion log (`part3/hardening_report.py`), a real candidate (`v5`) was built from it, and it was
    run through the evidence gate. Result: **v5 fails** -- identical real-attack recall to v2, but
    false alarms roughly triple on final_test (9.07% -> 28.3%). It looked like a clean, validated win
    against Part 3's own synthetic Red Team attacks (offset recall 29.7% -> 100%, no cost) but that
    does not transfer to the real ROAD final_test attacks, which target different CAN IDs entirely.
    **v2 remains the model to ship. Do not adopt v5.** Freeze-family detection (stuck at ~16% in every
    attempt so far) has its root cause confirmed 2026-09-25 -- see "Freeze-sensitivity root cause
    investigated" above and "Known problems and limits" #7 -- but fixing it needs a new raw signal
    for `frozen_break`, not attempted this close to the deadline. Any future watch-list change still
    needs validating against the full nested ambient LOCO, not a single-split spot check, before
    spending `final_test` on it a third time.
14. **Done, 2026-09-25:** wired the cloud-escalation path end to end (see the Cloud-escalation
    section above) -- `integration/run_demo.py --escalate --max-escalation-rate <rate>` now actually
    calibrates a band, sanitizes, calls the simulated cloud, and reports egress metrics, none of
    which happened before despite every underlying piece (`part1/escalation_policy.py`,
    `sanitizer.py`, `egress_metrics.py`, `mock_cloud_endpoint.py`) already being real and tested.
    This closes the HP Edge AI SJSUHack brief's explicit requirement for "an explicit, defensible,
    measurable decision about when to escalate to the cloud" (see `docs/`), which the team's own
    architecture doc had separately deprioritized -- **that de-scoping decision conflicts with the
    actual competition rubric and should be revisited with the team.** Found and fixed a real bug
    in `choose_band_half_width()` along the way (see the section above). **Still open:** the
    simulated cloud's fixed stand-in threshold (0.6) is badly mismatched to v2's real threshold
    (0.9988), so the real-data correction-rate number (97.9%) is not yet meaningful -- recalibrate
    the stand-in relative to each model's own threshold before quoting that number in the demo.
15. **Done, 2026-09-25, later corrected the same day:** README got a "Metrics -- how and why
    they were chosen" section (the brief asks for this by name). It originally compared final_test's
    326.4/hour false-alarm rate (`results/final_evaluation.json`'s v2 `updated`: fp=446, normal_hours
    = 4919/3600 = 1.3664h) against `results/crossval_attacks.json`'s "15.19/hour" --
    **that comparison used the wrong denominator: `normal_windows_in_attack_captures` is just 1 alarm
    over ~4 minutes of normal driving embedded in attack captures, far too small a sample to be a
    baseline.** Caught during the item 1-8 review pass (2026-09-25). Corrected: the robust comparison
    is `results/crossval_ambient.json`'s proper 9-capture leave-one-out estimate (30 alarms / 763
    windows = 141.5/hour), and the per-capture breakdown (see the dedicated section above) is now the
    headline evidence instead of any per-hour comparison -- 433/446 (97.1%) of v2's false alarms come
    from one highway capture, precisely because train/development have zero highway driving.
16. **Done, 2026-09-25:** escalation measured properly (see the Escalation quality section above):
    precision 48.1%, recall 28.0% (100% of false alarms, 14% of misses) on v2's Red Team test path,
    ~240 bytes and 0 raw CAN bytes per escalation. The stand-in cloud's 97.9% is now explicitly
    labelled as not a quality metric; no need to recalibrate it for the demo. README also reframed
    per the team's Revised Plan v3 D1/D3: the Nano is the OEM's on-prem security centre plus a
    simulated fleet (not in-car hardware), in-car latency on the Nano CPU is a lower bound, target
    user is the OEM vehicle security analyst.

## Rules for any code touching data

- **Split by capture, never by window.** Neighbouring windows of one drive are near-copies, so a window-level split leaks.
- **Get captures only through `RoadData` / `Manifest.capture_path()`**, which refuse final_test and separate captures. Only the final evaluation passes `final_evaluation=True`.
- **Never build a split from a glob, a seed or a hard-coded list.** The manifest is the split.
- A fabrication capture and its `_masquerade` twin are one recording and always share a split. Count normal windows from the fabrication capture only.
- Models, ranges and thresholds are fitted on **normal train/validation data only**. The threshold is never chosen from attack data or from in-sample scores.
- The Defender only sees neutral IDs (`cap07`, `veh01`) and never a `GroundTruthLabel`. File names contain the answer (`..._attack_1`), and `shared/schemas.py` rejects them.
- **Evaluation scores every window.** `keep_every > 1` is for training memory only.
- Cleaning removes only capture artifacts, never unusual content: unusual content is what attacks look like.
- Never overwrite a model that has been evaluated. Use a new version name (`--overwrite` only for never-evaluated models).
- Cross-validation numbers estimate the training *procedure*. Use them to choose settings (train/validation/development only), and report them beside the final-test number, never instead of it. Never tune on the final test.
- Detection results must be measured on real ROAD captures. Mock or synthetic numbers are labelled as such.

## Code style

Plain Python, pydantic models for contracts, small modules with a long docstring explaining the "why". Each module has a matching `tests/test_*.py` that uses fake data and never the real dataset. `tests/conftest.py` provides a shared fake ROAD folder (every capture kind, a fake DBC, metadata and a manifest). Match that when adding code.
