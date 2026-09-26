# EdgeGuard Defender (Part 2: Blue Team)

The Defender looks at one window of CAN traffic and returns **one attack score**, a threshold, a decision (`ATTACK` / `ACCEPT`) and a short evidence text. It runs fully locally, with no cloud connection needed for detection.

> **Status:** v1 and v2 are trained on real ROAD data using the **frozen split manifest** (`part1/split_manifest.json`) and Part 1's non-overlapping 1 s windows. v2 is the candidate for Part 3's evidence gate. A v3 hardening attempt was tested and not adopted (see below). All results below come from development captures and cross-validation only. Official results come from Part 3's evidence gate on the final test set.

---

## How it fits in the system

```
ROAD raw frames
      ↓
Part 1: cleaned, non-overlapping 1 s TrafficWindows  (part1.pipeline.RoadData)
      ↓
Part 3 (testing only): inject attack into a COPY
      ↓
Part 1: same preprocessing
      ↓
Part 2: score_window(window) -> DefenderOutput      <- this part
      ↓
Simulated consumer (alert / isolate / forward)  and  Part 3 evaluator (joins by window_id)
```

The Defender never receives labels, attack family, injection interval or the attack specification. The shared schema (`shared/schemas.py`) rejects any such field.

---

## What the Defender contains

| Version | Components | Purpose |
|---|---|---|
| **v1** | Stage 1 (timing) | Baseline |
| **v2** | v1's **unchanged** Stage 1 + new Stage 2 (payload), max fusion | Improved candidate |

**Stage 1: timing** (`defender/stage1.py`) learns from normal traffic, per CAN ID: which IDs exist, normal message rate and normal gap between frames. It flags unknown IDs, too many frames and gaps that are too short.

**Stage 2: payload** (`defender/stage2.py`) learns, per watched CAN ID and per field (each byte, and each byte pair as a 16-bit value in both byte orders), the normal value range and the largest normal jump. It flags out-of-range values, a normally-varying field frozen for a whole window, and abnormally large jumps.

**Fusion** (`defender/fusion.py`) takes the **higher** of the two scores. Evidence is labelled `[Stage 1]` or `[Stage 2]`.

**Scores** mean "how unusual compared with normal training windows" (0 to 1). They are **scores, not calibrated probabilities**.

---

## Key design decisions

1. **v2 reuses v1's Stage 1 unchanged.** `train_v2` loads v1's Stage 1, requires the same train and validation captures, and refuses otherwise. The only difference between v1 and v2 is Stage 2, so any improvement can be attributed exactly.
2. **Same false-alarm budget.** v2 uses v1's target false-alarm rate by default.
3. **Threshold chosen on validation only.** `choose_threshold()` picks the lowest threshold keeping `false alarms / normal validation windows <= target`. It is saved together with its model version and refused for any other version.
4. **Stage 2 watch-list: `0D0`, `6E0`.** These are the CAN IDs targeted by the **development** attacks only. On real data, watching all ~100 IDs flagged **170 of 184** normal validation windows (every new drive has some field doing something new). Watching only `0D0, 6E0` flagged **0 of 184**.
5. **Stage 2 learns ranges from every training frame**, not only sampled windows, streamed line by line so memory stays low.
6. **Models are never overwritten silently.** Retraining an existing version needs `--overwrite`.

---

## Setup

```bash
python -m venv .venv
# Windows:      .venv\Scripts\Activate.ps1
# Linux / Nano: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests -q
```

### ROAD data (not in this repository)

Download the ROAD dataset (Verma et al., ORNL) and place it so that these exist:

```
data/road/ambient/*.log
data/road/attacks/*.log
data/road/attacks/capture_metadata.json
```

`data/` is in `.gitignore` and is never uploaded.

---

## Commands

| Task | Command |
|---|---|
| Run all tests | `python -m pytest tests -q` |
| Train v1 and v2 on real data | `python -m defender.run_training --max-false-alarm-rate 0.01` |
| Development check (v1 vs v2, and v3 if trained) + Part 3 freeze repro | `python -m defender.dev_check --data-dir <road>` |
| Hardening experiment: train only v3 (v1 and v2 untouched) | `python -m defender.run_training --max-false-alarm-rate 0.01 --harden-v3` |
| Local run + latency benchmark (MOCK traffic) | `python -m defender.nano_runner --mock --output results/nano_benchmark_MOCK.json` |
| Diagnostic: Stage 2 on normal validation | `python -m defender.diagnose` |
| Diagnostic: test a watch-list | `python -m defender.diagnose_watch --watch "0D0,6E0"` |
| Cross-validation: leave-one-capture-out false alarms (ambient) | `python -m defender.crossval ambient --max-false-alarm-rate 0.01` |
| Cross-validation: 2-fold attack CV over development recordings | `python -m defender.crossval attacks --max-false-alarm-rate 0.01 --data-dir <road>` |

In PowerShell, **quote** CAN ID lists (`"0D0,6E0"`). Otherwise `6E0` is read as the number 6.

Training settings (build plan): 1.0 s **non-overlapping** windows from Part 1, at most 100 evenly spaced windows per training capture (about 1 MB of memory per real window; evaluation always scores every window). The Defender records the window length in `train_info_<version>.json` and refuses windows of any other length. Training takes a few minutes on a laptop.

**Cross-validation** (`defender/crossval.py`) refits the models in every fold and never reads test captures:
- `ambient`: leave one normal capture out. The threshold is chosen on out-of-fold scores of the remaining captures (nested leave-one-out), then false alarms are counted on the held-out capture. It gives a false-alarm estimate over every train and validation drive, not just the 2 validation drives.
- `attacks`: the manifest gives every development attack recording a fold (its index, `_1` or `_2`). Fold k is held out, the Stage 2 watch-list comes from the other fold's attack targets only, and every held-out window is scored against Part 1's labels.

These numbers estimate the training *procedure*. Use them to choose settings, and report them beside the final-test result, never instead of it. Never tune on the final test.

---

## Using the Defender from other parts

```python
from defender.defender import Defender, set_active_defender, score_window

set_active_defender(Defender.load("models", "v2"))   # or "v1"
output = score_window(window)   # TrafficWindow or dict; label fields are rejected
```

`DefenderOutput` fields: `window_id`, `attack_score`, `threshold`, `decision`, `evidence`, `model_version`, `latency_ms`.

`latency_ms` is Defender inference time only. Window collection time is not included.

**Windows and labels come from Part 1** (`part1.pipeline.RoadData`): window times are already **elapsed seconds from the capture start**, the same clock as ROAD's `injection_interval`, and `labelled_windows()` gives each window its private `GroundTruthLabel`.

**ID format:** ROAD metadata writes `0xd0`, while the logs and schema use `0D0`. Use `defender.stage2.normalize_can_id()` to convert.

---

## Split: the frozen manifest

The split is `part1/split_manifest.json` (built by `python -m part1.split_manifest build`; see `part1/split_manifest.py`):

| Group | Normal drives | Attacks |
|---|---|---|
| train | drive_basic_long, drive_extended_long, drive_radio_infotainment, idle_radio_infotainment, exercise_all_bits, drive_winter, dyno_reverse | none |
| validation | drive_benign_anomaly, drive_extended_short | none |
| development | none | `_1` and `_2` of max_speedometer, reverse_light_off, reverse_light_on, correlated_signal (each with its `_masquerade` twin), fuzzing_1, fuzzing_2 |
| final_test | drive_basic_short, highway_street_driving_diagnostics, highway_street_driving_long | all `_3` recordings, max_engine_coolant_temp (+ masquerade) |
| separate | none | 4 accelerator captures (no injected frames) |

The normal-drive groups come from `~/Downloads/road/splits.json`, which assigns them by measured coverage of driving regimes. The coolant attack (target `4E7`) is in final_test as an honest test of an **unwatched** target ID. Both highway drives are in final_test, so training has no highway driving (an out-of-domain test for false alarms).

---

## Earlier results (provisional split, 0.5 s overlapping windows): superseded

These were measured before the frozen manifest and Part 1's windows existed, on the old provisional split. They are kept for the record; the models they describe have been replaced. Window-level counts (1 s windows, 0.5 s stride, so windows overlap). A window counts as attacked if it overlaps the injection interval. Attack captures are checked in full; long ambient captures are sampled (at most 300 windows).

| Attack family (normal + masquerade) | v1 detected | v2 detected |
|---|---|---|
| max_speedometer | 0 / 100 | **100 / 100** |
| correlated_signal | 4 / 88 | **88 / 88** |
| fuzzing | 8 / 8 | 8 / 8 |
| reverse_light_off / on | 0 / 114 | 0 / 114 |
| **All development attacks** | **12 / 310 (4%)** | **196 / 310 (63%)** |

| Normal windows | v1 false alarms | v2 false alarms |
|---|---|---|
| Normal windows in attack captures + dyno_reverse | 15 / 636 | 15 / 636 |
| dyno_exercise_all_bits (deliberately unusual, 290 sampled windows) | 1 / 290 | 21 / 290 (7.2%) |
| **Total** | **16 / 926** | **36 / 926** |

An earlier run over all 4,344 exercise_all_bits windows gave the same picture: v1 15 (0.3%), v2 333 (7.7%).

Validation (threshold selection, target 1%): v1 and v2 both have threshold 0.964225 and 0 / 184 false alarms.

---

## Red-team finding and hardening attempt (v3)

**Finding (Part 3, development split):** in the first ambient_dyno_reverse window, Part 3 froze all 93 `0D0` frames at `3A710460F5000000` (bytes 1 and 5 changed in every frame; original values were ramping, e.g. `42710460F4000000`, `4A710460F3000000`). Both v1 and v2 accepted it, with exactly the same score as the untouched window.

**Why v2 missed it:** the frozen values are inside the normal range, there are no jumps inside a fully frozen window, and v2's frozen check measures a field's range width, which does not make these bytes stand out.

**Hardening attempt (v3):** a "rate" frozen check that scores how rare it is for a field to stay constant for a whole normal window. v3 = v1's Stage 1 + Stage 2 with this check, same captures, watch-list and false-alarm target.

**Result: no improvement, so v2 remains the candidate.**

| | v2 | v3 |
|---|---|---|
| Development detected | 196 / 310 | 196 / 310 |
| Development false alarms | 36 / 926 | 36 / 926 |
| Part 3 freeze repro | ACCEPT | ACCEPT |
| Threshold | 0.964225 | 0.986371 (higher) |

**Why:** `0D0` bytes 1 and 5 are constant in **7 of 574** normal training windows (about 1.2%), most likely when the vehicle is steady. A one-second freeze therefore looks like normal steady driving. v3 ranked the frozen window above about 98.8% of normal windows, which is not enough under a 1% false-alarm target. Catching it at window level would cost roughly 1.2% extra false alarms from this signal alone.

**What could separate them (future work):** cross-signal context (the signal freezes while related signals keep changing) or persistence over several consecutive windows.

v3 model files are not kept; the "rate" option remains in the code (`--harden-v3`) so the experiment can be reproduced.

---

## Known limitations

- **Reverse-light attacks are not detected** by v1 or v2. They set a flag to a value inside the normal range learned by Stage 2.
- **Freezing a watched signal at a plausible value is not detected** (Part 3's development finding). Within one second it looks like the vehicle holding steady (see the v3 section).
- **Stage 1 alone catches fuzzing only** on real data. ROAD fabrication attacks do not disturb timing enough to stand out.
- **Attacks on unwatched CAN IDs are invisible to Stage 2** (for example the coolant masquerade on `4E7`). Stage 1 may still see extra frames.
- **Accelerator captures** contain no injected frames and are outside the main results.
- **More false alarms on extreme-but-benign traffic** with v2 (7.7% on `exercise_all_bits`).
- **Development-only results.** Final numbers come from Part 3's evidence gate on the final test.
- Stage 1 and Stage 2 are statistical detectors running on the CPU. On the Nano they run locally without a GPU.

---

## Pending

- Nano benchmark with **real** windows (currently mock mode only).
- Evidence gate (Part 3): compare v1 and v2 on the final test set.

## Data citation

Verma, M. E., et al. *ROAD: The Real ORNL Automotive Dynamometer Controller Area Network Intrusion Detection Dataset.* arXiv:2012.14600, 2020.