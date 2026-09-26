# EdgeGuard

CAN-bus intrusion detection small enough for a car's gateway computer, trained, attacked and hardened from the car manufacturer's own on-prem security centre, escalating to the cloud only when it is unsure — and never sending raw vehicle data off-site.

Built for the HP Edge AI SJSUHack 2026 (Secure AI track).

## Who this is for

**Target user:** a vehicle security analyst at a car manufacturer (OEM), in its vehicle security operations centre. UNECE R155 already requires manufacturers to monitor and respond to cyberattacks across their fleets, and only the manufacturer can push an improved detector back to its own cars.

**The problem:** CAN-bus intrusions (a spoofed sensor reading, a safety-critical signal frozen mid-attack) have to be caught in real time, inside the vehicle. A car is not always connected, cannot wait on a cloud round trip to flag an attack on its brakes or speedometer, and its raw internal bus traffic is proprietary telemetry that should not leave the vehicle by default.

**Why not cloud-only:**
- **Hard latency budget** — detection has to happen in milliseconds, not after a network round trip.
- **Intermittent connectivity** — a car loses signal; cloud-only means blind spots exactly when an attack might occur.
- **Data residency** — raw CAN logs and signal-decode files are the manufacturer's intellectual property, and fleet telemetry is personal data (where and how people drive). They stay on-prem. EdgeGuard's sanitizer (`part1/sanitizer.py`) enforces this as an allow-list: only a derived score, threshold, decision and evidence string may ever leave, never a raw frame.

EdgeGuard's answer: run detection locally, on the edge device, always. Escalate to a cloud second opinion only for the subset of cases the local model is genuinely uncertain about, and never block the local decision on that call succeeding.

## Architecture

Two modes, both scoring the same real-time signal from CAN traffic (see `CLAUDE.md` for the full design discussion):

- **Ordinary detection** — replay CAN traffic -> Defender (Stage 1 timing + Stage 2 payload checks, fused) -> local decision (ALERT / ISOLATE / FORWARD), with uncertain cases optionally escalated to a cloud second opinion.
- **Adversarial testing and hardening** — an Attacker proposes synthetic attacks against real driving captures, a constrained injector applies them, the Defender scores both, and confirmed misses feed a hardening loop that produces (and evaluates) candidate model updates against real, held-out data. (Code still calls this module `part3/red_team_agent.py` -- kept as-is to avoid a late rename across imports/tests; "Attacker"/"Defender" is the terminology used everywhere else, including in this document.)

```
[in-car ECU]  Fleet replay -> preprocess -> Defender (Stage 1+2) -> local decision
                                                  |
                                       uncertain? (escalation_policy)
                                                  v
[in-car ECU]               sanitize -> [cloud, off-vehicle]  second opinion
                                 (never blocks the local decision)
```

Everything in that diagram except the bracketed cloud step runs, in production, on the
vehicle's own gateway computer. For this hackathon demo the whole real-time path (fleet
replay, Defender, escalation) is simulated on the ZGX Nano instead of real car hardware --
and the Nano *separately* also plays the OEM's on-prem security centre for training, Red
Team testing and hardening (next section). Two different roles on the same physical box;
neither one is an in-car deployment.

### Where the ZGX Nano sits

The ZGX Nano is a desktop-class workstation, not automotive hardware — one per car is not a product, and we don't claim it is. In EdgeGuard it plays two roles:

- **The manufacturer's on-prem security centre** (real): training, Attacker testing, the hardening evidence gate and the escalation decision all run here, so raw CAN data never leaves the building.
- **A simulated fleet** (demo only): it replays real ROAD captures through the in-car detector, standing in for the cars.

In one line: *vehicles simulated; the Nano is the security centre.*

Components:

| Part | What it does |
|---|---|
| `part1/` | Real ROAD data pipeline, windowing, decoding, the cloud-escalation path (`escalation_policy.py`, `sanitizer.py`, `mock_cloud_endpoint.py`, `egress_metrics.py`) |
| `defender/` | The detector itself: Stage 1 (timing), Stage 2 (payload range/frozen/jump checks), fusion, training, cross-validation |
| `part3/` | Attack proposal + injection (the Attacker), evasion logging, hardening-set construction |
| `integration/` | `run_demo.py` (one command, both modes) and `run_final_evaluation.py` (the one-time held-out evidence gate) |
| `dashboard/` | Live view of real Defender output (React + Vite) |

## Data splits

Five disjoint sets of real ROAD captures, by `capture_id` (`part1/split_manifest.json`):

| Split | Captures | Used for |
|---|---|---|
| `train` | 7 | Fit Stage 1 + Stage 2 reference distributions and normal ranges |
| `validation` | 2 | Choose the alert threshold and the escalation-band width |
| `development` | 18 | Leave-one-out cross-validation, Attacker attack testing, the hardening loop |
| `final_test` | 14 | One-time held-out evidence gate -- frozen, touched exactly twice so far (v1-vs-v2, then v2-vs-v5) |
| `separate` | 4 | Accelerator (Nano) benchmark captures only; not used in any accuracy number |

**Still an open team decision, disclosed rather than hidden:** the build plan itself is ambiguous about whether the threshold should be tuned on `validation` or `development` data (one section names each). We read it as threshold-from-`validation`, attacks-and-misses-from-`development`, and that is what the code does -- but the team has not formally signed off on that reading. See `CLAUDE.md`'s "Decisions the team must confirm" section.

## Metrics — how and why they were chosen

The brief asks for benchmarks "including how and why they were chosen." This is that answer; the numbers themselves are in the sections below and in `CLAUDE.md`.

| Metric | How we measure it | Why this metric |
|---|---|---|
| Recall, per attack family | Attacks detected ÷ attacks injected, per family, on the frozen `final_test` set (real ROAD attacks, touched once) and cross-validated on development attacks | One overall accuracy number hides which attack types are and aren't caught — `max_engine_coolant_temp` and `reverse_light_off` sit at 0% while others are near 100% |
| False alarms per hour | HIGH-severity decisions on normal driving ÷ hours of normal driving. Reported two ways: a leave-one-out cross-validated estimate on `development` (used to choose settings) and the frozen model's rate on `final_test` (used to report the result) — never one instead of the other | An IDS that cries wolf gets switched off; a rate from the training procedure alone can hide how it behaves on driving it has never seen — see "Where final_test's false alarms actually come from" below for the real per-capture breakdown |
| Detection latency | Inference time only (window collection excluded), mean/median/p95/max, measured on real ROAD windows on a laptop and on the actual ZGX Nano | The in-car safety claim is about milliseconds, and window time would dominate and hide the number that matters |
| What each layer adds | v1 (Stage 1 timing only) vs. v2 (Stage 1 + Stage 2 payload checks), same frozen `final_test` set | Shows Stage 2 is what catches anything beyond fuzzing (0.4%→64.9% recall) — the reason the second stage exists. (A rules-alone baseline is not broken out separately in this build; Stage 1 already includes the frequency/range checks.) |
| Car-side footprint | Trained model size (JSON parameters): 43 KB (v1) / 70 KB (v2, both stages); no compiled ML runtime dependency (plain Python, no numpy) | Supports "fits a gateway ECU": latency is stated as measured on the Nano's CPU (a lower bound, not proof), so size and dependency-freedom are the more portable evidence |
| Escalation quality | Escalation precision/recall against Attacker windows with known labels (`part1/escalation_quality.py`) — no cloud needed for this measurement — plus bytes sent per escalation | The brief's "explicit, defensible, measurable" escalation requirement; this is direct evidence rather than a proxy, unlike the simulated cloud's disagreement rate |
| Hardening | Evasion rate before/after a candidate update, on the held-out Attacker round the candidate was NOT built from, plus the frozen `final_test` evidence gate (recall must not drop, false alarms must not rise) | Proves whether a hardening loop actually improves robustness rather than overfitting the very evasions that triggered it — our own `v5` candidate looked like a clean win synthetically and failed this exact gate on real attacks |

## Quick start

```bash
git clone https://github.com/Psycoder0611/EdgeGuard.git
cd EdgeGuard
bash setup.sh                    # creates a venv, installs dependencies
source .venv/bin/activate        # in plain sh (e.g. on the Nano): . .venv/bin/activate
```

> **Nano note:** the Nano's default SSH shell is `/bin/sh` (dash), which doesn't reliably keep `source .venv/bin/activate` (or `conda activate`) active across commands -- you'll see `python: not found` right after one appears to work. The fix that always works regardless of shell: skip activation and call the venv's interpreter by its full path for every command (`.venv/bin/python -m ...`, never bare `python`). Full runbook: "Running on the Nano" below.

Train the models (real ROAD normal-driving data only; see `CLAUDE.md` for where to get it):

```bash
python -m defender.run_training --max-false-alarm-rate 0.01 --data-dir <path-to-road>
```

Run the demo — ordinary detection, Attacker attack testing, and (optionally) live cloud escalation, all in one command:

```bash
# terminal 1: the simulated cloud second opinion
python -m uvicorn part1.mock_cloud_endpoint:app --host 127.0.0.1 --port 9000

# terminal 2: the demo itself
python -m integration.run_demo --model-version v2 --data-dir <path-to-road> \
    --escalate --max-escalation-rate 0.3
```

Measure whether escalation targets the detector's real mistakes (no cloud needed — scored against the Attacker's known labels):

```bash
python -m integration.run_demo --model-version v2 --data-dir <path-to-road> \
    --mode test --escalate --max-escalation-rate 0.3
```

Run the dashboard:

```bash
cd dashboard && npm install && npm run dev
```

## Real results (real ROAD data, never synthetic numbers presented as real)

One-time held-out evaluation (`final_test`: 14 captures, 5,178 windows -- 259 attack, 4,919 normal -- never touched during development; see `CLAUDE.md`'s Final evaluation section):

| | v1 (Stage 1 only) | v2 (Stage 1 + Stage 2) |
|---|---|---|
| Recall | 0.39% | **64.9%** |
| False alarms | 0.41% | 9.07% |
| Precision | 4.8% | 27.4% |
| F1 | 0.7% | 38.5% |
| Mean inference | 0.46 ms | 2.26 ms |

v2's Stage 2 payload check is what catches anything beyond fuzzing attacks — the whole reason this project exists. It is also honestly not finished: false alarms on final_test (9.07%) are well above the 1% target, and two attack types are still missed entirely.

**Where final_test's false alarms actually come from.** The 446 false alarms above are not spread evenly across final_test's 3 normal captures (`part3/per_capture_metrics.py`, closing `part3/metrics.py`'s own TODO that overlapping windows are correlated and need capture-level reporting, not one blended number):

| Capture | Windows | False alarms | Rate |
|---|---|---|---|
| `ambient_highway_street_driving_long` | 3,764 | 433 | 11.5% |
| `ambient_highway_street_driving_diagnostics` | 469 | 9 | 1.9% |
| `ambient_dyno_drive_basic_short` | 444 | 4 | 0.9% |

433 of 446 (97.1%) of every false alarm v2 raises on final_test comes from one single highway capture; the two highway captures together account for 442 of 446 (99.1%), while the one non-highway capture sits at a 0.9% rate -- in line with the proper leave-one-out estimate on `development` driving (30 alarms / 763 windows over 9 held-out drives = 141.5/hour, `results/crossval_ambient.json`). This is a training-coverage gap, not model noise: `train`/`development` contain no highway driving at all (both real highway drives are in `final_test`), so Stage 2's payload ranges have never seen highway-speed values for the fields they watch. The blended 9.07% headline hides this; the per-capture number doesn't.

**Detection delay** (build plan requirement, same frozen `final_test` scoring pass, no re-run of anything): time from the real attack's onset to the first ATTACK-decision window closing, measured only on the 11 `final_test` captures with a real, continuous injection interval (fabrication/masquerade attacks — fuzzing has no single onset to measure from, so it's excluded).

| | v1 | v2 |
|---|---|---|
| Captures ever detected | 5 of 11 | 6 of 11 |
| Mean delay, when detected | 28.2 s | **0.57 s** |
| p95 delay, when detected | 39.5 s | 0.93 s |

v1 does eventually flag some attacks, but only after the payload-free timing signal happens to drift far enough — tens of seconds in. v2 catches what it catches almost immediately (under a second on average). Read this together with recall, not instead of it: these averages are each over a different, small subset of captures (the ones that model actually detected at all), not the same 11 for both — a fair comparison, not a cherry-picked one, but a small-n one.

## Running on the Nano

Code, the ROAD dataset and model checkpoints get to the Nano by `rsync`, not `git` -- `data/road` and `models/timing_cnn_v1` aren't tracked in git, and `rsync` moves code + data + models in one shot instead of a `git pull` plus separate manual copies:

```bash
rsync -avz --exclude '.venv' --exclude '__pycache__' --exclude '.git' \
    ~/Desktop/EdgeGuard/ <nano-user>@<nano-ip>:~/EdgeGuard/
ssh <nano-user>@<nano-ip>
cd ~/EdgeGuard && bash setup.sh
```

Then, for every command on the Nano:

- **Never rely on `source .venv/bin/activate` or `conda activate` sticking.** The Nano's SSH shell is `/bin/sh`, not bash, and doesn't source `.bashrc`/`conda.sh` on login -- both appear to run with no error, then the very next command says `python: not found`. Always call the venv's interpreter by its full path instead: `.venv/bin/python -m defender.run_training ...`, never bare `python`. This works regardless of shell, and regardless of whether a `conda`/`miniforge3` install also exists on the box from earlier work (ignore it -- it's a separate Python not kept in sync with `requirements.txt`).
- **`torch` is intentionally not in `requirements.txt`.** Only `defender/timing_cnn.py`, `defender/defender_llm.py` and `defender/fused_defender.py` need it -- v1/v2 deliberately have no ML runtime dependency (see the model-size row below). Install it once, inside the venv: `.venv/bin/python -m pip install torch`. On the GB10 the plain PyPI wheel pulled a full CUDA build automatically (`torch==2.14.0+cu130`) -- no special ARM/Jetson wheel needed on this hardware.
- **A system-wide `pip install` (outside the venv) refuses with `externally-managed-environment` (PEP 668).** That means you forgot the `.venv/bin/python -m pip install ...` prefix, not that anything is broken -- no need for `--break-system-packages` if you stick to the venv's own pip.

Benchmark commands (`defender/nano_runner.py`'s own docstring has every flag):

```bash
# 1. sanity check -- pipeline runs at all, no data needed
.venv/bin/python -m defender.nano_runner --mock --output results/nano_benchmark_MOCK.json

# 2. real latency, plain v2 (no torch needed)
.venv/bin/python -m defender.nano_runner --model-version v2 --data-dir data/road \
    --group validation --output results/nano_benchmark_REAL_v2.json

# 3. real latency, the shipped fused detector (v2 OR TimingCNN -- needs torch, see above)
.venv/bin/python -m defender.nano_runner --cnn-model-dir models/timing_cnn_v1 --model-dir models \
    --data-dir data/road --group validation --output results/nano_benchmark_REAL_fused.json
```

Pull results back the same way, reversed: `rsync -avz <nano-user>@<nano-ip>:~/EdgeGuard/results/ ~/Desktop/EdgeGuard/results/`.

## Measured on the ZGX Nano GB10

Run for real over SSH on the competition node (`spark-3f6`, aarch64): `bash setup.sh`, then both models scored against the same 815 real ROAD validation windows on-device (`results/nano_benchmark_v1_real.json`, `results/nano_benchmark_v2_real.json`):

| | v1 | v2 | Fused (v2 OR TimingCNN) |
|---|---|---|---|
| Mean inference | 0.227 ms | 1.325 ms | 6.345 ms |
| p95 inference | 0.284 ms | 1.658 ms | 7.884 ms |
| Throughput | 4,406 windows/s | 755 windows/s | 158 windows/s |
| Model size | 43 KB (JSON) | 70 KB (JSON) | 70 KB JSON + 16 KB torch checkpoint |

**What these numbers do and don't show.** The Nano's CPU is far faster than a car's gateway computer, so these latencies are a *lower bound* on in-car latency, not proof the detector fits one. The more portable evidence is size: the whole v2 detector is about 70 KB of parameters with no ML runtime dependency (plain Python — no numpy, no compiled libraries). For the security-centre side, the numbers show a single Nano can re-score fleet traffic far faster than real time.

The fused detector (`defender/fused_defender.py`, `results/nano_benchmark_REAL_fused.json`) trades some of that footprint for recall: 76.4% on `final_test` vs v2 alone's 64.9%, at the same false-alarm rate (see "Real results" above) -- but it pulls in PyTorch as a runtime dependency, which v1/v2 deliberately avoid, and costs ~5x the latency (still only 6.3 ms mean, 158 windows/s -- comfortably real-time for a 1-second window, just no longer a "fits a gateway ECU with no ML runtime" claim).

## Is escalation sending the right windows to the cloud?

The HP brief asks for "an explicit, defensible, measurable decision about when to escalate." Our rule: escalate a window when its attack score falls within a band around the threshold; the band is the widest one that keeps escalations under a budget (30% here) on normal *validation* traffic. The local decision is always made first and never waits on the cloud.

To measure whether that rule targets real mistakes, we score it against windows whose true label is known — **the Attacker's synthetic attacks injected into real `development` captures, not the real ROAD `final_test` attacks reported above** (`part1/escalation_quality.py`, `results/escalation_quality_v2/demo_test.json`). No cloud is involved in this measurement.

| v2, 2,511 windows (the Attacker's synthetic attacks on `development` captures) | |
|---|---|
| Escalated | 898 (35.8%) |
| Escalation precision (escalated windows the local model got wrong) | 48.1% (432 of 898 escalated) |
| Escalation recall (local mistakes that were escalated) | 28.0% |
| — of false alarms | **249 of 249 (100%)** |
| — of missed attacks | 183 of 1,293 (14%) |
| Error rate outside the band | 68.8% |
| Accuracy: local only → ceiling with a perfect second opinion | 38.6% → 55.8% |
| Sent per escalation | 241 bytes mean (max 269), **0 raw CAN bytes** |

**What this says, honestly:**

- **Escalation catches every false alarm.** Every one of v2's 249 false alarms falls inside the band, so a second opinion gets to review each one before it reaches the analyst — directly aimed at alert fatigue.
- **It cannot rescue confident misses.** 86% of missed attacks score below the uncertainty band, and the error rate *outside* the band (68.8%) is higher than inside it (48.1%). A score-based second opinion never sees those windows. They are mostly a detection-coverage problem — in the hardening analysis, 70% of v2's misses on the Attacker's attacks were on signals Stage 2 doesn't watch — and escalation is no substitute for coverage.
- **What leaves the site is tiny and contains no raw data:** about 240 bytes per escalation, limited by an allow-list to `attack_score, decision, evidence, model_version, threshold, window_id`.
- **Caveat on the mix:** the test set is the Attacker's synthetic attacks on development captures, two attacked windows per normal one, so the accuracy figures reflect that mix, not real-world attack rates. On normal traffic alone (the Nano ordinary run), the same band escalated 240 of 815 windows (29.4%, within the 30% budget).

## What EdgeGuard does not yet catch, and why

- **`max_engine_coolant_temp` (0% recall, both models).** The one final-test attack target Stage 2 never watches. It confirms the watch-list does not generalize to an unseen CAN ID — a real, disclosed limitation, not a hidden one.
- **`reverse_light_off` (0% recall).** A one-bit signal flip inside a byte. Stage 2's payload checks work at the byte level; catching this needs decoded-signal checking (`part1/decode.py` exists and is tested, but nothing wires it into Stage 2 scoring yet). **Deliberately not attempted this close to the deadline:** any change to Stage 2's scoring is a new model, which means spending `final_test` a third time to validate it honestly -- the same risk that sank `v3`/`v4`/`v5` under time pressure. Documented as the clear next step, not attempted as a last-minute fix.
- **Freeze-family attacks cap out around 16% recall, and we found out why (2026-09-25 diagnostic, no code change).** `frozen_break` scores a freeze by its percentile rank against every normal training window, but the watched field `0D0` legitimately sits pinned at its single most-extreme historical value in 224 of 581 (38.6%) normal training windows -- real dyno driving genuinely produces long runs of an unchanging value on this signal. A synthetic freeze attack pinning the same field to that same value therefore only ranks at the 61st percentile of "normal," nowhere near the ~99.9th-percentile alert threshold, even though the underlying frozen-value check itself fires correctly every time. A real sensitivity limit of percentile-rank scoring on this vehicle's driving data, not a bug -- full diagnostic in `CLAUDE.md`.
- **A real hardening attempt (`v5`) failed its evidence-gate test.** We built a candidate from confirmed Attacker-attack misses, validated it looked great against the Attacker's own synthetic attacks, then ran it against the real held-out attacks: identical recall, nearly triple the false alarms. Full story, including why, in `CLAUDE.md`'s Hardening loop section — we're documenting this because a negative result honestly reported is worth more than a cherry-picked positive one.
- **Confident misses are invisible to escalation** (see the escalation section above): the uncertainty band only sees windows near the threshold.
- **The cloud second opinion itself is simulated.** The escalation path is wired and real (calibrated on validation data, sanitizes before sending, keeps working when the cloud is unreachable — tested on a laptop and on the Nano), but the stand-in cloud uses a fixed threshold not calibrated to our model, so the rate at which it "disagrees" (97.9%) is not a quality metric and we don't use it as one. The measured escalation numbers above don't depend on it.
- **One vehicle, on a dynamometer.** ROAD is recorded from a single car; generalization to other vehicles is untested.
- **In-car latency is measured on the Nano's CPU as a proxy,** not on a real gateway computer (see the hardware section).

## Full engineering log

`CLAUDE.md` is the living build log: every real result, every negative finding, every open decision the team still needs to make, with real numbers and honest caveats throughout. Start there for anything this README doesn't cover.
