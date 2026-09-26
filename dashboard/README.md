# EdgeGuard dashboard (Member 4: dashboard and demo UI)

React + Vite dashboard for the EdgeGuard demo. Wired to a REAL replay of
`DefenderOutput` records by default (`DATA_SOURCE = "live"` in `App.jsx`) --
real ROAD development captures, scored by the real trained v2 Defender, not
synthetic mock data. The mock generator (`src/data/mockRun.js`) is still
available by flipping `DATA_SOURCE` back to `"mock"` for offline UI work
that doesn't need real numbers (see "Data sources" below).

## Layout

- **System architecture (live)** — always visible at the top. Animates your
  team's own architecture doc: the ordinary-detection lane (Fleet replay →
  Blue Team → Simulated consumer) pulses live, in sync with each replayed
  window, since that's the lane that actually runs per window. The
  adversarial testing/hardening lane (Red Team → injector → ground truth →
  evidence gate) is shown as a static, dimmed strip underneath, labeled
  "offline" — that lane runs once, comparing two frozen model versions, not
  per window, so animating it live would misrepresent how the pipeline
  actually works. See `src/components/ArchitectureDiagram.jsx`.
- **KPIs tab** — the original metrics dashboard (below).
- **Fleet view tab** — a glowing wireframe car on a route (`CarsView.jsx`),
  cyan while normal, red mid-attack, with a checkpoint gate (the Blue Team
  detector) and incoming Red Team arrows. Dark by design — the neon glow
  effect needs a dark surface to read.

## Sections (matches the task card)

- **Replay status** — capture id, window index/progress, play/pause/reset, speed.
- **Attack score** — current `attack_score` vs. `threshold`, labeled as a score
  (not a probability) per the team's own schema comment.
- **Affected CAN ID** — best-effort, parsed out of the free-text `evidence`
  field (see `src/utils/parseEvidence.js`). There's no dedicated CAN-ID field
  in `DefenderOutput` yet, so this can silently return nothing for evidence
  strings that don't say "ID <hex>" (e.g. Stage 1's `unknown_ids` check).
  Worth raising with whoever owns `defender/stage1.py`/`stage2.py`/`fusion.py`
  if the team wants this to be reliable rather than best-effort.
- **Simulated response** — mirrors `defender/simulated_consumer.py`'s three
  actions (`SIMULATED_ALERT` / `SIMULATED_ISOLATION` / `SIMULATED_FORWARD`).
  The Alert/Isolate toggle mirrors that module's `attack_action` constructor
  argument, which the team said was a "team setting."
- **Local / cloud routing** — presentational only. There is no cloud
  escalation path anywhere in the current codebase (confirmed by reading
  `defender/`), and the team's own architecture doc scopes that out as an
  open decision. Shown here as a placeholder so the layout doesn't need
  reworking once that decision is made.
- **Alert timeline** — a scrolling strip of every window played so far,
  color-coded, hover/focus for per-window detail (window id, score, evidence).

## Running it

```bash
npm install
npm run dev       # local dev server
npm run build     # production build -> dist/
```

## Mock data

`src/data/mockRun.js` generates one deterministic 240-window replay (same
sequence every reload — useful for rehearsing before the real recording) with
three multi-window attack bursts, since a real injected attack spans several
consecutive 1-second windows, not one isolated blip. The threshold value
(`0.9642248722316866`) matches the mock example already shared.

## Data sources

- **`"live"` (default)** -- `src/data/liveFeed.js` loads `src/data/realRun.json`,
  a precomputed replay built by `dashboard/scripts/build_real_run.py`: it
  scores four real ROAD development captures (`correlated_signal_attack_2`,
  `max_speedometer_attack_1`, `reverse_light_on_attack_2`,
  `reverse_light_off_attack_1`) end to end with the real trained v2 Defender
  (`defender/defender.py`) and writes out real `attack_score` / `evidence` /
  `latency_ms` per window, in order. 216 windows, 86 flagged ATTACK.
  The last capture is a deliberate **hard case**: v2 misses all 8 of its
  attack windows. So that a miss is visible at all, each window also carries
  an evaluator-side `truth` field ("attack" / "normal") from Part 1's private
  labels, added AFTER scoring and never passed to the Defender. The dashboard
  uses it only to mark mistakes in amber (8 missed attacks, 1 false alarm in
  `reverse_light_on_attack_2`); every decision shown is still the Defender's
  own. `meta.latencyMeasuredOn` records which machine timed the run. This is a REPLAY of a precomputed run, not a
  socket to a running Nano -- there is no live streaming inference process
  anywhere in this repo yet, so "live" means "real recorded Defender output,"
  not "real-time." `App.jsx` loads it asynchronously (`connectLiveFeed()`
  returns a Promise) and shows a brief loading state first.
- **`"mock"`** -- `src/data/mockRun.js`'s deterministic generator, for UI work
  that shouldn't depend on real data being present or on retraining.

To regenerate `realRun.json` (e.g. after retraining v2, or to use different
captures), from the EdgeGuard repo root:

```bash
python -m dashboard.scripts.build_real_run --data-dir ~/Downloads/road
```

No component under `App.jsx` needs to change for either data source -- they
only ever consume `DefenderOutput`-shaped objects.

## Known gaps / open questions (worth raising at the team discussion)

- **No live streaming pipeline.** `"live"` replays a precomputed
  `realRun.json`, not a socket to a running Nano -- see "Data sources"
  above. Wiring an actual live stream is separate future work.
- CAN ID parsing is best-effort text parsing, not a real field — see above.
  Checked against every unique evidence string in the real replay: 26 of 28
  resolve a CAN ID; the other 2 are the "no anomaly" strings, which
  correctly have none to report.
- Local/cloud routing has no backend; it's UI-only until the team decides
  the HP edge/cloud requirement.
- `part1/windowing.py` locks window/stride to 1.0s/1.0s ("no overlap"), but
  `defender/README.md`'s results table describes 0.5s stride ("windows
  overlap"). Not this dashboard's bug, but worth flagging to whoever owns
  those two files since it'll affect real replay speed/pacing once wired up.

## Analyze your own capture (upload a .log)

The **Analyze capture** button (top right) uploads a ROAD-format `.log`
file (SocketCAN text) from this computer to a local server that scores it with
the real Defender (`part1.windowing.make_windows` -> `defender.Defender`, v2),
streaming each window back live. The file never leaves the machine and is
deleted after scoring. Press **Replay in dashboard** to play it in every tab.
The Metrics tab's **Edge device** card shows live CPU, memory, GPU (context
only; the Defender is CPU-only), frames/s, latency and model size, refreshed
every 3 s. EdgeGuard has no language model, so it counts CAN frames, not tokens.

```bash
# terminal 1, repo root
source .venv/bin/activate
pip install psutil            # optional, for exact CPU / memory
python -m uvicorn integration.analyze_server:app --host 127.0.0.1 --port 8000

# terminal 2
cd dashboard && npm run dev   # Vite proxies /api to port 8000
```
