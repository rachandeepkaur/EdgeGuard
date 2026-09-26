## Part 3: Red Team and Evaluation

Part 3 creates test attacks and measures how well the Defender detects them. It does not train or run the Defender itself.

### How it works

1. Receive a raw `TrafficWindow` from Part 1.
2. Propose an attack against a CAN ID in that window (`red_team_agent.propose`, or the two concrete functions below).
3. Validate the proposal, then inject it into a **copy** of the raw window (`attack_injector.inject`).
4. Pass the modified window through Part 1's preprocessing before sending it to the Defender.
5. Compare the Defender's output with hidden labels to calculate detection metrics.
6. Record confirmed misses from the development split for later Defender retraining.
7. Compare baseline and updated Defender results on the held-out final test split.

### Attack families (build plan S5)

Two families are implemented, matching the plan's "start with two repeatable attack families":

- **freeze** (`propose_freeze` / `inject_freeze`) -- picks a CAN ID whose payload changes in the window, then repeats its first payload for the rest of the interval. Tests whether the Defender notices a value that stopped changing.
- **offset** (`propose_offset` / `inject_offset`) -- picks the most STABLE payload byte in the window, then adds a bounded signed delta to it (clamped back into 0-255) for the interval, pushing it toward whichever extreme has more headroom. This is the ROAD-style "spoof a sensor reading" attack (e.g. `max_speedometer_attack` sets a value near its maximum) rather than freeze's "stop a changing value". It targets a single payload byte because that is exactly what `defender/stage2.py` learns per-CAN-ID ranges over ("byte fields"), so it is a meaningful test of Stage 2's `out_of_range` check.

Both go through `attack_spec.AttackSpec` (`family: "freeze" | "offset"`); `offset` additionally carries `byte_index` (0-7) and `offset` (a signed delta, -128..127, clamped after addition). `red_team_agent.propose(window, family)` and `attack_injector.inject(window, spec)` dispatch to the concrete function for you.

Other attack families and an adaptive or LLM-based attacker are future work (build plan S10: defer until the core demo is complete).

### Files

| File | Purpose |
| --- | --- |
| `attack_spec.py` | Defines a freeze or offset attack proposal. |
| `red_team_agent.py` | Proposes a freeze or offset attack from a raw window. |
| `attack_validator.py` | Checks whether the proposal can modify the window. |
| `attack_injector.py` | Applies the attack to a copy of the window. |
| `evaluator.py` | Joins Defender outputs with hidden labels by window ID. |
| `metrics.py` | Calculates detection counts and rates. |
| `evasion_log.py` | Records confirmed development-set misses. |
| `hardening_set.py` | Prepares a list of misses for retraining. |
| `evidence_gate.py` | Compares Defender versions on held-out test results. |

### Tests and current status

From the repository root, run:

```bash
python -m pytest tests/test_{attack_injector,attack_validator,evaluator,evasion_log,evidence_gate,hardening_set,metrics,red_team_agent}.py -q
```

Ported onto current `main` (this branch was originally built against the pre-`part1.pipeline`/`RoadData` code and had drifted significantly; `evaluator.py`, `metrics.py`, `evasion_log.py`, `hardening_set.py` and `evidence_gate.py` needed no changes since they only depend on `shared/schemas.py`'s `DefenderOutput`/`GroundTruthLabel`, which are unchanged in the fields they use). The freeze family and its tests are unchanged from the original branch; the offset family, its injector/validator/proposer logic, and its tests are new.

Not yet done: wiring this into Block 4 integration (one command running replay -> Red Team -> injector -> Defender -> evaluator), and the "second attack family" gap this closes was the last missing piece the build plan called out for Block 3.
