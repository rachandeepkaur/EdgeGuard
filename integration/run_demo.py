"""
Block 4 integration: one command that connects replay, attack testing,
scoring, decision and evaluation (build plan S9, Block 4; completion
check: "One command runs both ordinary and test paths").

    Ordinary path   Fleet replay (part1.pipeline.RoadData, via
                    defender.nano_runner.real_windows) -> preprocess ->
                    Defender.score_window -> SimulatedConsumer decision
                    (ALERT/ISOLATION or FORWARD). No labels: this is what
                    the pipeline looks like in ordinary use.

                    --escalate additionally wires build plan section 7's
                    cloud-escalation path onto these same windows: a band
                    half-width is calibrated on VALIDATION scores only
                    (part1.escalation_policy.choose_band_half_width, never
                    on the windows being scored), then each output within
                    that band around the threshold is sanitized
                    (part1.sanitizer) and sent to a simulated cloud second
                    opinion (part1.mock_cloud_endpoint) -- run that
                    service separately first:
                        uvicorn part1.mock_cloud_endpoint:app --port 9000
                    The local decision is never blocked on this call: a
                    failed or slow simulated-cloud call is recorded (
                    part1.egress_metrics) as a failure, not silently
                    dropped, and the ordinary path's own decisions and
                    actions are unaffected either way.

    Test path       Fleet replay of DEVELOPMENT captures only (Red Team
                    development-only rule, see part3.red_team_agent) ->
                    Red Team proposes an attack (part3.red_team_agent
                    .propose) -> injector applies it to a COPY
                    (part3.attack_injector.inject) -> preprocess ->
                    Defender.score_window on both the untouched window (a
                    negative control) and the attacked one -> the label is
                    known for certain here (we made the attack), so
                    part3.evaluator + part3.metrics report real detection
                    numbers: recall overall and by family, false-alarm
                    rate on the paired normal controls.

                    With --escalate, the test path also measures whether
                    the escalation band targets the local model's REAL
                    mistakes (part1.escalation_quality): escalation
                    precision and recall against these known labels,
                    error rate outside the band, and bytes that would
                    leave the device. No cloud call is needed for this --
                    it is the measurable half of the escalation decision.
                    The band is the same one the ordinary path uses,
                    calibrated on validation scores only.

Usage:
    python -m integration.run_demo --model-version v2 --data-dir ~/Downloads/road

Runs both paths by default (--mode both). --mode ordinary or --mode test
runs just one. Never touches final_test or separate captures -- neither
path is the final evaluation (build plan S9, Block 5 does that once).
"""

import argparse
import dataclasses
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from defender.defender import Defender
from defender.nano_runner import REAL_MODE_GROUPS, real_windows
from defender.simulated_consumer import SIMULATED_ALERT, SIMULATED_ISOLATION, SimulatedConsumer
from part1.egress_metrics import EgressReport, EscalationOutcome, compute_egress_report
from part1.escalation_policy import choose_band_half_width, should_escalate
from part1.escalation_quality import compute_escalation_quality
from part1.mock_cloud_endpoint import request_second_opinion
from part1.pipeline import RoadData, preprocess
from part1.sanitizer import sanitize
from part1.split_manifest import MANIFEST_PATH, load_manifest
from part3.attack_injector import inject
from part3.evaluator import EvaluationResult, evaluate_batch
from part3.evasion_log import append_evasion, make_evasion_entry
from part3.metrics import MetricsReport, compute_metrics
from part3.red_team_agent import propose
from shared.schemas import GroundTruthLabel, TrafficWindow

ATTACK_FAMILIES = ("freeze", "offset")
DEFAULT_CLOUD_URL = "http://127.0.0.1:9000"   # part1.mock_cloud_endpoint, run separately


def _p95(values: List[float]) -> float:
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=20, method="inclusive")[18]


class OrdinarySummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_version: str
    data_dir: str
    group: str
    captures: List[str]
    windows: int
    attack_decisions: int
    action_counts: Dict[str, int]
    latency_ms_mean: float
    latency_ms_median: float
    latency_ms_p95: float
    latency_ms_max: float
    escalation_band_half_width: Optional[float] = None
    egress: Optional[Dict[str, object]] = None


class TestSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_version: str
    data_dir: str
    captures: List[str]
    families_tested: List[str]
    attacks_proposed: Dict[str, int]
    attacks_skipped: Dict[str, int]
    windows_evaluated: int
    recall_overall: Optional[float]
    recall_by_family: Dict[str, Optional[float]]
    false_alarm_rate: Optional[float]
    precision: Optional[float]
    f1: Optional[float]
    mean_inference_ms: Optional[float]
    escalation_quality: Optional[Dict[str, object]] = None


def _save(path, data: dict) -> None:
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------
# Ordinary path
# ---------------------------------------------------------------------

def calibrate_escalation_band(defender: Defender, data_dir, manifest_path,
                              max_escalation_rate: float) -> float:
    """Band half-width for should_escalate(), calibrated on VALIDATION
    scores only -- never on the windows the ordinary path is about to
    score for the demo, and never on development or final_test (mirrors
    threshold.py's own choose_threshold contract)."""
    manifest = load_manifest(manifest_path)
    road = RoadData(data_dir, manifest=manifest)
    scores = [
        defender.score_window(preprocess(window)).attack_score
        for name in manifest.names("validation")
        for window in road.windows(name)
    ]
    return choose_band_half_width(scores, defender.threshold, max_escalation_rate)


def run_ordinary(defender: Defender, data_dir, manifest_path, group: str,
                 captures: Optional[List[str]], max_windows: Optional[int],
                 attack_action: str, *,
                 escalate: bool = False,
                 max_escalation_rate: Optional[float] = None,
                 cloud_url: str = DEFAULT_CLOUD_URL,
                 cloud_timeout_s: float = 2.0) -> OrdinarySummary:
    windows, capture_names = real_windows(data_dir, manifest_path, group,
                                          captures=captures, max_windows=max_windows)
    consumer = SimulatedConsumer(attack_action=attack_action)
    latencies: List[float] = []
    attack_decisions = 0

    band_half_width = None
    outcomes: List[EscalationOutcome] = []
    if escalate:
        if max_escalation_rate is None:
            raise ValueError("escalate=True requires max_escalation_rate")
        band_half_width = calibrate_escalation_band(
            defender, data_dir, manifest_path, max_escalation_rate)

    for window in windows:
        output = defender.score_window(preprocess(window))
        consumer.handle(output)
        latencies.append(output.latency_ms)
        if output.decision == "ATTACK":
            attack_decisions += 1

        if escalate:
            if should_escalate(output, band_half_width).escalate:
                advice, added_ms = request_second_opinion(
                    sanitize(output), base_url=cloud_url, timeout_s=cloud_timeout_s)
                outcomes.append(EscalationOutcome(
                    window_id=output.window_id, escalated=True,
                    added_latency_ms=added_ms,
                    corrected=(advice.advice != "confirm") if advice is not None else None,
                ))
            else:
                outcomes.append(EscalationOutcome(window_id=output.window_id, escalated=False))

    egress: Optional[EgressReport] = compute_egress_report(outcomes) if escalate else None

    return OrdinarySummary(
        model_version=defender.model_version,
        data_dir=str(data_dir),
        group=group,
        captures=capture_names,
        windows=len(windows),
        attack_decisions=attack_decisions,
        action_counts=consumer.summary(),
        latency_ms_mean=statistics.fmean(latencies),
        latency_ms_median=statistics.median(latencies),
        latency_ms_p95=_p95(latencies),
        latency_ms_max=max(latencies),
        escalation_band_half_width=band_half_width,
        egress=dataclasses.asdict(egress) if egress is not None else None,
    )


# ---------------------------------------------------------------------
# Test path
# ---------------------------------------------------------------------

def _development_captures(manifest_path, captures: Optional[List[str]]) -> List[str]:
    manifest = load_manifest(manifest_path)
    if captures is None:
        names = manifest.names("development")
    else:
        names = captures
        for name in names:
            group = manifest.entry(name).group
            if group != "development":
                raise ValueError(
                    f"{name} is a {group!r} capture; Red Team attacks may only target "
                    "development captures (see part3.red_team_agent)"
                )
    if not names:
        raise ValueError("no development captures found")
    return names


def run_test(defender: Defender, data_dir, manifest_path,
            captures: Optional[List[str]], max_windows: Optional[int],
            families: List[str], evasion_log_path: Optional[str], *,
            escalation_band_half_width: Optional[float] = None) -> TestSummary:
    names = _development_captures(manifest_path, captures)
    manifest = load_manifest(manifest_path)
    road = RoadData(data_dir, manifest=manifest)

    outputs = []
    labels: List[GroundTruthLabel] = []
    specs_by_window_id = {}     # window_id -> AttackSpec, for the evasion log only
    proposed = {family: 0 for family in families}
    skipped = {family: 0 for family in families}
    seen_windows = 0

    def raw_windows():
        for name in names:
            for window in road.windows(name):
                yield window

    for window in raw_windows():
        if max_windows is not None and seen_windows >= max_windows:
            break
        seen_windows += 1

        normal_out = defender.score_window(preprocess(window))
        outputs.append(normal_out)
        labels.append(GroundTruthLabel(window_id=window.window_id, is_attack=False))

        for variant_index, family in enumerate(families, start=1):
            try:
                spec = propose(window, family)
                attacked: TrafficWindow = inject(window, spec, variant_index=variant_index)
            except ValueError:
                skipped[family] += 1
                continue
            proposed[family] += 1

            attack_out = defender.score_window(preprocess(attacked))
            outputs.append(attack_out)
            labels.append(GroundTruthLabel(
                window_id=attacked.window_id, is_attack=True,
                family=spec.family, target=spec.target_can_id,
            ))
            specs_by_window_id[attacked.window_id] = spec

    if not outputs:
        raise ValueError("no windows were scored; nothing to evaluate")

    results: List[EvaluationResult] = evaluate_batch(outputs, labels)
    overall: MetricsReport = compute_metrics(results)

    recall_by_family: Dict[str, Optional[float]] = {}
    for family in families:
        family_results = [r for r in results if r.family == family]
        family_metrics = compute_metrics(family_results)
        recall_by_family[family] = family_metrics.recall

    if evasion_log_path is not None:
        for result in results:
            if result.outcome == "FN":
                # Use the REAL spec this attack was made from, never a
                # reconstruction -- the evasion log must record what
                # actually happened, for honest later hardening.
                spec = specs_by_window_id[result.window_id]
                append_evasion(evasion_log_path, make_evasion_entry(
                    spec=spec, result=result, split="development",
                ))

    quality = (
        compute_escalation_quality(outputs, labels, escalation_band_half_width)
        if escalation_band_half_width is not None else None
    )

    return TestSummary(
        model_version=defender.model_version,
        data_dir=str(data_dir),
        captures=names,
        families_tested=list(families),
        attacks_proposed=proposed,
        attacks_skipped=skipped,
        windows_evaluated=len(results),
        recall_overall=overall.recall,
        recall_by_family=recall_by_family,
        false_alarm_rate=overall.false_alarm_rate,
        precision=overall.precision,
        f1=overall.f1,
        mean_inference_ms=overall.mean_inference_ms,
        escalation_quality=dataclasses.asdict(quality) if quality is not None else None,
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Block 4 integration: one command, ordinary and/or test paths."
    )
    parser.add_argument("--model-version", required=True,
                        help="trained Defender version to run, e.g. v1 or v2")
    parser.add_argument("--data-dir", default="data/road",
                        help="ROAD folder with ambient/ attacks/ (default data/road)")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH),
                        help="frozen split manifest")
    parser.add_argument("--model-dir", default="models",
                        help="where trained models are saved (default models)")
    parser.add_argument("--mode", default="both", choices=("ordinary", "test", "both"),
                        help="which path(s) to run (default both)")

    parser.add_argument("--ordinary-group", default="validation", choices=REAL_MODE_GROUPS,
                        help="split group for the ordinary path (default validation)")
    parser.add_argument("--ordinary-captures",
                        help="comma-separated capture names, instead of --ordinary-group")
    parser.add_argument("--ordinary-max-windows", type=int,
                        help="cap the number of ordinary-path windows (default: all)")
    parser.add_argument("--attack-action", default=SIMULATED_ALERT,
                        choices=(SIMULATED_ALERT, SIMULATED_ISOLATION),
                        help="simulated action for an ATTACK decision (default "
                             f"{SIMULATED_ALERT})")

    parser.add_argument("--test-captures",
                        help="comma-separated development capture names, instead of all "
                             "development captures")
    parser.add_argument("--test-max-windows", type=int,
                        help="cap the number of test-path base windows (default: all)")
    parser.add_argument("--test-families", default=",".join(ATTACK_FAMILIES),
                        help=f"comma-separated attack families to test (default all: "
                             f"{','.join(ATTACK_FAMILIES)})")
    parser.add_argument("--evasion-log",
                        help="optional path to append confirmed development-set misses "
                             "(part3.evasion_log), for later Defender hardening")

    parser.add_argument("--escalate", action="store_true",
                        help="also run the cloud-escalation path (build plan section 7): "
                             "on ordinary-path windows it calls the simulated cloud (run "
                             "it first, see the module docstring); on test-path windows it "
                             "measures escalation precision/recall against known labels "
                             "(no cloud needed). Requires --max-escalation-rate")
    parser.add_argument("--max-escalation-rate", type=float, default=None,
                        help="team decision, e.g. 0.1 (no default) -- escalation budget, "
                             "calibrated on VALIDATION scores; only used with --escalate")
    parser.add_argument("--cloud-url", default=DEFAULT_CLOUD_URL,
                        help=f"simulated cloud base URL (default {DEFAULT_CLOUD_URL}; "
                             "see part1.mock_cloud_endpoint)")
    parser.add_argument("--cloud-timeout-s", type=float, default=2.0,
                        help="per-call timeout for the simulated cloud second opinion "
                             "(default 2.0)")

    parser.add_argument("--output-dir", default="results",
                        help="where to save demo_ordinary.json / demo_test.json "
                             "(default results)")
    args = parser.parse_args(argv)

    if args.ordinary_max_windows is not None and args.ordinary_max_windows < 1:
        parser.error("--ordinary-max-windows must be at least 1")
    if args.test_max_windows is not None and args.test_max_windows < 1:
        parser.error("--test-max-windows must be at least 1")
    if args.escalate and args.max_escalation_rate is None:
        parser.error("--escalate requires --max-escalation-rate")
    families = [f.strip() for f in args.test_families.split(",") if f.strip()]
    unknown = [f for f in families if f not in ATTACK_FAMILIES]
    if unknown:
        parser.error(f"unsupported --test-families {unknown}; choose from {ATTACK_FAMILIES}")

    defender = Defender.load(args.model_dir, args.model_version)
    band_half_width: Optional[float] = None

    if args.mode in ("ordinary", "both"):
        ordinary_captures = (
            [c.strip() for c in args.ordinary_captures.split(",")]
            if args.ordinary_captures else None
        )
        ordinary = run_ordinary(
            defender, args.data_dir, args.manifest, args.ordinary_group,
            ordinary_captures, args.ordinary_max_windows, args.attack_action,
            escalate=args.escalate, max_escalation_rate=args.max_escalation_rate,
            cloud_url=args.cloud_url, cloud_timeout_s=args.cloud_timeout_s,
        )
        _save(Path(args.output_dir) / "demo_ordinary.json", ordinary.model_dump())
        band_half_width = ordinary.escalation_band_half_width
        print("=== Ordinary path ===")
        print(f"Model version:  {ordinary.model_version}")
        print(f"Captures:       {', '.join(ordinary.captures)}")
        print(f"Windows scored: {ordinary.windows}  "
              f"(ATTACK decisions: {ordinary.attack_decisions})")
        print(f"Actions:        {ordinary.action_counts}")
        print(f"Latency ms:     mean {ordinary.latency_ms_mean:.3f} | "
              f"median {ordinary.latency_ms_median:.3f} | "
              f"p95 {ordinary.latency_ms_p95:.3f} | max {ordinary.latency_ms_max:.3f}")
        print(f"Saved to:       {Path(args.output_dir) / 'demo_ordinary.json'}")

        if ordinary.egress is not None:
            e = ordinary.egress
            print(f"\n=== Cloud escalation path (simulated -- {args.cloud_url}) ===")
            print(f"Band:           threshold +/- {ordinary.escalation_band_half_width:.4f} "
                  f"(calibrated on validation scores, budget {args.max_escalation_rate})")
            print(f"Escalated:      {e['escalations']} / {e['windows']} "
                  f"({e['escalation_rate']:.4f})")
            print(f"Failed calls:   {e['failed_calls']}")
            print(f"Added latency:  mean {e['mean_added_latency_ms']} ms | "
                  f"p95 {e['p95_added_latency_ms']} ms")
            print(f"Stand-in cloud disagreed: {e['corrections']} (rate {e['correction_rate']}) "
                  f"-- NOT a quality metric: the stand-in's fixed threshold is not "
                  f"calibrated to this model. See 'Escalation quality' on the test path.")

    if args.mode in ("test", "both"):
        test_captures = (
            [c.strip() for c in args.test_captures.split(",")]
            if args.test_captures else None
        )
        if args.escalate and band_half_width is None:
            band_half_width = calibrate_escalation_band(
                defender, args.data_dir, args.manifest, args.max_escalation_rate)
        test = run_test(
            defender, args.data_dir, args.manifest, test_captures,
            args.test_max_windows, families, args.evasion_log,
            escalation_band_half_width=band_half_width if args.escalate else None,
        )
        _save(Path(args.output_dir) / "demo_test.json", test.model_dump())
        print("\n=== Test path (Red Team attacks on development captures) ===")
        print(f"Model version:      {test.model_version}")
        print(f"Captures:           {', '.join(test.captures)}")
        print(f"Families tested:    {test.families_tested}")
        print(f"Attacks proposed:   {test.attacks_proposed}")
        print(f"Attacks skipped:    {test.attacks_skipped}  (no valid attack on that window)")
        print(f"Windows evaluated:  {test.windows_evaluated}")
        print(f"Recall overall:     {test.recall_overall}")
        print(f"Recall by family:   {test.recall_by_family}")
        print(f"False alarm rate:   {test.false_alarm_rate}  (on paired normal controls)")
        print(f"Saved to:           {Path(args.output_dir) / 'demo_test.json'}")

        if test.escalation_quality is not None:
            q = test.escalation_quality

            def pct(value):
                return "n/a" if value is None else f"{value:.1%}"

            print("\n=== Escalation quality (test path, known labels, no cloud needed) ===")
            print(f"Band:                   threshold +/- {q['band_half_width']:.4f} "
                  f"(calibrated on validation scores)")
            print(f"Escalated:              {q['escalations']} / {q['windows']} "
                  f"({pct(q['escalation_rate'])})")
            print(f"Local mistakes:         {q['local_errors']} "
                  f"({q['local_false_positives']} false alarms, "
                  f"{q['local_false_negatives']} misses)")
            print(f"Escalation precision:   {pct(q['escalation_precision'])}  "
                  f"(escalated windows the local model actually got wrong)")
            print(f"Escalation recall:      {pct(q['escalation_recall'])}  "
                  f"(local mistakes that were escalated: "
                  f"{q['escalated_false_positives']} false alarms, "
                  f"{q['escalated_false_negatives']} misses)")
            print(f"Error rate outside band: {pct(q['error_rate_outside_band'])}")
            print(f"Accuracy:               local {pct(q['local_accuracy'])} | "
                  f"ceiling with a perfect second opinion "
                  f"{pct(q['local_plus_perfect_cloud_accuracy'])}")
            mean_bytes = q['bundle_bytes_mean']
            print(f"Egress per escalation:  "
                  f"{'n/a' if mean_bytes is None else f'{mean_bytes:.0f}'} bytes mean, "
                  f"max {q['bundle_bytes_max']} | raw CAN bytes: 0 "
                  f"(bundle fields: {', '.join(q['bundle_fields'])})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
