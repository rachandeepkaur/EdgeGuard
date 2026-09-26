"""
Run the EdgeGuard Defender locally and measure inference latency (Part 2).

Purpose (HP requirement): show that primary detection runs ON THE NANO,
with no cloud connection, and report how fast it is.

What it records for every window:
    window_id, decision, attack_score, latency_ms
And a summary:
    number of windows, mean / median / p95 / max latency_ms,
    windows per second (inference only), number of ATTACK decisions,
    machine name and CPU architecture, model_version, and whether the
    traffic was MOCK.

latency_ms is Defender inference only (see defender.py). Window
collection time is NOT included and must be measured separately.

Two modes:

    --mock
        python -m defender.nano_runner --mock --output results/nano_benchmark_MOCK.json

        Trains a small MOCK model in memory on MOCK traffic. Its numbers show
        that the pipeline runs and how fast it is on this machine; they say
        NOTHING about detection quality on ROAD.

    --model-version (real mode)
        python -m defender.nano_runner --model-version v1 \\
            --data-dir ~/Downloads/road --output results/nano_benchmark_REAL_v1.json

        Loads a trained Defender (defender.py's Defender.load()) and benchmarks
        it on real ROAD windows read through part1.pipeline.RoadData. Windows
        come from --group (default "validation") or an explicit --captures
        list; --group only ever accepts train / validation / development --
        final_test and separate are reserved for the final evaluation and are
        not reachable from here (see CLAUDE.md's "Rules for any code touching
        data"). This is a SPEED benchmark, not an evaluation: it reports
        latency and throughput on real traffic, not detection accuracy.
"""

import argparse
import itertools
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict

from defender.defender import Defender
from defender.simulated_consumer import SimulatedConsumer
from defender.stage1 import Stage1Model
from defender.threshold import choose_threshold
from part1.pipeline import RoadData
from part1.split_manifest import MANIFEST_PATH, load_manifest
from shared.schemas import TrafficWindow

REAL_MODE_GROUPS = ("train", "validation", "development")


class WindowRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    window_id: str
    decision: str
    attack_score: float
    latency_ms: float


class BenchmarkSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mock_traffic: bool
    model_version: str
    machine: str
    architecture: str
    python_version: str
    windows: int
    attack_decisions: int
    latency_ms_mean: float
    latency_ms_median: float
    latency_ms_p95: float
    latency_ms_max: float
    windows_per_second: float
    wall_clock_seconds: float
    note: str


def _p95(values: List[float]) -> float:
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=20, method="inclusive")[18]


def run_benchmark(defender, windows: Iterable[TrafficWindow],
                  mock_traffic: bool,
                  consumer: Optional[SimulatedConsumer] = None):
    """Score every window locally. Returns (records, summary).

    defender only needs .score_window(window) -> DefenderOutput and a
    .model_version attribute -- the same duck-typed interface
    defender.defender.Defender, defender.timing_cnn.TimingCNNDefender,
    defender.fused_defender.FusedV2TimingCNNDefender and
    defender.defender_llm.DefenderLLMDefender all already share (it is what
    integration/run_final_evaluation.py relies on too) -- so this benchmarks
    whichever one you are actually about to ship, not only a plain v1/v2
    Defender."""
    if not (hasattr(defender, "score_window") and hasattr(defender, "model_version")):
        raise TypeError("run_benchmark() needs an object with score_window() and "
                        "model_version -- e.g. a Defender, TimingCNNDefender, or "
                        "FusedV2TimingCNNDefender")
    consumer = consumer or SimulatedConsumer()

    records: List[WindowRecord] = []
    started = time.perf_counter()
    for window in windows:
        output = defender.score_window(window)
        consumer.handle(output)
        records.append(WindowRecord(window_id=output.window_id, decision=output.decision,
                                    attack_score=output.attack_score,
                                    latency_ms=output.latency_ms))
    wall_clock = time.perf_counter() - started
    if not records:
        raise ValueError("No windows were given, nothing to benchmark")

    latencies = [r.latency_ms for r in records]
    total_inference_s = sum(latencies) / 1000.0
    summary = BenchmarkSummary(
        mock_traffic=mock_traffic,
        model_version=defender.model_version,
        machine=platform.node(),
        architecture=platform.machine(),
        python_version=platform.python_version(),
        windows=len(records),
        attack_decisions=sum(1 for r in records if r.decision == "ATTACK"),
        latency_ms_mean=statistics.fmean(latencies),
        latency_ms_median=statistics.median(latencies),
        latency_ms_p95=_p95(latencies),
        latency_ms_max=max(latencies),
        windows_per_second=len(records) / total_inference_s if total_inference_s > 0 else 0.0,
        wall_clock_seconds=wall_clock,
        note=(
            "MOCK traffic: shows the pipeline runs locally and its speed; "
            "NOT a detection result on ROAD." if mock_traffic else
            "Real windows. Inference latency only; window collection time excluded."
        ),
    )
    return records, summary


def save_results(path, records, summary) -> None:
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps({"summary": summary.model_dump(),
                                "windows": [r.model_dump() for r in records]}, indent=2),
                    encoding="utf-8")


def build_mock_defender() -> Defender:
    """MOCK model: trained in memory on MOCK traffic only."""
    from defender.mock_traffic import make_mock_normal_window
    stage1 = Stage1Model().fit([make_mock_normal_window(i, "cap90") for i in range(40)])
    validation = [stage1.score(make_mock_normal_window(i, "cap92")).score
                  for i in range(100, 130)]
    threshold = choose_threshold(validation, 0.0)
    return Defender(stage1, threshold, "MOCK_v1")


def mock_windows(count: int) -> List[TrafficWindow]:
    """MOCK mix: every 10th window is a MOCK fuzzing attack."""
    from defender.mock_traffic import make_mock_fuzzing_window, make_mock_normal_window
    return [make_mock_fuzzing_window(i) if i % 10 == 9 else make_mock_normal_window(i, "cap93")
            for i in range(count)]


def real_windows(data_dir, manifest_path, group: str,
                 captures: Optional[List[str]] = None,
                 max_windows: Optional[int] = None) -> Tuple[List[TrafficWindow], List[str]]:
    """Real ROAD windows for the benchmark, via part1.pipeline.RoadData.

    captures, if given, is used verbatim instead of --group (still read
    through RoadData, so final_test / separate captures are still refused
    unless RoadData itself is given final_evaluation=True, which this
    benchmark never does). Returns (windows, capture_names_used)."""
    manifest = load_manifest(manifest_path)
    names = captures if captures else manifest.names(group)
    if not names:
        raise ValueError(f"No captures found for group {group!r} under manifest {manifest_path}")
    road = RoadData(data_dir, manifest=manifest)
    chained = itertools.chain.from_iterable(road.windows(name) for name in names)
    if max_windows is not None:
        chained = itertools.islice(chained, max_windows)
    windows = list(chained)
    if not windows:
        raise ValueError(f"Captures {names} under {data_dir} produced no windows")
    return windows, names


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the Defender locally and measure latency.")
    parser.add_argument("--mock", action="store_true",
                        help="use MOCK model and MOCK traffic")
    parser.add_argument("--windows", type=int, default=200,
                        help="number of MOCK windows to score (default 200; --mock only)")
    parser.add_argument("--model-version",
                        help="real mode: trained model version to benchmark, e.g. v1 or v2")
    parser.add_argument("--cnn-model-dir",
                        help="real mode: if given, benchmark FusedV2TimingCNNDefender instead "
                             "of a plain Defender -- loads v2 from --model-dir plus the "
                             "TimingCNN checkpoint from this directory "
                             "(defender/train_timing_cnn.py's output)")
    parser.add_argument("--data-dir", default="data/road",
                        help="ROAD folder with ambient/ attacks/ (default data/road; real mode "
                             "only)")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH),
                        help="frozen split manifest (real mode only)")
    parser.add_argument("--model-dir", default="models",
                        help="where trained models are saved (default models; real mode only)")
    parser.add_argument("--group", default="validation", choices=REAL_MODE_GROUPS,
                        help="split group to pull real windows from (default validation; real "
                             "mode only -- never final_test/separate, see CLAUDE.md)")
    parser.add_argument("--captures",
                        help="comma-separated capture names, instead of --group (real mode "
                             "only)")
    parser.add_argument("--max-windows", type=int,
                        help="cap the number of real windows scored (default: all of them; "
                             "real mode only)")
    parser.add_argument("--output", required=True, help="where to save the JSON results")
    args = parser.parse_args(argv)

    if args.mock and (args.model_version or args.cnn_model_dir):
        parser.error("--mock and --model-version/--cnn-model-dir are mutually exclusive; "
                     "choose one mode")
    if not args.mock and args.model_version is None and args.cnn_model_dir is None:
        parser.error("Choose a mode: --mock, --model-version (e.g. v1 or v2), or "
                     "--cnn-model-dir for the fused detector")

    if args.mock:
        if args.windows < 1:
            parser.error("--windows must be at least 1")
        records, summary = run_benchmark(build_mock_defender(), mock_windows(args.windows),
                                         mock_traffic=True)
        save_results(args.output, records, summary)

        print("=== EdgeGuard Defender local run (MOCK traffic) ===")
        print(f"Machine:        {summary.machine} ({summary.architecture})")
        print(f"Model version:  {summary.model_version}")
        print(f"Windows scored: {summary.windows}  (ATTACK decisions: {summary.attack_decisions})")
        print(f"Latency ms:     mean {summary.latency_ms_mean:.3f} | median "
              f"{summary.latency_ms_median:.3f} | p95 {summary.latency_ms_p95:.3f} | "
              f"max {summary.latency_ms_max:.3f}")
        print(f"Throughput:     {summary.windows_per_second:.0f} windows/second (inference only)")
        print(f"Saved to:       {args.output}")
        print(summary.note)
        return 0

    if args.max_windows is not None and args.max_windows < 1:
        parser.error("--max-windows must be at least 1")
    captures = [c.strip() for c in args.captures.split(",")] if args.captures else None

    if args.cnn_model_dir:
        from defender.fused_defender import FusedV2TimingCNNDefender
        from defender.timing_cnn import TimingCNNModel
        v2 = Defender.load(args.model_dir, "v2")
        cnn = TimingCNNModel.load(args.cnn_model_dir)
        defender = FusedV2TimingCNNDefender(v2, cnn)
    else:
        defender = Defender.load(args.model_dir, args.model_version)
    windows, capture_names = real_windows(args.data_dir, args.manifest, args.group,
                                          captures=captures, max_windows=args.max_windows)
    records, summary = run_benchmark(defender, windows, mock_traffic=False)
    save_results(args.output, records, summary)

    print("=== EdgeGuard Defender local run (REAL ROAD traffic) ===")
    print(f"Machine:        {summary.machine} ({summary.architecture})")
    print(f"Model version:  {summary.model_version}")
    print(f"Data dir:       {args.data_dir}")
    print(f"Captures:       {', '.join(capture_names)}")
    print(f"Windows scored: {summary.windows}  (ATTACK decisions: {summary.attack_decisions})")
    print(f"Latency ms:     mean {summary.latency_ms_mean:.3f} | median "
          f"{summary.latency_ms_median:.3f} | p95 {summary.latency_ms_p95:.3f} | "
          f"max {summary.latency_ms_max:.3f}")
    print(f"Throughput:     {summary.windows_per_second:.0f} windows/second (inference only)")
    print(f"Saved to:       {args.output}")
    print(summary.note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
