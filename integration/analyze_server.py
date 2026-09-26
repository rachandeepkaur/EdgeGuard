"""
Local analysis server for the dashboard: upload a ROAD-format CAN capture
(.log, SocketCAN text format) and score it with the REAL EdgeGuard Defender,
streaming each window's DefenderOutput back as it is produced.

    # from the EdgeGuard repo root, .venv active
    python -m uvicorn integration.analyze_server:app --host 127.0.0.1 --port 8000

    # then, in dashboard/:  npm run dev   (Vite proxies /api to port 8000)

This is the SAME path as the ordinary demo run, nothing re-implemented:
    part1.windowing.make_windows   read -> clean -> 1 s windows
    defender.defender.Defender     Stage 1 + Stage 2, fused (model v2 by default)
The uploaded file stays on this machine (a temp file, deleted after
scoring) and is never sent anywhere else. No ground-truth labels exist for
an arbitrary upload, so the stream carries only what the Defender itself
outputs -- the same rule dashboard/scripts/build_real_run.py follows.

Endpoints
    GET  /api/health    model version, threshold, window length, model size
    POST /api/analyze   raw file body (?name=<file>.log) -> NDJSON stream:
                        {"type":"meta"...}, {"type":"window"...} per window,
                        {"type":"progress"...}, then {"type":"done"...}
                        or {"type":"error"...}
    GET  /api/system    device telemetry for the Metrics tab (CPU, memory,
                        GPU if nvidia-smi exists) + cumulative scoring counters

EdgeGuard has no language model, so there are no "tokens": the equivalent
unit of work here is a CAN frame (and a 1 s window). The telemetry reports
frames/s and windows/s instead of tokens/s.
"""

import json
import os
import platform
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from defender.defender import Defender
from part1.cleaning import CleaningReport
from part1.fleet_simulator import capture_span_us
from part1.windowing import make_windows

try:                        # optional: better CPU/memory numbers when installed
    import psutil
except ImportError:         # pragma: no cover - depends on the environment
    psutil = None

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.environ.get("EDGEGUARD_MODEL_DIR", REPO_ROOT / "models"))
MODEL_VERSION = os.environ.get("EDGEGUARD_MODEL_VERSION", "v2")
REAL_RUN = REPO_ROOT / "dashboard" / "src" / "data" / "realRun.json"
UPLOAD_CAPTURE_ID = "up01"          # neutral id, never the uploaded file name
MAX_UPLOAD_BYTES = 1_500_000_000    # ROAD's largest captures are ~400 MB
PROGRESS_EVERY = 10                 # windows between progress events

app = FastAPI(title="EdgeGuard local analysis server")

_defender: Optional[Defender] = None
_lock = threading.Lock()
_started = time.time()

# Cumulative counters since the server started, for /api/system.
_totals = {
    "bytes_ingested": 0,
    "frames_processed": 0,
    "windows_scored": 0,
    "attacks_flagged": 0,
    "captures_analyzed": 0,
    "scoring_ms": 0.0,
}
_last_run: dict = {}
_active = {"running": False, "name": None, "windows": 0, "frames": 0, "started": None}


def defender() -> Defender:
    global _defender
    if _defender is None:
        _defender = Defender.load(MODEL_DIR, MODEL_VERSION)
    return _defender


def escalation_band() -> Optional[float]:
    """The calibrated band half-width saved with the real replay (computed on
    validation scores by build_real_run.py). Reused, never re-fitted here."""
    try:
        meta = json.loads(REAL_RUN.read_text(encoding="utf-8"))["meta"]
        if meta.get("modelVersion") == MODEL_VERSION:
            return meta.get("escalationBandHalfWidth")
    except (OSError, KeyError, ValueError):
        pass
    return None


def model_bytes() -> int:
    return sum(p.stat().st_size for p in MODEL_DIR.glob(f"*_{MODEL_VERSION}.json"))


@app.get("/api/health")
def health():
    d = defender()
    return {
        "ok": True,
        "model_version": d.model_version,
        "threshold": d.threshold,
        "window_s": d.window_s or 1.0,
        "model_bytes": model_bytes(),
        "escalation_band_half_width": escalation_band(),
    }


@app.post("/api/analyze")
async def analyze(request: Request, name: str = "capture.log"):
    if not name.lower().endswith(".log"):
        raise HTTPException(400, "Upload a ROAD-format .log capture (SocketCAN text).")
    if _active["running"]:
        raise HTTPException(409, "Another capture is being analyzed; try again when it finishes.")

    # Save the body to a local temp file (streamed; never held in memory whole).
    fd, tmp_path = tempfile.mkstemp(prefix="edgeguard_upload_", suffix=".log")
    size = 0
    upload_started = time.perf_counter()
    try:
        with os.fdopen(fd, "wb") as fh:
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "File is larger than 1.5 GB.")
                fh.write(chunk)
    except BaseException:
        os.unlink(tmp_path)
        raise
    upload_s = time.perf_counter() - upload_started
    if size == 0:
        os.unlink(tmp_path)
        raise HTTPException(400, "The uploaded file is empty.")

    return StreamingResponse(_score_stream(tmp_path, name, size, upload_s),
                             media_type="application/x-ndjson")


def _line(obj) -> bytes:
    return (json.dumps(obj) + "\n").encode("utf-8")


def _score_stream(path: str, name: str, size: int, upload_s: float):
    """Sync generator (Starlette runs it in a worker thread)."""
    d = defender()
    window_s = d.window_s or 1.0
    report = CleaningReport()
    latencies = []
    windows = attacks = frames = 0
    started = time.perf_counter()
    with _lock:
        _active.update(running=True, name=name, windows=0, frames=0, started=time.time())
    try:
        try:
            total = max(int(capture_span_us(path) // int(window_s * 1_000_000)), 0)
        except ValueError as exc:            # no parseable CAN frames at all
            yield _line({"type": "error", "message": str(exc).replace(path, name)})
            return

        yield _line({
            "type": "meta",
            "file_name": name,
            "bytes": size,
            "upload_s": round(upload_s, 3),
            "capture_id": UPLOAD_CAPTURE_ID,
            "total_windows": total,
            "model_version": d.model_version,
            "threshold": d.threshold,
            "window_s": window_s,
            "escalation_band_half_width": escalation_band(),
        })

        for window in make_windows(path, UPLOAD_CAPTURE_ID, window_s=window_s,
                                   stride_s=window_s, report=report):
            out = d.score_window(window)
            windows += 1
            frames += len(window.frames)
            attacks += out.decision == "ATTACK"
            latencies.append(out.latency_ms)
            yield _line({
                "type": "window",
                "window_id": out.window_id,
                "attack_score": round(out.attack_score, 4),
                "threshold": out.threshold,
                "decision": out.decision,
                "evidence": out.evidence,
                "model_version": out.model_version,
                "latency_ms": round(out.latency_ms, 3),
                "frames": len(window.frames),
            })
            with _lock:
                _active.update(windows=windows, frames=frames)
            if windows % PROGRESS_EVERY == 0:
                elapsed = time.perf_counter() - started
                yield _line({"type": "progress", "windows": windows, "total_windows": total,
                             "frames": frames, "elapsed_s": round(elapsed, 3)})

        elapsed = time.perf_counter() - started
        latencies.sort()
        p95 = latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))] if latencies else 0.0
        summary = {
            "type": "done",
            "file_name": name,
            "bytes": size,
            "windows": windows,
            "attacks": attacks,
            "frames_read": report.frames_read,
            "frames_kept": report.frames_kept,
            "duplicates_dropped": report.duplicates_dropped,
            "skipped_lines": report.skipped_lines,
            "elapsed_s": round(elapsed, 3),
            "windows_per_s": round(windows / elapsed, 1) if elapsed > 0 else None,
            "frames_per_s": round(report.frames_read / elapsed) if elapsed > 0 else None,
            "mean_latency_ms": round(sum(latencies) / len(latencies), 3) if latencies else None,
            "p95_latency_ms": round(p95, 3),
        }
        with _lock:
            _totals["bytes_ingested"] += size
            _totals["frames_processed"] += report.frames_read
            _totals["windows_scored"] += windows
            _totals["attacks_flagged"] += attacks
            _totals["captures_analyzed"] += 1
            _totals["scoring_ms"] += sum(latencies)
            _last_run.clear()
            _last_run.update(summary)
        yield _line(summary)
    except Exception as exc:                  # e.g. CleaningGateError, out-of-order frames
        yield _line({"type": "error", "message": str(exc).replace(path, name)})
    finally:
        with _lock:
            _active.update(running=False)
        try:
            os.unlink(path)
        except OSError:
            pass


# ---------------------------------------------------------------- telemetry --

_cpu_prev = {"wall": time.perf_counter(), "proc": time.process_time()}


def _cpu() -> dict:
    cores = os.cpu_count() or 1
    if psutil is not None:
        return {"system_pct": psutil.cpu_percent(interval=None),
                "process_pct": psutil.Process().cpu_percent(interval=None) / cores,
                "cores": cores, "source": "psutil"}
    wall, proc = time.perf_counter(), time.process_time()
    dw, dp = wall - _cpu_prev["wall"], proc - _cpu_prev["proc"]
    _cpu_prev.update(wall=wall, proc=proc)
    process_pct = (dp / dw / cores * 100) if dw > 0 else 0.0
    system_pct = None
    if hasattr(os, "getloadavg"):
        system_pct = min(os.getloadavg()[0] / cores * 100, 100.0)
    return {"system_pct": system_pct, "process_pct": process_pct, "cores": cores,
            "source": "load average (install psutil for exact CPU %)"}


def _memory() -> dict:
    if psutil is not None:
        vm = psutil.virtual_memory()
        return {"used_bytes": vm.total - vm.available, "total_bytes": vm.total,
                "pct": vm.percent, "process_bytes": psutil.Process().memory_info().rss}
    info = {}
    try:                                       # Linux (the Nano)
        for row in Path("/proc/meminfo").read_text().splitlines():
            key, val = row.split(":", 1)
            info[key] = int(val.split()[0]) * 1024
        total, avail = info["MemTotal"], info["MemAvailable"]
        rss = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        return {"used_bytes": total - avail, "total_bytes": total,
                "pct": (total - avail) / total * 100, "process_bytes": rss}
    except (OSError, KeyError, ValueError):
        return {"used_bytes": None, "total_bytes": None, "pct": None, "process_bytes": None}


def _gpu() -> dict:
    """GPU is reported for context only: the v2 Defender is pure Python on
    the CPU and does not use the GPU."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"available": False}
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2).stdout.strip().splitlines()[0]
        name, util, used, total = [x.strip() for x in out.split(",")]

        def num(x):
            try:
                return float(x)
            except ValueError:
                return None           # e.g. "[N/A]" on unified-memory GB10
        return {"available": True, "name": name, "util_pct": num(util),
                "mem_used_mb": num(used), "mem_total_mb": num(total)}
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return {"available": False}


@app.get("/api/system")
def system():
    with _lock:
        totals = dict(_totals)
        last = dict(_last_run)
        active = dict(_active)
    avg_ms = totals["scoring_ms"] / totals["windows_scored"] if totals["windows_scored"] else None
    if active["running"] and active["started"]:
        run_s = max(time.time() - active["started"], 1e-6)
        active["windows_per_s"] = active["windows"] / run_s
        active["frames_per_s"] = active["frames"] / run_s
    return {
        "host": platform.node(),
        "machine": platform.machine(),
        "os": f"{platform.system()} {platform.release()}",
        "uptime_s": round(time.time() - _started),
        "cpu": _cpu(),
        "memory": _memory(),
        "gpu": _gpu(),
        "model": {"version": MODEL_VERSION, "bytes": model_bytes(),
                  "runtime": "Pure Python, CPU only (no GPU, no ML runtime)"},
        "totals": {**totals, "mean_latency_ms": avg_ms},
        "last_run": last,
        "active": active,
    }
