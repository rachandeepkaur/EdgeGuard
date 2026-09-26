// "Analyze a capture" drawer: pick a ROAD-format .log file from this device,
// stream it to the local analysis server, watch the real Defender score it
// window by window, then replay the result in the dashboard.
import { useEffect, useRef, useState } from "react";
import { analyzeCapture } from "../data/analyzeClient";

const START_CMD =
  "cd ~/Desktop/EdgeGuard && source .venv/bin/activate && python -m uvicorn integration.analyze_server:app --host 127.0.0.1 --port 8000";

function fmtBytes(n) {
  if (n == null) return "-";
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)} KB`;
  return `${n} B`;
}

function fmtTime(s) {
  const m = Math.floor(s / 60);
  const r = Math.floor(s % 60);
  return `${String(m).padStart(2, "0")}:${String(r).padStart(2, "0")}`;
}

export default function UploadPanel({ open, onClose, onReplay, uploadedName, onUseDemo, source }) {
  const [state, setState] = useState({ phase: "idle" });
  const [drag, setDrag] = useState(false);
  const [now, setNow] = useState(0);
  const inputRef = useRef(null);
  const abortRef = useRef(null);
  const runRef = useRef([]);

  const busy = state.phase === "uploading" || state.phase === "processing";

  useEffect(() => {
    if (!busy) return undefined;
    const t = setInterval(() => setNow(performance.now()), 250);
    return () => clearInterval(t);
  }, [busy]);

  useEffect(() => () => abortRef.current?.(), []);

  function start(file) {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".log")) {
      setState({ phase: "error", message: "Please choose a ROAD-format .log capture." });
      return;
    }
    runRef.current = [];
    const t0 = performance.now();
    setNow(t0);
    setState({ phase: "uploading", file, t0, loaded: 0, total: file.size });

    abortRef.current = analyzeCapture(file, (ev) => {
      if (ev.type === "upload") {
        setState((s) => ({ ...s, loaded: ev.loaded, total: ev.total }));
      } else if (ev.type === "meta") {
        setState((s) => ({ ...s, phase: "processing", meta: ev, windows: 0, attacks: 0, frames: 0 }));
      } else if (ev.type === "window") {
        runRef.current.push(ev);
      } else if (ev.type === "progress") {
        const run = runRef.current;
        setState((s) => ({
          ...s,
          windows: run.length,
          frames: ev.frames,
          attacks: run.filter((w) => w.decision === "ATTACK").length,
        }));
      } else if (ev.type === "done") {
        setState((s) => ({ ...s, phase: "done", summary: ev, windows: ev.windows, attacks: ev.attacks }));
      } else if (ev.type === "error") {
        setState((s) => ({ ...s, phase: "error", message: ev.message, offline: ev.offline }));
      }
    });
  }

  function replay() {
    const meta = state.meta;
    onReplay({
      run: runRef.current,
      meta: {
        captureId: meta.capture_id,
        modelVersion: meta.model_version,
        threshold: meta.threshold,
        escalationBandHalfWidth: meta.escalation_band_half_width,
        fileName: meta.file_name,
      },
      summary: state.summary,
    });
  }

  function reset() {
    abortRef.current?.();
    setState({ phase: "idle" });
  }

  if (!open) return null;

  const elapsed = busy ? (now - state.t0) / 1000 : state.summary?.elapsed_s ?? 0;
  const uploadPct = state.total ? Math.round((state.loaded / state.total) * 100) : 0;
  const total = state.meta?.total_windows || 0;
  const scorePct = total ? Math.min(100, Math.round(((state.windows || 0) / total) * 100)) : 0;
  const stepIdx = { idle: -1, uploading: 0, processing: 1, done: 3, error: -1 }[state.phase];

  return (
    <div className="drawer-overlay" onClick={busy ? undefined : onClose}>
      <aside className="drawer" onClick={(e) => e.stopPropagation()} role="dialog" aria-label="Analyze a capture">
        <div className="drawer-head">
          <div>
            <span className="eyebrow">Local &middot; on this device</span>
            <h2 className="drawer-title">Analyze a capture</h2>
          </div>
          <button className="icon-btn" onClick={onClose} aria-label="Close">
            <svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M3 3 L13 13 M13 3 L3 13" /></svg>
          </button>
        </div>

        {(state.phase === "idle" || state.phase === "error") && (
          <label
            className={`dropzone ${drag ? "dropzone-drag" : ""}`}
            onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => { e.preventDefault(); setDrag(false); start(e.dataTransfer.files[0]); }}
          >
            <input
              ref={inputRef}
              type="file"
              accept=".log"
              hidden
              onChange={(e) => { start(e.target.files[0]); e.target.value = ""; }}
            />
            <svg viewBox="0 0 40 40" width="44" height="44" fill="none" stroke="#22d3ee" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M8 26 V32 H32 V26" /><path d="M20 24 V8" /><path d="M13 15 L20 8 L27 15" />
            </svg>
            <strong>Open CAN capture</strong>
            <span className="hint">Select a ROAD .log file from this device</span>
            <span className="dropzone-tag">.LOG ONLY</span>
          </label>
        )}

        {state.phase === "error" && (
          <div className="drawer-error">
            <strong>{state.message}</strong>
            {state.offline && (
              <>
                <span className="hint">Start it in a new Terminal window, then try again:</span>
                <code className="cmd">{START_CMD}</code>
              </>
            )}
          </div>
        )}

        {stepIdx >= 0 && (
          <>
            <div className="file-chip">
              <span className="mono">{state.file.name}</span>
              <span className="hint">{fmtBytes(state.file.size)}</span>
            </div>

            <ol className="steps">
              {["Upload to edge", "Clean, window & score", "Ready to replay"].map((label, i) => (
                <li key={label} className={i < stepIdx ? "step-done" : i === stepIdx ? "step-active" : ""}>
                  <span className="step-dot">{i < stepIdx ? "✓" : i + 1}</span>
                  {label}
                </li>
              ))}
            </ol>

            <div className="proc-bar">
              <span className="label">{state.phase === "uploading" ? "Uploading" : state.phase === "processing" ? "Scoring" : "Done"}</span>
              <div className="progress-track">
                <div
                  className="progress-fill"
                  style={{ width: `${state.phase === "uploading" ? uploadPct : state.phase === "done" ? 100 : scorePct}%` }}
                />
              </div>
              <span className="mono proc-elapsed">{fmtTime(elapsed)}</span>
            </div>

            {state.phase !== "uploading" && (
              <div className="drawer-stats">
                <Stat label="Windows" value={`${state.windows ?? 0}${total ? ` / ${total}` : ""}`} tone="cyan" />
                <Stat label="Attacks flagged" value={state.attacks ?? 0} tone="red" />
                <Stat label="CAN frames" value={(state.summary?.frames_read ?? state.frames ?? 0).toLocaleString()} tone="blue" />
                <Stat
                  label="Windows / s"
                  value={state.summary?.windows_per_s ?? (elapsed > 0 ? Math.round((state.windows || 0) / elapsed) : 0)}
                  tone="amber"
                />
              </div>
            )}

            {state.phase === "done" && (
              <div className="drawer-actions">
                <button className="btn btn-primary-lg" onClick={replay}>Replay in dashboard</button>
                <button className="btn" onClick={reset}>Analyze another</button>
              </div>
            )}
            {busy && (
              <div className="drawer-actions">
                <button className="btn" onClick={reset}>Cancel</button>
              </div>
            )}
          </>
        )}

        <div className="drawer-foot">
          <span className="hint">
            Now showing: <strong>{source === "upload" ? uploadedName : "demo replay (3 ROAD captures)"}</strong>
          </span>
          {source === "upload" && (
            <button className="btn btn-chip" onClick={onUseDemo}>Back to demo</button>
          )}
        </div>
      </aside>
    </div>
  );
}

function Stat({ label, value, tone }) {
  return (
    <div className={`kpi-tile kpi-${tone} drawer-stat`}>
      <span className="kpi-label">{label}</span>
      <div className="kpi-value mono">{value}</div>
    </div>
  );
}
