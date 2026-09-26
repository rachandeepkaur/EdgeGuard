import { useState } from "react";
import { formatLatency } from "../utils/latency";
import { OUTCOME_TEXT } from "../utils/outcome";

// Security-console style log of every replayed window, newest on top.
// Hover a line for the Defender's full evidence text.
export default function EventLog({ events }) {
  const [filter, setFilter] = useState("all");
  const shown = filter === "alerts" ? events.filter((e) => e.decision === "ATTACK" || e.outcome) : events;
  const alertCount = events.filter((e) => e.decision === "ATTACK").length;

  return (
    <section className="card event-log">
      <div className="card-head">
        <h2>Event log</h2>
        <span className="card-head-meta">
          <span className="hint mono">{events.length} events &middot; {alertCount} alerts</span>
          <div className="speed-group" role="group" aria-label="Filter events">
            {[["all", "All"], ["alerts", "Alerts"]].map(([id, label]) => (
              <button
                key={id}
                className={`btn btn-chip ${filter === id ? "btn-chip-active" : ""}`}
                onClick={() => setFilter(id)}
                title={id === "alerts" ? "ATTACK decisions, plus missed attacks and false alarms" : "Every window"}
              >
                {label}
              </button>
            ))}
          </div>
        </span>
      </div>

      <ol className="log-lines mono" aria-live="off">
        {shown.length === 0 && <li className="log-empty">Waiting for events&hellip;</li>}
        {shown.map((e) => {
          const isAttack = e.decision === "ATTACK";
          return (
            <li key={e.seq} className={`log-line ${isAttack ? "log-attack" : "log-accept"}`} title={e.evidence}>
              <span className="log-time">{e.time.toLocaleTimeString([], { hour12: false })}</span>
              <span className="log-win">{e.windowId}</span>
              <span className="log-dec">{e.decision}</span>
              <span className="log-can">{e.canId ? `CAN ${e.canId}` : "-"}</span>
              <span className="log-check">{e.check ?? "no anomaly"}</span>
              <span className="log-lat">{formatLatency(e.latencyMs) ?? "-"}</span>
              <span className="log-act">&rarr; {e.action}</span>
              {e.outcome && <span className="log-outcome">{OUTCOME_TEXT[e.outcome].badge.toUpperCase()}</span>}
            </li>
          );
        })}
      </ol>
    </section>
  );
}
