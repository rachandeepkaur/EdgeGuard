import { parseEvidence } from "../utils/parseEvidence";
import { ACTIONS } from "./actionMeta";
import { formatLatency } from "../utils/latency";
import { useAnimatedNumber } from "../utils/useAnimatedNumber";

// Attacker side (Red Team): what is being hit right now and so far.
export function AttackerPanel({ output, history }) {
  const { canId, stage } = parseEvidence(output?.evidence);
  const isAttack = output?.decision === "ATTACK";
  const attacks = history.filter((h) => h.decision === "ATTACK");

  const counts = {};
  for (const h of attacks) {
    const id = parseEvidence(h.evidence).canId;
    if (id) counts[id] = (counts[id] || 0) + 1;
  }
  const top = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
  const peak = attacks.reduce((m, h) => Math.max(m, h.attack_score), 0);

  return (
    <section className={`card side-card side-attacker ${isAttack ? "side-active" : ""}`}>
      <div className="card-head">
        <h2>Attacker <span className="side-tag">Red Team</span></h2>
        {isAttack && <span className="status-badge status-critical">Active</span>}
      </div>

      <span className="label">Affected CAN ID</span>
      <div className={`can-id-value mono ${canId ? "" : "can-id-unknown"}`}>{canId ?? "-"}</div>
      {stage && <span className="hint">from {stage} evidence</span>}

      <div className="mini-stats">
        <div>
          <span className="label">Top target</span>
          <span className="mono mini-val">{top ? `${top[0]} ×${top[1]}` : "-"}</span>
        </div>
        <div>
          <span className="label">Peak score</span>
          <span className="mono mini-val">{attacks.length ? peak.toFixed(3) : "-"}</span>
        </div>
      </div>
    </section>
  );
}

// Defender side (Blue Team): how the simulated consumer responds.
export function DefenderPanel({ output, history, attackAction, onAttackActionChange }) {
  const action = !output ? null : output.decision === "ATTACK" ? attackAction : "SIMULATED_FORWARD";
  const meta = action ? ACTIONS[action] : null;
  const clean = history.filter((h) => h.decision === "ACCEPT").length;
  const blocked = history.length - clean;
  const shownClean = Math.round(useAnimatedNumber(clean));
  const shownBlocked = Math.round(useAnimatedNumber(blocked));

  return (
    <section className="card side-card side-defender">
      <div className="card-head">
        <h2>Defender <span className="side-tag">Blue Team</span></h2>
      </div>

      <span className="label">Response</span>
      {meta ? (
        <span className={`status-badge ${meta.tone}`}>{meta.label}</span>
      ) : (
        <span className="hint">Waiting for first window</span>
      )}

      <div className="mini-stats">
        <div>
          <span className="label">Passed clean</span>
          <span className="mono mini-val">{shownClean}</span>
        </div>
        <div>
          <span className="label">Flagged</span>
          <span className="mono mini-val">{shownBlocked}</span>
        </div>
        <div title="Inference time for this window, measured by the Defender (window collection excluded)">
          <span className="label">Decided in</span>
          <span className="mono mini-val">{formatLatency(output?.latency_ms) ?? "-"}</span>
        </div>
      </div>

      <div className="toggle-row">
        <span className="label">On attack</span>
        <div className="speed-group" role="group" aria-label="Attack action (team setting)">
          {["SIMULATED_ALERT", "SIMULATED_ISOLATION"].map((a) => (
            <button
              key={a}
              className={`btn btn-chip ${attackAction === a ? "btn-chip-active" : ""}`}
              onClick={() => onAttackActionChange(a)}
            >
              {a === "SIMULATED_ALERT" ? "Alert" : "Isolate"}
            </button>
          ))}
        </div>
      </div>
    </section>
  );
}

export function RoutingPanel({ output, escalated, quality }) {
  return (
    <section className="card routing-panel">
      <div className="card-head">
        <h2>Local / cloud routing</h2>
      </div>
      <div className="routing-row">
        <span className="status-badge status-good">Local decision</span>
        {output && (
          <span
            className={`status-badge ${escalated ? "status-warning" : "status-confident"}`}
            title={
              escalated
                ? "Score is inside the uncertainty band: a real run would send a sanitized bundle to the cloud for a second opinion."
                : "Score is outside the uncertainty band: the local decision stands, no cloud call."
            }
          >
            {escalated ? "Escalate to cloud" : "Confident, stays local"}
          </span>
        )}
      </div>

      {quality && (
        <>
          <div className="metric-chips metric-chips-wide">
            <Chip value={`${(quality.escalationPrecision * 100).toFixed(1)}%`} label="escalation precision" />
            <Chip value={`${quality.escalatedFalsePositives}/${quality.localFalsePositives}`} label="false alarms reviewed" />
            <Chip value={`${quality.escalatedFalseNegatives}/${quality.localFalseNegatives}`} label="misses caught" />
            <Chip value={`${Math.round(quality.bundleBytesMean)} B`} label="per escalation, 0 raw CAN" />
          </div>
          <details className="routing-details">
            <summary>Details</summary>
            <p className="hint">
              The local decision never waits on the cloud. Numbers from {quality.source.split(" ")[0]}.
              Fields sent: {quality.bundleFields.join(", ")}.
            </p>
          </details>
        </>
      )}
    </section>
  );
}

function Chip({ value, label }) {
  return (
    <div className="metric-chip">
      <span className="mono metric-chip-val">{value}</span>
      <span className="metric-chip-lab">{label}</span>
    </div>
  );
}
