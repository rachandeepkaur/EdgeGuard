import { parseEvidence } from "../utils/parseEvidence";
import { useCarDrive } from "../utils/useCarDrive";
import { formatLatency } from "../utils/latency";
import { outcomeOf, OUTCOME_TEXT } from "../utils/outcome";

// Glowing wireframe car scene -- a 2D SVG outline drawing with a neon glow
// filter, not a literal 3D wireframe render (that needs a 3D engine and
// buys little over this for a demo). Cyan while ACCEPT, red while ATTACK,
// with a checkpoint gate (the Blue Team detector) and incoming Red Team
// arrows during an attack window.

export default function CarsView({ output, attackAction, isPlaying = true }) {
  const isAttack = output?.decision === "ATTACK";
  // Hook runs every render regardless (rules of hooks) -- it just has
  // nothing to react to alert-wise until `output` exists.
  const { x, y, angle, opacity } = useCarDrive(isAttack, { running: isPlaying });

  if (!output) {
    return (
      <section className="card cars-scene">
        <p className="hint">Press play on the replay to start the scene.</p>
      </section>
    );
  }

  const { canId } = parseEvidence(output.evidence);
  const scorePct = Math.round(output.attack_score * 100);
  const thresholdPct = Math.round(output.threshold * 100);
  const latency = formatLatency(output.latency_ms);
  const outcome = outcomeOf(output);
  const tilt = Math.max(Math.min(angle * 0.2, 12), -12); // subtle lean into curves, clamped

  return (
    <section className="card cars-scene">
      <div className="card-head">
        <h2>Fleet view</h2>
        <span className="card-head-meta">
          {outcome && (
            <span className="status-badge status-warning" title={OUTCOME_TEXT[outcome].tooltip}>
              {OUTCOME_TEXT[outcome].badge}
            </span>
          )}
          <span className={`status-badge ${isAttack ? "status-critical" : "status-good"}`}>
            {isAttack ? "Red Team attack" : "Normal traffic"}
          </span>
        </span>
      </div>

      {/* No key={tick} here on purpose: the car's "draw itself" animation
          should play once when this tab is opened, not restart every
          window (which, at 4x playback, would interrupt it every ~75ms and
          leave the car permanently half-drawn -- caught by screenshotting
          an attack frame during fast playback). Color/state changes are
          handled by className swaps below, which re-render without
          remounting. */}
      <div className={`cars-stage ${isAttack ? "cars-stage-attack" : ""}`}>
        <svg viewBox="0 0 900 260" className="cars-svg">
          <path className="road-base" d="M 0 190 C 200 130, 380 230, 480 170 C 620 90, 760 190, 900 140" />
          <path className={`road-glow ${isAttack ? "road-glow-critical" : "road-glow-good"}`}
                d="M 0 190 C 200 130, 380 230, 480 170 C 620 90, 760 190, 900 140" />

          {/* checkpoint gate: the Blue Team detector */}
          <g className="gate" transform="translate(480,60)">
            <path d="M -50 0 L 50 0" className="gate-bar" />
            <path d="M -50 0 L -50 110" className="gate-post" />
            <path d="M 50 0 L 50 110" className="gate-post" />
            <circle cx="0" cy="0" r="9" className={isAttack ? "gate-light-critical" : "gate-light-defender"} />
          </g>

          {/* incoming Red Team vectors, attack only */}
          {isAttack && (
            <g className="attack-arrows">
              <path d="M 400 -10 L 465 55" markerEnd="url(#arrow)" />
              <path d="M 560 -10 L 495 55" markerEnd="url(#arrow)" />
            </g>
          )}
          <defs>
            <marker id="arrow" markerWidth="10" markerHeight="10" refX="5" refY="5" orient="auto">
              <path d="M0,0 L10,5 L0,10 z" className="arrow-head" />
            </marker>
          </defs>

          {/* wireframe car, drawn as glowing outline strokes. Drives one-way
              (never reverses); on reaching the end it fades out, jumps back
              to the start, and fades in -- a seamless-looking loop on an
              open road. Freezes for 3s the moment an alert fires. */}
          <g className={`car-wireframe ${isAttack ? "car-wireframe-critical" : "car-wireframe-good"}`}
             style={{ opacity }}
             transform={`translate(${x},${y}) rotate(${tilt})`}>
            <path d="M -70 10 C -70 -18 -46 -34 -20 -34 L 18 -34 C 42 -34 56 -20 66 4
                     L 78 8 C 86 10 86 26 78 28 L -78 28 C -86 26 -86 10 -78 8 Z" />
            <path d="M -34 -34 L -24 -12 L 28 -12 L 20 -34" />
            <line x1="-24" y1="-12" x2="-20" y2="28" />
            <line x1="20" y1="-12" x2="24" y2="28" />
            <circle cx="-42" cy="28" r="14" />
            <circle cx="42" cy="28" r="14" />
            <line x1="-78" y1="8" x2="78" y2="8" />
          </g>
        </svg>

        <div className="cars-readout">
          <div className={`speed-sign ${isAttack ? "bad" : "good"}`}>
            <div className="num mono">{output.attack_score.toFixed(2)}</div>
            <div className="lab">SCORE</div>
          </div>
          <div className="limit-sign mono">{thresholdPct}<span className="limit-unit">THR</span></div>
        </div>
      </div>

      <div className="cars-footline hint">
        {output.window_id} &middot; {canId ? `CAN ${canId}` : "no CAN ID in evidence"} &middot; score {scorePct}% vs threshold {thresholdPct}%
        {latency && ` · decided in ${latency}`}
        {isAttack && ` · responding with ${attackAction === "SIMULATED_ISOLATION" ? "isolation" : "alert"}`}
        {outcome && <span className="outcome-text">{` · ${OUTCOME_TEXT[outcome].caption}`}</span>}
      </div>
    </section>
  );
}
