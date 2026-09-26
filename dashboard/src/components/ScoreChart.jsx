// Attack score over time (defender's score per window) vs. the threshold.
// Hover for window details, click anywhere to jump the replay there.
import { useEffect, useRef, useState } from "react";

const H = 190;
const PAD = { top: 12, right: 12, bottom: 22, left: 34 };

export default function ScoreChart({ run, history, index, onSeek }) {
  const wrapRef = useRef(null);
  const [width, setWidth] = useState(800);
  const [hover, setHover] = useState(null);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return undefined;
    const ro = new ResizeObserver(([entry]) => setWidth(Math.max(entry.contentRect.width, 280)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const total = run.length;
  const innerW = width - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const x = (i) => PAD.left + (total > 1 ? (i / (total - 1)) * innerW : 0);
  const y = (v) => PAD.top + (1 - v) * innerH;
  const threshold = run[0]?.threshold ?? 1;

  const line = history.map((h, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(h.attack_score).toFixed(1)}`).join(" ");
  const area = history.length > 1 ? `${line} L${x(history.length - 1)},${y(0)} L${x(0)},${y(0)} Z` : "";

  function indexFromEvent(e) {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - rect.left - PAD.left;
    return Math.min(Math.max(Math.round((px / innerW) * (total - 1)), 0), total - 1);
  }

  const hovered = hover !== null ? run[hover] : null;
  const hoverPlayed = hover !== null && hover < history.length;

  return (
    <section className="card score-chart">
      <div className="card-head">
        <h2>Attack score over time</h2>
        <span className="legend">
          <span className="legend-line legend-line-blue" /> Defender score
          <span className="legend-dot legend-dot-critical" /> Attack
          <span className="legend-line legend-line-dash" /> Threshold
        </span>
      </div>

      <div className="chart-wrap" ref={wrapRef}>
        <svg
          width={width}
          height={H}
          className="chart-svg"
          onMouseMove={(e) => setHover(indexFromEvent(e))}
          onMouseLeave={() => setHover(null)}
          onClick={(e) => onSeek(indexFromEvent(e))}
          role="img"
          aria-label="Attack score per window. Click to jump the replay."
        >
          <defs>
            <linearGradient id="scoreFill" x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%" stopColor="var(--defender)" stopOpacity="0.35" />
              <stop offset="100%" stopColor="var(--defender)" stopOpacity="0" />
            </linearGradient>
          </defs>

          {[0, 0.5, 1].map((v) => (
            <g key={v}>
              <line x1={PAD.left} x2={width - PAD.right} y1={y(v)} y2={y(v)} className="chart-grid" />
              <text x={PAD.left - 6} y={y(v) + 4} className="chart-axis" textAnchor="end">{v.toFixed(1)}</text>
            </g>
          ))}
          <text x={PAD.left} y={H - 4} className="chart-axis">window 1</text>
          <text x={width - PAD.right} y={H - 4} className="chart-axis" textAnchor="end">{total}</text>

          {area && <path d={area} fill="url(#scoreFill)" />}
          {line && <path d={line} className="chart-line" />}

          <line x1={PAD.left} x2={width - PAD.right} y1={y(threshold)} y2={y(threshold)} className="chart-threshold" />

          {history.map((h, i) =>
            h.decision === "ATTACK" ? (
              <circle key={h.window_id} cx={x(i)} cy={y(h.attack_score)} r="3.5" className="chart-attack-dot" />
            ) : null
          )}

          {history.length > 0 && (
            <g>
              <line x1={x(index)} x2={x(index)} y1={PAD.top} y2={y(0)} className="chart-now" />
              <circle cx={x(index)} cy={y(history[history.length - 1].attack_score)} r="5" className="chart-now-dot" />
            </g>
          )}

          {hover !== null && (
            <line x1={x(hover)} x2={x(hover)} y1={PAD.top} y2={y(0)} className="chart-crosshair" />
          )}
        </svg>

        {hovered && (
          <div
            className="chart-tooltip"
            style={{ left: Math.min(Math.max(x(hover), 90), width - 90) }}
          >
            <div className="mono">{hovered.window_id}</div>
            {hoverPlayed ? (
              <div>
                score <span className="mono">{hovered.attack_score.toFixed(3)}</span>{" "}
                <span className={hovered.decision === "ATTACK" ? "tt-red" : "tt-green"}>{hovered.decision}</span>
              </div>
            ) : (
              <div className="hint">not played yet</div>
            )}
            <div className="hint">click to jump here</div>
          </div>
        )}
      </div>
    </section>
  );
}
