// Row of live KPI tiles, recomputed from the windows played so far.
import { isInEscalationBand } from "../utils/escalation";
import { useAnimatedNumber } from "../utils/useAnimatedNumber";

export default function KpiTiles({ history, total, bandHalfWidth }) {
  const played = history.length;
  const attacks = history.filter((h) => h.decision === "ATTACK").length;
  const attackRate = played ? (attacks / played) * 100 : 0;
  const escalated = history.filter((h) => isInEscalationBand(h, bandHalfWidth)).length;
  const latencies = history.map((h) => h.latency_ms).filter((v) => typeof v === "number");
  const avgLatency = latencies.length ? latencies.reduce((a, b) => a + b, 0) / latencies.length : null;
  // Displayed values glide to the real ones; the real ones drive everything else.
  const shownPlayed = Math.round(useAnimatedNumber(played));
  const shownAttacks = Math.round(useAnimatedNumber(attacks));
  const shownLatency = useAnimatedNumber(avgLatency);
  const shownEscalated = Math.round(useAnimatedNumber(escalated));

  return (
    <section className="kpi-row" aria-label="Key metrics">
      <Tile tone="cyan" label="Windows scored" value={shownPlayed} unit={`/ ${total}`}>
        <div className="kpi-bar">
          <div className="kpi-bar-fill" style={{ width: `${total ? (played / total) * 100 : 0}%` }} />
        </div>
      </Tile>

      <Tile tone="red" label="Attacker hits" value={shownAttacks} unit="windows">
        <span className="kpi-sub">{attackRate.toFixed(1)}% of traffic</span>
      </Tile>

      <Tile tone="blue" label="Defender latency" value={shownLatency == null ? "-" : shownLatency.toFixed(2)} unit="ms avg">
        <Sparkline values={latencies.slice(-60)} />
      </Tile>

      <Tile tone="amber" label="Sent to cloud" value={shownEscalated} unit="windows">
        <span className="kpi-sub">{played ? ((escalated / played) * 100).toFixed(1) : "0.0"}% in escalation band</span>
      </Tile>
    </section>
  );
}

function Tile({ tone, label, value, unit, children }) {
  return (
    <div className={`kpi-tile kpi-${tone}`}>
      <span className="kpi-label">{label}</span>
      <div className="kpi-value mono">
        {value}
        {unit && <span className="kpi-unit">{unit}</span>}
      </div>
      {children}
    </div>
  );
}

function Sparkline({ values }) {
  if (values.length < 2) return <div className="kpi-spark kpi-spark-empty" />;
  const w = 160;
  const h = 28;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const pts = values.map((v, i) => [(i / (values.length - 1)) * w, h - 2 - ((v - min) / span) * (h - 4)]);
  const line = pts.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
  const [lx, ly] = pts[pts.length - 1];
  return (
    <svg className="kpi-spark" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" aria-label="Latency, recent windows">
      <title>{`Latency, last ${values.length} windows: min ${min.toFixed(2)} ms, max ${max.toFixed(2)} ms`}</title>
      <path d={`${line} L${w},${h} L0,${h} Z`} className="kpi-spark-area" />
      <path d={line} className="kpi-spark-line" vectorEffect="non-scaling-stroke" />
      <circle cx={lx} cy={ly} r="2.5" className="kpi-spark-dot" />
    </svg>
  );
}
