// EdgeGuard architecture, drawn as a glowing "circuit board".
//
// Live lane (top): traffic window -> Defender server -> simulated consumer.
// Inside the Defender server (per the team's hand-drawn design):
//   - Rule DB: the shipped rule engine, Stage 1 timing + Stage 2 payload
//     checks (defender/defender.py). This is what the replay scores with.
//   - Timing CNN: causal Conv1d over 100 ms timing bins (defender/timing_cnn.py).
//   - Defender LLM: FUTURE, drawn dashed and never animated. On ROAD's
//     anonymized DBC it can't beat the rules; its value is plain-language
//     explanations on a real labelled fleet (defender/defender_llm.py).
// Rule DB and Timing CNN are combined by OR (defender/fused_defender.py):
// ATTACK if either fires. Not a weighted sum -- their scores live on
// incomparable scales. A weighted sum is the plan once the LLM joins.
// Glowing packets travel along the traces; each box lights up the moment a
// packet reaches it. An attack turns the packets red and the server red.
//
// Offline lane (bottom): the adversarial testing & hardening loop
// (ROAD source -> Attacker -> Defender -> ground truth -> train/test ->
// evidence gate), with an amber feedback trace from the evidence gate back
// up to the live Defender (the gate decides which model version is used).
// In the real system this lane runs once per model comparison, not per
// window, so it loops here as an illustration and is labeled offline.
//
// All motion is SVG SMIL (animateMotion / animate), timed so both lanes
// share one step length: a packet always arrives exactly when the next box
// lights up.

const STEP = 2; // seconds per box, both lanes (slow and readable)
const START = 1.2; // wait for the entrance pop-in before the loop starts

const C = {
  cyan: "#22d3ee",
  blue: "#3b82f6",
  red: "#f04452",
  amber: "#fbbf24",
  green: "#22c55e",
  violet: "#a78bfa",
  pink: "#e879f9",
  slate: "#94a3b8",
};

const ACTION_STYLE = {
  SIMULATED_FORWARD: { color: C.green, icon: "forward", label: "Forwarded" },
  SIMULATED_ALERT: { color: C.amber, icon: "bell", label: "Alert raised" },
  SIMULATED_ISOLATION: { color: C.red, icon: "lock", label: "Isolated" },
};

const VIEW_H = 600;

// Live lane geometry
const LIVE_Y = 190; // shared centre line: traffic, Timing CNN, OR, action
const LIVE_W = 220;
const LIVE_H = 88;
const TRAFFIC_X = 150;
const ACTION_X = 1060;
const SRV = { x: 320, y: 54, w: 560, h: 272 }; // Defender server container
const SUB_X = 500; // centre x of the three sub-models
const SUB_W = 250;
const SUB_H = 52;
const SUB_Y = { rule: 116, cnn: 190, llm: 264 };
const OR_X = 745;
const OR_R = 24;

// Offline lane geometry
const OFF_Y = 520;
const OFF_W = 156;
const OFF_H = 72;
const OFF_X = [98, 299, 500, 701, 902, 1103];

export default function ArchitectureDiagram({ output, attackAction }) {
  const isAttack = output?.decision === "ATTACK";
  const action = !output ? "SIMULATED_FORWARD" : isAttack ? attackAction : "SIMULATED_FORWARD";
  const act = ACTION_STYLE[action];
  const trafficColor = isAttack ? C.red : C.cyan;
  const serverColor = isAttack ? C.red : C.blue;

  const offline = [
    { label: "ROAD source", color: C.cyan, icon: "db" },
    { label: "Attacker", color: C.red, icon: "target" },
    { label: "Defender", color: C.blue, icon: "shield" },
    { label: "Missed attack?", color: C.violet, icon: "check" },
    { label: "Train vs. test", color: C.pink, icon: "split" },
    { label: "v1 vs fusion", color: C.amber, icon: "gate" },
  ];

  const liveT = STEP * 3; // traffic -> defender server -> action
  const offT = STEP * offline.length;

  const subLeft = SUB_X - SUB_W / 2;
  const subRight = SUB_X + SUB_W / 2;
  const fanX = SRV.x + 22; // where the incoming trace fans out to the sub-models
  const inD = `M ${TRAFFIC_X + LIVE_W / 2} ${LIVE_Y} L ${fanX} ${LIVE_Y}`;
  const outD = `M ${OR_X + OR_R} ${LIVE_Y} L ${ACTION_X - LIVE_W / 2} ${LIVE_Y}`;
  const fanD = (y) => `M ${fanX} ${LIVE_Y} C ${fanX + 20} ${LIVE_Y}, ${subLeft - 20} ${y}, ${subLeft} ${y}`;
  const joinD = (y) => `M ${subRight} ${y} C ${subRight + 50} ${y}, ${OR_X - OR_R - 40} ${LIVE_Y}, ${OR_X - OR_R} ${LIVE_Y}`;

  const offTraces = [0, 1, 2, 3, 4].map((i) => ({
    id: `eg-off-${i}`,
    d: `M ${OFF_X[i] + OFF_W / 2} ${OFF_Y} L ${OFF_X[i + 1] - OFF_W / 2} ${OFF_Y}`,
    color: offline[i].color,
  }));
  // Drawn defender -> gate so the label reads left-to-right; the packet
  // rides it in reverse (gate -> defender).
  const loopD = `M ${SRV.x + SRV.w - 40} ${SRV.y + SRV.h} C ${SRV.x + SRV.w} 420, ${OFF_X[5]} 400, ${OFF_X[5]} ${OFF_Y - OFF_H / 2}`;

  return (
    <section className="card arch-diagram">
      <div className="card-head">
        <h2>System architecture</h2>
        <span className={`status-badge ${isAttack ? "status-critical" : "status-good"}`}>
          {isAttack ? "Attack blocked" : "Traffic clean"}
        </span>
      </div>

      <div className="arch-board">
        <svg viewBox={`0 0 1200 ${VIEW_H}`} className="arch-svg" role="img"
             aria-label="EdgeGuard architecture: live detection through the Defender server (Rule DB, Timing CNN, future Defender LLM) and the offline hardening loop">
          <defs>
            <pattern id="eg-grid" width="24" height="24" patternUnits="userSpaceOnUse">
              <circle cx="1" cy="1" r="1" fill="rgba(148,163,184,0.13)" />
            </pattern>
            <linearGradient id="eg-scan" x1="0" x2="1">
              <stop offset="0%" stopColor="#22d3ee" stopOpacity="0" />
              <stop offset="50%" stopColor="#22d3ee" stopOpacity="0.10" />
              <stop offset="100%" stopColor="#22d3ee" stopOpacity="0" />
            </linearGradient>
            <filter id="eg-glow" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="4" result="b" />
              <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
            </filter>
            <filter id="eg-soft" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="14" />
            </filter>
            <path id="eg-in" d={inD} />
            <path id="eg-out" d={outD} />
            <path id="eg-fan-rule" d={fanD(SUB_Y.rule)} />
            <path id="eg-fan-cnn" d={fanD(SUB_Y.cnn)} />
            <path id="eg-join-rule" d={joinD(SUB_Y.rule)} />
            <path id="eg-join-cnn" d={joinD(SUB_Y.cnn)} />
            {offTraces.map((t) => <path key={t.id} id={t.id} d={t.d} />)}
            <path id="eg-loop" d={loopD} />
          </defs>

          <rect width="1200" height={VIEW_H} fill="url(#eg-grid)" />
          <rect y="0" width="220" height={VIEW_H} fill="url(#eg-scan)">
            <animate attributeName="x" values="-240;1220" dur="7s" repeatCount="indefinite" />
          </rect>

          <text x="24" y="40" className="arch-lane-title" fill={C.cyan}>LIVE DETECTION</text>
          <text x="24" y="412" className="arch-lane-title" fill={C.violet}>OFFLINE · HARDENING LOOP</text>

          {/* Defender server container */}
          <g className="arch-enter" style={{ animationDelay: "0.25s" }}>
            <rect x={SRV.x} y={SRV.y} width={SRV.w} height={SRV.h} rx="20" fill={serverColor} opacity="0.05" />
            <rect x={SRV.x} y={SRV.y} width={SRV.w} height={SRV.h} rx="20" fill="none"
                  stroke={serverColor} strokeOpacity="0.55" strokeWidth="1.4" />
            <text x={SRV.x + 20} y={SRV.y + 26} className="arch-group-title" fill={serverColor}>DEFENDER SERVER</text>
            {isAttack && (
              <rect x={SRV.x - 5} y={SRV.y - 5} width={SRV.w + 10} height={SRV.h + 10} rx="24" fill="none"
                    stroke={C.red} strokeWidth="2" className="arch-alarm" />
            )}
          </g>
          <text x={SRV.x + SRV.w / 2} y={SRV.y + SRV.h + 22} textAnchor="middle" className="arch-note" fill={C.slate}>
            Fused: ATTACK if Rule DB OR Timing CNN fires &middot; this replay shows Rule DB scores
          </text>

          {/* feedback: evidence gate promotes the model the live Defender runs */}
          <Trace d={loopD} color={C.amber} dashed />
          <text className="arch-loop-label" fill={C.amber}>
            <textPath href="#eg-loop" startOffset="42%">◂ promotes model</textPath>
          </text>
          <Packets href="#eg-loop" color={C.amber} T={offT} begin={START + 5.25 * STEP} travel={0.7 * STEP} reverse />

          {/* traffic -> server -> sub-models -> OR -> action */}
          <Trace d={inD} color={trafficColor} />
          <Packets href="#eg-in" color={trafficColor} T={liveT} begin={START + 0.3 * STEP} travel={0.35 * STEP} />
          {["rule", "cnn"].map((k) => (
            <g key={k}>
              <Trace d={fanD(SUB_Y[k])} color={trafficColor} />
              <Packets href={`#eg-fan-${k}`} color={trafficColor} T={liveT} begin={START + 0.65 * STEP} travel={0.35 * STEP} />
              <Trace d={joinD(SUB_Y[k])} color={C.blue} />
              <Packets href={`#eg-join-${k}`} color={C.blue} T={liveT} begin={START + 1.3 * STEP} travel={0.35 * STEP} />
            </g>
          ))}
          {/* future LLM: dashed, no packets */}
          <Trace d={fanD(SUB_Y.llm)} color={C.slate} dashed faint />
          <Trace d={joinD(SUB_Y.llm)} color={C.slate} dashed faint />
          <Trace d={outD} color={act.color} />
          <Packets href="#eg-out" color={act.color} T={liveT} begin={START + 1.65 * STEP} travel={0.35 * STEP} />

          <Node label="Traffic window" color={trafficColor} icon="wave" x={TRAFFIC_X} y={LIVE_Y} w={LIVE_W} h={LIVE_H}
                step={1} T={liveT} begin={START} enter={0} big />
          <Node label="Rule DB" sub="Stage 1 timing + Stage 2 payload" color={C.blue} icon="rules"
                x={SUB_X} y={SUB_Y.rule} w={SUB_W} h={SUB_H} step="2a" T={liveT} begin={START + STEP} enter={0.35}
                alarm={isAttack} />
          <Node label="Timing CNN" sub="Conv1d over 100 ms timing bins" color={C.violet} icon="cnn"
                x={SUB_X} y={SUB_Y.cnn} w={SUB_W} h={SUB_H} step="2b" T={liveT} begin={START + STEP} enter={0.45} />
          <FutureNode label="Defender LLM" sub="plain-language explanations" x={SUB_X} y={SUB_Y.llm} w={SUB_W} h={SUB_H} />
          <Combiner x={OR_X} y={LIVE_Y} r={OR_R} color={serverColor} T={liveT} begin={START + 1.65 * STEP} />
          <Node label={act.label} color={act.color} icon={act.icon} x={ACTION_X} y={LIVE_Y} w={LIVE_W} h={LIVE_H}
                step={3} T={liveT} begin={START + 2 * STEP} enter={0.6} big />

          {offTraces.map((t, i) => (
            <g key={t.id}>
              <Trace d={t.d} color={t.color} />
              <Packets href={`#${t.id}`} color={t.color} T={offT} begin={START + (i + 0.3) * STEP} travel={0.7 * STEP} />
            </g>
          ))}
          {offline.map((n, i) => (
            <Node key={`o${i}`} {...n} x={OFF_X[i]} y={OFF_Y} w={OFF_W} h={OFF_H}
                  step={i + 1} T={offT} begin={START + i * STEP} enter={0.75 + i * 0.15} />
          ))}
        </svg>
      </div>
    </section>
  );
}

// The OR gate that fuses Rule DB and Timing CNN into one decision.
function Combiner({ x, y, r, color, T, begin }) {
  return (
    <g transform={`translate(${x} ${y})`}>
      <g className="arch-enter" style={{ animationDelay: "0.55s" }}>
        <circle r={r + 8} fill={color} opacity="0.08" filter="url(#eg-soft)">
          <animate attributeName="opacity" values="0.08;0.7;0.08;0.08" keyTimes="0;0.03;0.2;1"
                   dur={`${T}s`} begin={`${begin}s`} repeatCount="indefinite" />
        </circle>
        <circle r={r} fill="#0c1118" stroke={color} strokeWidth="1.8" />
        <circle r={r} fill={color} opacity="0.12" />
        <text y="5" textAnchor="middle" className="arch-or" fill={color}>OR</text>
        <text y={r + 18} textAnchor="middle" className="arch-note" fill={C.slate}>score</text>
      </g>
    </g>
  );
}

// A planned component: dashed, dimmed, never animated, tagged FUTURE.
function FutureNode({ label, sub, x, y, w, h }) {
  return (
    <g transform={`translate(${x} ${y})`}>
      <g className="arch-enter" style={{ animationDelay: "0.55s" }} opacity="0.75">
        <rect x={-w / 2} y={-h / 2} width={w} height={h} rx="16" fill="#0c1118" stroke={C.slate}
              strokeWidth="1.4" strokeDasharray="6 5" />
        <g transform={`translate(${-w / 2 + 28} 0)`}>
          <circle r="13" fill={C.slate} opacity="0.14" />
          <Icon name="chat" color={C.slate} s={0.75} />
        </g>
        <text x={-w / 2 + 50} y="-3" className="arch-label arch-label-sm" fill={C.slate}>{label}</text>
        <text x={-w / 2 + 50} y="14" className="arch-sub" fill={C.slate}>{sub}</text>
        <g transform={`translate(${w / 2 - 34} ${-h / 2})`}>
          <rect x="-28" y="-9" width="56" height="18" rx="9" fill="#0c1118" stroke={C.amber} strokeWidth="1.2" />
          <text y="4" textAnchor="middle" className="arch-tag" fill={C.amber}>FUTURE</text>
        </g>
      </g>
    </g>
  );
}

function Trace({ d, color, dashed = false, faint = false }) {
  return (
    <g opacity={faint ? 0.6 : 1}>
      {!faint && <path d={d} stroke={color} strokeOpacity="0.22" strokeWidth="6" fill="none" strokeLinecap="round" />}
      <path d={d} stroke={color} strokeOpacity="0.75" strokeWidth="1.6" fill="none"
            strokeDasharray={dashed ? "6 7" : "2 9"} strokeLinecap="round" className="arch-trace-flow" />
    </g>
  );
}

// A glowing comet: head + two fading tail dots, riding the trace once per cycle.
function Packets({ href, color, T, begin, travel, reverse = false }) {
  const f = +(travel / T).toFixed(4);
  return [0, 0.07, 0.14].map((lag, k) => (
    <circle key={k} r={[6, 4, 2.6][k]} fill={color} opacity="0" filter="url(#eg-glow)">
      <animateMotion dur={`${T}s`} begin={`${begin + lag}s`} repeatCount="indefinite"
                     keyPoints={reverse ? "1;0;0" : "0;1;1"} keyTimes={`0;${f};1`} calcMode="linear">
        <mpath href={href} />
      </animateMotion>
      <animate attributeName="opacity" dur={`${T}s`} begin={`${begin + lag}s`} repeatCount="indefinite"
               values={`0;${1 - k * 0.3};${1 - k * 0.3};0;0`} keyTimes={`0;0.005;${(f - 0.01).toFixed(4)};${f};1`} />
    </circle>
  ));
}

function Node({ label, sub, color, icon, x, y, w, h, step, T, begin, enter, big = false, alarm = false }) {
  const n = T / STEP; // boxes in this lane
  const a = (0.04 / (n / 3)).toFixed(4); // quick rise
  const b = (0.6 / n).toFixed(4); // highlight lasts ~60% of one step
  const iconX = -w / 2 + (big ? 42 : 28);
  const textX = -w / 2 + (big ? 76 : 50);

  return (
    <g transform={`translate(${x} ${y})`}>
      <g className="arch-enter" style={{ animationDelay: `${enter}s` }}>
        {/* soft aura, flares when a packet arrives */}
        <rect x={-w / 2} y={-h / 2} width={w} height={h} rx="16" fill={color} opacity="0.08" filter="url(#eg-soft)">
          <animate attributeName="opacity" values="0.08;0.75;0.08;0.08" keyTimes={`0;${a};${b};1`}
                   dur={`${T}s`} begin={`${begin}s`} repeatCount="indefinite" />
        </rect>

        {/* ripple ring */}
        <rect x={-w / 2} y={-h / 2} width={w} height={h} rx="16" fill="none" stroke={color} strokeWidth="2" opacity="0">
          <animateTransform attributeName="transform" type="scale" values="1;1.16;1.16" keyTimes={`0;${b};1`}
                            dur={`${T}s`} begin={`${begin}s`} repeatCount="indefinite" />
          <animate attributeName="opacity" values="0.9;0;0" keyTimes={`0;${b};1`}
                   dur={`${T}s`} begin={`${begin}s`} repeatCount="indefinite" />
        </rect>

        <g>
          <animateTransform attributeName="transform" type="scale" values="1;1.07;1;1"
                            keyTimes={`0;${a};${(2 * a).toFixed(4)};1`} dur={`${T}s`} begin={`${begin}s`} repeatCount="indefinite" />
          <rect x={-w / 2} y={-h / 2} width={w} height={h} rx="16" fill="#0c1118" stroke={color} strokeWidth="1.6" />
          <rect x={-w / 2} y={-h / 2} width={w} height={h} rx="16" fill={color} opacity="0.1" />
          {/* bright edge that lights up with the box */}
          <rect x={-w / 2} y={-h / 2} width={w} height={h} rx="16" fill="none" stroke={color} strokeWidth="3"
                opacity="0" filter="url(#eg-glow)">
            <animate attributeName="opacity" values="0;1;0;0" keyTimes={`0;${a};${b};1`}
                     dur={`${T}s`} begin={`${begin}s`} repeatCount="indefinite" />
          </rect>

          <text x={w / 2 - 12} y={-h / 2 + 18} className="arch-step" textAnchor="end" fill={color}>
            {String(step).padStart(2, "0")}
          </text>

          <g transform={`translate(${iconX} 0)`}>
            {icon === "shield" && (
              <circle r={big ? 24 : 17} fill="none" stroke={alarm ? C.red : color} strokeWidth="1.4"
                      strokeDasharray="3 5" opacity="0.8">
                <animateTransform attributeName="transform" type="rotate" from="0" to="360" dur="6s" repeatCount="indefinite" />
              </circle>
            )}
            <circle r={big ? 18 : 13} fill={alarm ? C.red : color} opacity={alarm ? 0.3 : 0.18} />
            <Icon name={icon} color={alarm ? "#ff8a94" : color} s={big ? 1 : 0.75} />
          </g>

          <text x={textX} y={big ? 7 : sub ? -3 : 5} className={big ? "arch-label" : "arch-label arch-label-sm"} fill={color}>
            {label}
          </text>
          {sub && <text x={textX} y="14" className="arch-sub" fill={color}>{sub}</text>}

          {alarm && (
            <rect x={-w / 2 - 5} y={-h / 2 - 5} width={w + 10} height={h + 10} rx="20" fill="none"
                  stroke={C.red} strokeWidth="2" className="arch-alarm" />
          )}
        </g>
      </g>
    </g>
  );
}

// Tiny line icons, 24x24 centered on 0,0.
function Icon({ name, color, s = 1 }) {
  const p = { fill: "none", stroke: color, strokeWidth: 2, strokeLinecap: "round", strokeLinejoin: "round" };
  const paths = {
    wave: <path d="M-10 0 L-6 0 L-3 -7 L1 7 L4 -3 L6 0 L10 0" {...p} />,
    shield: <><path d="M0 -10 L8 -6.5 V0 C8 5 4.5 8.5 0 10 C-4.5 8.5 -8 5 -8 0 V-6.5 Z" {...p} /><path d="M-3.5 0 L-1 2.5 L4 -2.5" {...p} /></>,
    forward: <><path d="M-8 0 H7" {...p} /><path d="M2 -5 L7 0 L2 5" {...p} /></>,
    bell: <><path d="M-6 3 V-2 C-6 -6 -3 -8 0 -8 C3 -8 6 -6 6 -2 V3 L8 5 H-8 Z" {...p} /><path d="M-2 8 H2" {...p} /></>,
    lock: <><rect x="-7" y="-2" width="14" height="10" rx="2" {...p} /><path d="M-4 -2 V-5 C-4 -9 4 -9 4 -5 V-2" {...p} /></>,
    db: <><ellipse cx="0" cy="-6" rx="7" ry="3" {...p} /><path d="M-7 -6 V6 C-7 10 7 10 7 6 V-6" {...p} /><path d="M-7 0 C-7 4 7 4 7 0" {...p} /></>,
    target: <><circle r="8" {...p} /><circle r="3.5" {...p} /><path d="M0 -12 V-9 M0 9 V12 M-12 0 H-9 M9 0 H12" {...p} /></>,
    check: <><circle r="8.5" {...p} /><path d="M-4 0 L-1 3 L4.5 -3" {...p} /></>,
    split: <><path d="M-8 0 H-2 L4 -6 H8 M-2 0 L4 6 H8" {...p} /></>,
    gate: <><path d="M-8 8 V-5 M8 8 V-5 M-10 -5 H10" {...p} /><path d="M-4 -1 L0 3 L4 -1" {...p} /></>,
    rules: <><path d="M-8 -6 H8 M-8 0 H8 M-8 6 H2" {...p} /></>,
    cnn: <path d="M-9 7 V1 M-4.5 7 V-4 M0 7 V-1 M4.5 7 V-8 M9 7 V-3" {...p} />,
    chat: <><path d="M-9 -7 H9 V4 H-1 L-5 8 V4 H-9 Z" {...p} /><path d="M-4 -2 H4" {...p} /></>,
  };
  return <g transform={`scale(${s})`}>{paths[name]}</g>;
}
