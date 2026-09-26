import { useEffect, useRef, useState } from "react";
import { parseEvidence } from "./parseEvidence";
import { outcomeOf } from "./outcome";

const MAX_EVENTS = 200;

// "Payload anomaly (large_jump): ..." -> "large_jump". Null for windows whose
// evidence names no triggered check (normal traffic).
const CHECK_RE = /\(([a-z_]+)\)/;

const ACTION_WORD = {
  SIMULATED_ALERT: "alert",
  SIMULATED_ISOLATION: "isolate",
  SIMULATED_FORWARD: "forward",
};

/**
 * Append-only console log of replayed windows, newest first. Each entry is
 * captured the moment its window is shown (wall-clock time, and the attack
 * action in force then), so changing the action later doesn't rewrite
 * history -- like a real security console. Everything else is read from the
 * Defender output itself. Cleared when `resetKey` changes (a different run).
 */
export function useEventLog(current, attackAction, resetKey) {
  const [events, setEvents] = useState([]);
  const seqRef = useRef(0);
  const lastIdRef = useRef(null);
  const keyRef = useRef(resetKey);

  useEffect(() => {
    // A different run starts a fresh log.
    const fresh = keyRef.current !== resetKey;
    if (fresh) {
      keyRef.current = resetKey;
      lastIdRef.current = null;
    }
    // Skip repeats of the same window (e.g. React re-running the effect).
    if (!current || current.window_id === lastIdRef.current) return;
    lastIdRef.current = current.window_id;
    const { stage, canId, detail } = parseEvidence(current.evidence);
    const isAttack = current.decision === "ATTACK";
    const entry = {
      seq: ++seqRef.current,
      time: new Date(),
      windowId: current.window_id,
      decision: current.decision,
      canId,
      stage,
      check: CHECK_RE.exec(detail)?.[1] ?? null,
      latencyMs: current.latency_ms,
      action: ACTION_WORD[isAttack ? attackAction : "SIMULATED_FORWARD"],
      outcome: outcomeOf(current),
      evidence: current.evidence,
    };
    setEvents((prev) => [entry, ...(fresh ? [] : prev)].slice(0, MAX_EVENTS));
  }, [current, attackAction, resetKey]);

  return events;
}
