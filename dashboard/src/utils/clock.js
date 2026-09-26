import { useEffect, useState } from "react";

// Every window is exactly 1 s of driving (the Defender refuses any other
// length), so a window's position in its capture IS its drive time.
const WINDOW_S = 1;

/** Wall-clock time, re-rendered once per second. */
export function useClock() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);
  return now;
}

/**
 * Seconds of driving into the current capture at the END of this window,
 * read from its id ("cap34_w0019" -> 20 s). Falls back to the replay
 * position when the id has no window number.
 */
export function driveSeconds(windowId, fallbackIndex) {
  const m = /_w(\d+)$/.exec(windowId ?? "");
  const n = m ? Number(m[1]) : fallbackIndex;
  return (n + 1) * WINDOW_S;
}

export function formatHms(totalSeconds) {
  const s = Math.max(0, Math.floor(totalSeconds));
  const pad = (v) => String(v).padStart(2, "0");
  return `${pad(Math.floor(s / 3600))}:${pad(Math.floor((s % 3600) / 60))}:${pad(s % 60)}`;
}
