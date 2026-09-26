import { useEffect, useRef, useState } from "react";

// Same two cubic-bezier segments as the road path drawn in CarsView --
// keep in sync with the `d` attribute there if that road shape ever changes.
const SEGMENTS = [
  { p0: [0, 190], p1: [200, 130], p2: [380, 230], p3: [480, 170] },
  { p0: [480, 170], p1: [620, 90], p2: [760, 190], p3: [900, 140] },
];

// The car's silhouette is ~86px wide from its own center. Keeping travel
// inset by this much (in t-space) keeps its edges inside the viewBox at
// both ends of the road instead of poking out of frame.
// Computed (not guessed) from the bezier control points above: the t values
// where the car's x-position is exactly at the safe edge (86px inset, the
// car's half-width, on a 900-wide viewBox), plus a small safety margin.
// Verified against the rendered output before shipping -- an earlier
// eyeballed guess (0.06/0.94) still clipped the car by ~15px on both ends.
const MIN_T = 0.09;
const MAX_T = 0.88;
const FADE_SECONDS = 0.35;

function cubicPoint(seg, t) {
  const { p0, p1, p2, p3 } = seg;
  const mt = 1 - t;
  const x = mt * mt * mt * p0[0] + 3 * mt * mt * t * p1[0] + 3 * mt * t * t * p2[0] + t * t * t * p3[0];
  const y = mt * mt * mt * p0[1] + 3 * mt * mt * t * p1[1] + 3 * mt * t * t * p2[1] + t * t * t * p3[1];
  const dx = 3 * mt * mt * (p1[0] - p0[0]) + 6 * mt * t * (p2[0] - p1[0]) + 3 * t * t * (p3[0] - p2[0]);
  const dy = 3 * mt * mt * (p1[1] - p0[1]) + 6 * mt * t * (p2[1] - p1[1]) + 3 * t * t * (p3[1] - p2[1]);
  return { x, y, angle: Math.atan2(dy, dx) * (180 / Math.PI) };
}

/** Position + heading for progress `t` in [0, 1] across both road segments. */
export function pointOnRoad(t) {
  const clamped = Math.min(Math.max(t, 0), 1);
  const segIndex = clamped < 0.5 ? 0 : 1;
  const localT = segIndex === 0 ? clamped * 2 : (clamped - 0.5) * 2;
  return cubicPoint(SEGMENTS[segIndex], localT);
}

/**
 * Drives the car one-way along the road (never reverses). On reaching the
 * end it fades out, jumps back to the start, and fades back in -- a cheap
 * way to get a seamless-looking loop on an open (non-closed-track) road
 * without a visible teleport snap.
 *
 * Freezes everything (position AND any in-progress fade) for `pauseMs` the
 * moment `alert` goes from false to true -- a deliberate "braking" beat,
 * independent of how long the underlying attack window(s) last in the data.
 *
 * `running: false` (the replay is paused) holds the car where it is.
 */
export function useCarDrive(alert, { speed = 0.05, pauseMs = 3000, running = true } = {}) {
  const [t, setT] = useState(MIN_T);
  const [opacity, setOpacity] = useState(1);
  const phaseRef = useRef("driving"); // 'driving' | 'fading-out' | 'fading-in'
  const fadeElapsedRef = useRef(0);
  const lastFrameRef = useRef(null);
  const pausedUntilRef = useRef(0);
  const prevAlertRef = useRef(false);
  const rafRef = useRef(null);
  // Read inside the animation loop, so pausing takes effect without
  // restarting it.
  const runningRef = useRef(running);
  useEffect(() => {
    runningRef.current = running;
  }, [running]);

  useEffect(() => {
    if (alert && !prevAlertRef.current) {
      pausedUntilRef.current = performance.now() + pauseMs;
      // If an alert happens to start mid-fade (the car is between the two
      // ends of the road, briefly fading out/in for the loop), the freeze
      // below would otherwise lock in whatever opacity it had at that
      // instant -- which can be near 0, making the car invisible for the
      // whole 3s alert. Force it fully visible and out of the fade state
      // before freezing, so the "braking" moment is always visible.
      phaseRef.current = "driving";
      setOpacity(1);
    }
    prevAlertRef.current = alert;
  }, [alert, pauseMs]);

  useEffect(() => {
    function frame(now) {
      if (lastFrameRef.current === null) lastFrameRef.current = now;
      const dt = (now - lastFrameRef.current) / 1000;
      lastFrameRef.current = now;

      if (now >= pausedUntilRef.current) {
        // Paused replay: the car holds its position. A loop fade already in
        // progress still finishes, so it never freezes half-transparent.
        if (phaseRef.current === "driving" && runningRef.current) {
          setT((prev) => {
            const next = prev + speed * dt;
            if (next >= MAX_T) {
              phaseRef.current = "fading-out";
              fadeElapsedRef.current = 0;
              return MAX_T;
            }
            return next;
          });
        } else if (phaseRef.current === "fading-out") {
          fadeElapsedRef.current += dt;
          const progress = Math.min(fadeElapsedRef.current / FADE_SECONDS, 1);
          setOpacity(1 - progress);
          if (progress >= 1) {
            setT(MIN_T);
            phaseRef.current = "fading-in";
            fadeElapsedRef.current = 0;
          }
        } else if (phaseRef.current === "fading-in") {
          fadeElapsedRef.current += dt;
          const progress = Math.min(fadeElapsedRef.current / FADE_SECONDS, 1);
          setOpacity(progress);
          if (progress >= 1) {
            phaseRef.current = "driving";
          }
        }
      }
      rafRef.current = requestAnimationFrame(frame);
    }
    rafRef.current = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(rafRef.current);
  }, [speed]);

  return { ...pointOnRoad(t), t, opacity };
}
