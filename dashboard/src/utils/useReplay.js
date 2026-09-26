import { useCallback, useEffect, useMemo, useRef, useState } from "react";

// Speed value for "one window per real second" -- the rate windows actually
// arrive from the car (every window is exactly 1 s; the Defender refuses any
// other length). The numeric speeds are multiples of the demo interval.
export const REAL_TIME = "realtime";

/**
 * Steps through a fixed array of DefenderOutput-shaped records over time,
 * simulating a live replay for the demo. Works the same whether `run` came
 * from the mock generator or (later) a real feed already collected into an
 * array.
 *
 * autoPlay: start playing as soon as there is something to play.
 * loop: wrap back to the first window at the end instead of stopping.
 */
export function useReplay(run, { intervalMs = 350, realTimeMs = 1000, autoPlay = false, loop = false } = {}) {
  const [index, setIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(autoPlay);
  const [speed, setSpeed] = useState(1);
  const timerRef = useRef(null);

  const total = run.length;
  const clampedIndex = Math.min(index, Math.max(total - 1, 0));
  const tickMs = speed === REAL_TIME ? realTimeMs : intervalMs / speed;

  useEffect(() => {
    if (!isPlaying || total === 0) return undefined;
    timerRef.current = setInterval(() => {
      setIndex((i) => {
        if (i + 1 >= total) {
          if (loop) return 0;
          setIsPlaying(false);
          return i;
        }
        return i + 1;
      });
    }, tickMs);
    return () => clearInterval(timerRef.current);
  }, [isPlaying, tickMs, total, loop]);

  const play = useCallback(() => {
    if (clampedIndex + 1 >= total) setIndex(0);
    setIsPlaying(true);
  }, [clampedIndex, total]);
  const pause = useCallback(() => setIsPlaying(false), []);
  const reset = useCallback(() => {
    setIsPlaying(false);
    setIndex(0);
  }, []);

  const seek = useCallback(
    (i) => setIndex(Math.min(Math.max(Math.round(i), 0), Math.max(total - 1, 0))),
    [total]
  );

  const history = useMemo(() => run.slice(0, clampedIndex + 1), [run, clampedIndex]);
  const current = history[history.length - 1] ?? null;

  return {
    current,
    history,
    index: clampedIndex,
    total,
    isPlaying,
    speed,
    setSpeed,
    play,
    pause,
    reset,
    seek,
  };
}
