import { useEffect, useRef, useState } from "react";

function prefersReducedMotion() {
  return typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
}

/**
 * Glides the displayed value toward `target` instead of jumping. If the
 * target changes mid-animation it retargets from wherever it currently is,
 * so fast playback just looks like a smooth count. Display only -- callers
 * format the result (round it for counts). Null passes straight through.
 */
export function useAnimatedNumber(target, durationMs = 350) {
  const [value, setValue] = useState(target);
  const valueRef = useRef(target);
  const rafRef = useRef(null);

  useEffect(() => {
    if (target == null || valueRef.current == null || prefersReducedMotion()) {
      valueRef.current = target;
      setValue(target);
      return undefined;
    }
    const from = valueRef.current;
    const start = performance.now();
    const step = (now) => {
      const t = Math.min((now - start) / durationMs, 1);
      const eased = 1 - (1 - t) ** 3; // ease-out cubic
      valueRef.current = from + (target - from) * eased;
      setValue(valueRef.current);
      if (t < 1) rafRef.current = requestAnimationFrame(step);
    };
    rafRef.current = requestAnimationFrame(step);
    return () => cancelAnimationFrame(rafRef.current);
  }, [target, durationMs]);

  return value;
}
