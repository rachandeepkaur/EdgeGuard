// Formats DefenderOutput.latency_ms (per-window inference time, measured by
// the Defender itself) for display. Returns null when the field is missing,
// so callers can hide the readout instead of showing a made-up number.
export function formatLatency(ms) {
  return typeof ms === "number" && Number.isFinite(ms) ? `${ms.toFixed(1)} ms` : null;
}
