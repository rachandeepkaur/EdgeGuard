// Mirrors part1/escalation_policy.py's should_escalate() in the browser, so
// the dashboard can show -- per replayed window -- whether the REAL
// calibrated band would send it for a cloud second opinion. This is
// display logic only; the actual escalation decision in a real run is made
// by the Python path (integration/run_demo.py --escalate), never here.
export function isInEscalationBand(output, bandHalfWidth) {
  if (!output || bandHalfWidth == null) return false;
  const low = Math.max(0, output.threshold - bandHalfWidth);
  const high = Math.min(1, output.threshold + bandHalfWidth);
  return output.attack_score >= low && output.attack_score <= high;
}
