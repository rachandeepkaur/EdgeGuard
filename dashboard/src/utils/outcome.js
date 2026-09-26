// Compares a replayed window's Defender decision with its evaluator-side
// `truth` field ("attack" / "normal"), written by build_real_run.py AFTER
// scoring -- the Defender never sees it. Used only to mark mistakes; it never
// changes a decision. Returns null when the window is correct or carries no
// truth (mock data, uploaded captures).
export function outcomeOf(output) {
  if (!output?.truth) return null;
  if (output.truth === "attack" && output.decision === "ACCEPT") return "missed";
  if (output.truth === "normal" && output.decision === "ATTACK") return "false_alarm";
  return null;
}

export const OUTCOME_TEXT = {
  missed: {
    badge: "Missed attack",
    caption: "truth: attack, fusion missed it",
    tooltip: "Ground truth says this window holds injected frames, but the Defender accepted it (evaluator view; the Defender never sees the truth).",
  },
  false_alarm: {
    badge: "False alarm",
    caption: "truth: normal traffic, fusion false alarm",
    tooltip: "Ground truth says this window is normal traffic, but the Defender flagged it (evaluator view; the Defender never sees the truth).",
  },
};
