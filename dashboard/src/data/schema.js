// Mirrors shared/schemas.py DefenderOutput from the EdgeGuard repo.
// Keep this the single source of truth for the shape the UI expects, so
// swapping the mock feed for a real one (see liveFeed.js) never touches
// the components below it.
//
// DefenderOutput fields (Python -> JS, same names):
//   window_id      string
//   attack_score   number, 0..1
//   threshold      number, 0..1
//   decision       "ATTACK" | "ACCEPT"
//   evidence       string, free text set by whichever stage produced the score
//   model_version  string, e.g. "v1" | "v2"
//   latency_ms     number | null, optional

export const DECISIONS = Object.freeze({
  ATTACK: "ATTACK",
  ACCEPT: "ACCEPT",
});
