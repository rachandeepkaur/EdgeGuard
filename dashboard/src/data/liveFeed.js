// Real feed: a full replay of real EdgeGuard v2 Defender output, scored
// against real ROAD development captures (see
// dashboard/scripts/build_real_run.py, which writes realRun.json). This is
// REAL inference output -- real attack_score, real evidence text, real
// per-window latency -- not the mock generator in mockRun.js.
//
// It's a REPLAY of a precomputed run, not a socket to a running Nano: there
// is no live streaming inference pipeline anywhere in this repo yet (Part
// 2's Defender and Part 1's real windows exist, but nothing keeps a Nano
// process running and pushing scores out). Wiring an actual live socket is
// still open; this is the honest version of "real feed" available today,
// and it's a drop-in replacement for buildMockRun() -- same shape, same
// useReplay() hook, no component below App.jsx changes.
//
// To regenerate realRun.json against a fresh model or different captures:
//   python -m dashboard.scripts.build_real_run --data-dir ~/Downloads/road

import realRunData from "./realRun.json";

export async function connectLiveFeed() {
  const { run, meta } = realRunData;
  if (!Array.isArray(run) || run.length === 0) {
    throw new Error(
      "realRun.json has no windows -- regenerate it with " +
      "'python -m dashboard.scripts.build_real_run --data-dir <road>' from the repo root."
    );
  }
  return {
    run,
    meta: {
      captureId: meta.captureId,
      modelVersion: meta.modelVersion,
      threshold: meta.threshold,
      escalationBandHalfWidth: meta.escalationBandHalfWidth ?? null,
    },
  };
}
