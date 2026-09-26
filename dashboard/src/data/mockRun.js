// Generates a mock "replay" of DefenderOutput records -- a fixed sequence of
// windows from one capture, with a couple of attack bursts of consecutive
// windows (that's how a real injected attack looks: it spans several
// 1-second windows in a row, not one isolated blip). Shaped exactly like
// what defender/defender.py's score_window() returns, so this array can be
// swapped for a real feed (see liveFeed.js) without touching any component.
//
// Nothing here is a saved benchmark result -- it's UI-only mock data, same
// as the example the team shared: window_id "cap19_w00017" etc.

import { DECISIONS } from "./schema";

const THRESHOLD = 0.9642248722316866; // matches the team's example mock exactly
const CAPTURE_ID = "cap19";
const MODEL_VERSION = "v2";

const STAGE1_EVIDENCE = [
  (id) => `ID ${id}: 4.2x normal frame rate`,
  (id) => `ID ${id}: shortest gap 1.80 ms, normal ~9.40 ms`,
  (id) => `1 unknown CAN ID(s), e.g. ${id} (6 frames)`,
];

const STAGE2_EVIDENCE = [
  (id) => `Payload anomaly (large_jump): ID ${id}`,
  (id) => `Payload anomaly (out_of_range): ID ${id}`,
  (id) => `Payload anomaly (frozen): ID ${id}`,
];

const WATCHED_IDS = ["0D0", "6E0"];

function normalScore(rand) {
  // Normal validation windows cluster well under threshold; occasional
  // small bumps, matching the dev-check false-alarm rate being nonzero.
  return Math.min(0.35 + rand() * 0.35, 0.92);
}

function attackScore(rand) {
  return 0.97 + rand() * 0.03;
}

function mulberry32(seed) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeOutput(index, score, decision, evidenceFn, canId) {
  const evidence = decision === DECISIONS.ATTACK
    ? `[Stage 2] ${evidenceFn(canId)}`
    : `[Stage 1] ID ${canId}: 1.0x normal frame rate`;
  return {
    window_id: `${CAPTURE_ID}_w${String(index).padStart(5, "0")}`,
    attack_score: Number(score.toFixed(4)),
    threshold: THRESHOLD,
    decision,
    evidence,
    model_version: MODEL_VERSION,
    latency_ms: Number((8 + Math.random() * 12).toFixed(2)),
  };
}

/**
 * Build one deterministic mock run (same sequence every reload, so a demo
 * rehearsal matches the real recording).
 *
 * @param {number} length total windows in the run
 * @param {number} seed
 */
export function buildMockRun(length = 240, seed = 7) {
  const rand = mulberry32(seed);
  const run = [];

  // A couple of multi-window attack bursts, spaced out through the run.
  const bursts = [
    { start: Math.floor(length * 0.28), len: 5, stage: STAGE2_EVIDENCE[0] },
    { start: Math.floor(length * 0.55), len: 3, stage: STAGE1_EVIDENCE[1] },
    { start: Math.floor(length * 0.82), len: 6, stage: STAGE2_EVIDENCE[2] },
  ];

  for (let i = 0; i < length; i++) {
    const burst = bursts.find((b) => i >= b.start && i < b.start + b.len);
    if (burst) {
      const canId = WATCHED_IDS[i % WATCHED_IDS.length];
      run.push(makeOutput(i, attackScore(rand), DECISIONS.ATTACK, burst.stage, canId));
    } else {
      const canId = WATCHED_IDS[(i + 1) % WATCHED_IDS.length];
      run.push(makeOutput(i, normalScore(rand), DECISIONS.ACCEPT, STAGE1_EVIDENCE[0], canId));
    }
  }
  return run;
}

// Same real band used throughout the README/CLAUDE.md (v2, 30% budget,
// calibrated on real validation scores) -- reused here so mock mode
// exercises the same escalation display logic as the real feed.
const ESCALATION_BAND_HALF_WIDTH = 0.09502826737612236;

export const CAPTURE_META = {
  captureId: CAPTURE_ID,
  modelVersion: MODEL_VERSION,
  threshold: THRESHOLD,
  escalationBandHalfWidth: ESCALATION_BAND_HALF_WIDTH,
};
