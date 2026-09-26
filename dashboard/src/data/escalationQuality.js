// Real numbers from results/escalation_quality_v2/demo_test.json (real ROAD
// data, v2, part1/escalation_quality.py). Not recomputed in the browser --
// copied from that file so the dashboard cites the same evidence the README
// and CLAUDE.md do. Regenerate that JSON (see CLAUDE.md's Escalation
// quality section) and update this file by hand if the model changes.
export const ESCALATION_QUALITY = {
  windows: 2511,
  escalations: 898,
  escalationRate: 0.35762644364794904,
  escalationPrecision: 0.48106904231625836,
  escalationRecall: 0.2801556420233463,
  localFalsePositives: 249,
  escalatedFalsePositives: 249,
  localFalseNegatives: 1293,
  escalatedFalseNegatives: 183,
  bundleBytesMean: 241.28841870824053,
  bundleFields: ["attack_score", "decision", "evidence", "model_version", "threshold", "window_id"],
  source: "results/escalation_quality_v2/demo_test.json -- real ROAD Red Team test path, no cloud needed",
};
