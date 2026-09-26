// Shared metadata for the three actions defender/simulated_consumer.py can
// produce, so the detail panel and the architecture diagram render them
// identically instead of duplicating the mapping.
export const ACTIONS = {
  SIMULATED_ALERT: { label: "Simulated alert", tone: "status-warning", nodeTone: "node-warning" },
  SIMULATED_ISOLATION: { label: "Simulated isolation", tone: "status-critical", nodeTone: "node-critical" },
  SIMULATED_FORWARD: { label: "Simulated forward", tone: "status-good", nodeTone: "node-good" },
};
