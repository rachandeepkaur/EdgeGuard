// Best-effort parser for DefenderOutput.evidence.
//
// `evidence` is free text (see defender/fusion.py, stage1.py, stage2.py in
// the EdgeGuard repo) -- there is no dedicated "affected CAN ID" field.
// Fusion prefixes the string with "[Stage 1] " or "[Stage 2] ". Most
// sub-checks then say "ID <hex>", but Stage 1's unknown_ids check instead
// says "... e.g. <hex> (<n> frames)" with no "ID" prefix. This parser
// handles both and returns null (never a guess) when it can't find one.
//
// If the team standardizes the evidence format later, this is the one
// place to update.

const STAGE_RE = /^\[(Stage 1|Stage 2)\]\s*/i;
const ID_PREFIXED_RE = /\bID\s+([0-9A-Fa-f]{2,3})\b/;
const EG_HEX_RE = /\be\.g\.\s+([0-9A-Fa-f]{2,3})\b/;

export function parseEvidence(evidence) {
  if (!evidence || typeof evidence !== "string") {
    return { stage: null, canId: null, detail: "" };
  }

  const stageMatch = evidence.match(STAGE_RE);
  const stage = stageMatch ? stageMatch[1] : null;
  const detail = stageMatch ? evidence.slice(stageMatch[0].length) : evidence;

  const idMatch = detail.match(ID_PREFIXED_RE) || detail.match(EG_HEX_RE);
  const canId = idMatch ? idMatch[1].toUpperCase() : null;

  return { stage, canId, detail };
}
