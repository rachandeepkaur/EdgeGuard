"""
DefenderLLM (Part 2 extension, comparison + explanation layer -- added
2026-09-25): the third score in the Defender Server architecture (RuleDB=v2
percentile-rank fusion, TimingCNN, DefenderLLM -> weighted sum), per the
hand-drawn Red Team/Blue Team design.

CONSTRAINT DISCOVERED BEFORE BUILDING THIS: ROAD's DBC
(signal_extractions/DBC/anonymized.dbc; see part1/decode.py's own
docstring) is fully anonymized -- every signal is named "Unknown_N" with no
real-world meaning (no "speed", "brake", "reverse_light"). This is a
dataset limitation, not an architecture flaw: a real production DBC (an
OEM's own fleet) DOES have real signal names, and DefenderLLM is designed
for that case -- see signal_names below, which is the ONLY thing that
changes between "demo on ROAD" and "deployed on a real labeled fleet." On
an anonymized DBC, an LLM can still only reason over anonymous numbers, the
same six raw signals defender.ml_baseline already uses (plus, optionally,
individually-decoded field values -- see decoded_fields below, still
unlabeled here). Given ml_baseline's own honest negative result (logistic
regression over the six numbers did NOT beat v2), a zero-shot LLM scoring
the same numbers is not expected to beat v2 on THIS dataset either -- LLMs
are not naturally strong at implicit numeric-threshold reasoning zero-shot.
Built and evaluated with that expectation stated up front -- see
defender/crossval_defender_llm.py for the real dev-CV numbers, never
assumed.

The value an LLM genuinely adds here regardless of dataset, is a
natural-language EXPLANATION alongside its score: DefenderOutput.evidence
is model-generated prose, not just "top contributor: unknown_ids" -- useful
for a SOC-analyst/demo framing independent of whether the score itself
raises recall.

SIGNAL-NAME HOOK (production readiness, demonstrated but not exercisable on
ROAD): DefenderLLMDefender optionally takes a `decoder` (part1.decode.Decoder,
already built and already correct -- it decodes real bit-precise signal
values, just under ROAD's anonymized names) and a `signal_names` mapping
("0D0:Unknown_4" -> "brake_pressure", say). When both are supplied, each
decoded field's most recent value is added to the prompt under its real
name, so the model reasons over actual signal semantics. On ROAD,
signal_names is empty (there is nothing to map to), so the same code
degrades gracefully to numbered fields -- feeding a real fleet's DBC names
here needs no other code change anywhere in this module.

Local, ungated, CPU-friendly instruct model by default
(Qwen/Qwen2.5-0.5B-Instruct) via `transformers`. NOT part of the shipped
detector -- comparison/demo only, same status as defender.ml_baseline and
defender.timing_cnn. Nothing in defender/defender.py, run_demo.py or
run_training.py imports this module.
"""

import json
import re
import time
from typing import Callable, Dict, List, Optional, Tuple

from defender.ml_baseline import FEATURES, extract_features
from defender.stage1 import Stage1Model
from defender.stage2 import Stage2Model
from defender.threshold import choose_threshold, make_decision
from part1.decode import Decoder
from shared.schemas import DefenderOutput, TrafficWindow

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

_PROMPT_TEMPLATE = """You are a CAN-bus intrusion detector for a vehicle's gateway ECU. You are given six anomaly-check values computed for a 1-second window of bus traffic. Each is a raw count or magnitude, not yet normalized -- larger means more unusual.

unknown_ids (count of CAN IDs never seen in normal training): {unknown_ids}
rate_excess (how far the message rate exceeds the normal rate): {rate_excess}
short_gaps (count of inter-frame gaps shorter than ever seen normal): {short_gaps}
out_of_range (how far a payload field's value is outside its normal range): {out_of_range}
frozen_break (how surprising it is that a normally-varying field is frozen this window): {frozen_break}
large_jump (how far a payload field jumped between consecutive frames, beyond normal): {large_jump}
{decoded_section}
Judge whether this window is an ATTACK or NORMAL. Answer with ONLY a JSON object: {{"attack_probability": <float 0 to 1>, "reason": "<one short sentence>"}}

JSON:"""


def _decoded_section(decoded_fields: Optional[Dict[str, float]],
                     signal_names: Optional[Dict[str, str]]) -> str:
    """Renders the optional per-signal block. Empty decoded_fields (ROAD's
    default -- see module docstring) renders nothing, so the prompt is
    byte-for-byte the six-number prompt when no decoder is wired in."""
    if not decoded_fields:
        return ""
    signal_names = signal_names or {}
    lines = [f"  {signal_names.get(field, field)}: {round(value, 2)}"
            for field, value in sorted(decoded_fields.items())]
    return (
        "\nIndividual signal values in this window (most recent reading each; "
        "a labeled production DBC gives these their real names, e.g. "
        "brake_pressure -- any name below still starting with 'Unknown_' is "
        "exactly what ROAD's anonymized public DBC calls it, unrelated to how "
        "well this signal can be reasoned about):\n" + "\n".join(lines) + "\n"
    )


def build_prompt(features: List[float], decoded_fields: Optional[Dict[str, float]] = None,
                 signal_names: Optional[Dict[str, str]] = None) -> str:
    """The same six raw signals defender.ml_baseline uses (see that module's
    FEATURES/extract_features), plus an optional named-signal section (see
    _decoded_section). Isolates "can an LLM reason over these numbers
    better than a percentile-rank fusion" as the only new variable when
    decoded_fields is omitted -- same spirit as ml_baseline isolating
    "supervised learning vs. percentile-rank scoring"."""
    values = dict(zip(FEATURES, (round(f, 4) for f in features)))
    return _PROMPT_TEMPLATE.format(
        decoded_section=_decoded_section(decoded_fields, signal_names), **values)


def latest_decoded_values(decoder: Decoder, window: TrafficWindow) -> Dict[str, float]:
    """decoder.decode_frames() returns one value per FRAME of a field's CAN
    ID; a prompt needs one number per field, so this keeps only the most
    recent (last) reading of each -- a snapshot, not the whole window's
    history. NaN readings (payload too short for that field -- see
    part1/decode.py) are dropped rather than shown as a bogus number."""
    decoded = decoder.decode_frames(window.frames)
    out = {}
    for field, values in decoded.items():
        for value in reversed(values):
            if value == value:  # skips NaN without importing math
                out[field] = value
                break
    return out


_JSON_OBJECT = re.compile(r"\{.*?\}", re.DOTALL)
_FLOAT_IN_TEXT = re.compile(r"(\d*\.\d+|\d+)")


def parse_response(text: str) -> Tuple[float, str]:
    """Best-effort parse of the model's completion into (probability,
    reason). Never raises: a model that ignores the JSON instruction still
    yields SOME probability estimate (fallback: the first number-looking
    token in its text, else 0.5) rather than crashing a whole crossval/
    final_test run on one bad generation."""
    match = _JSON_OBJECT.search(text)
    if match:
        try:
            record = json.loads(match.group(0))
            prob = float(record.get("attack_probability", 0.5))
            reason = str(record.get("reason", "")).strip() or "(no reason given)"
            return max(0.0, min(1.0, prob)), reason
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    number = _FLOAT_IN_TEXT.search(text)
    prob = float(number.group(1)) if number else 0.5
    if not (0.0 <= prob <= 1.0):
        prob = 0.5
    fallback_reason = text.strip().replace("\n", " ")[:200] or "(unparseable model output)"
    return prob, fallback_reason


GenerateFn = Callable[[str], str]


class DefenderLLMModel:
    """Wraps a local instruct model as a plain (prompt -> completion)
    function. generate_fn is injectable, so this class -- and everything
    built on it (DefenderLLMDefender, defender.crossval_defender_llm) -- is
    fully testable with NO transformers/torch install, by passing a fake
    generate_fn. load_local() is the ONLY place that imports transformers,
    and only runs when a real model is actually requested (smoke test /
    crossval / final eval), never from the unit tests."""

    def __init__(self, generate_fn: GenerateFn, model_name: str = DEFAULT_MODEL_NAME):
        self.generate_fn = generate_fn
        self.model_name = model_name
        self.threshold = 1.0
        self.calibration_windows = 0

    @classmethod
    def load_local(cls, model_name: str = DEFAULT_MODEL_NAME,
                   max_new_tokens: int = 60) -> "DefenderLLMModel":
        """Loads a real local Hugging Face causal LM and wraps it as
        generate_fn. Needs `transformers` + `torch` installed -- run this
        on the machine that actually has them (this project's Mac Terminal
        or the Nano), never assumed available here."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        model.eval()

        def generate_fn(prompt: str) -> str:
            chat = [{"role": "user", "content": prompt}]
            inputs = tokenizer.apply_chat_template(
                chat, add_generation_prompt=True, return_tensors="pt")
            with torch.no_grad():
                output = model.generate(
                    inputs, max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id)
            return tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True)

        return cls(generate_fn, model_name)

    def score_one(self, features: List[float], decoded_fields: Optional[Dict[str, float]] = None,
                 signal_names: Optional[Dict[str, str]] = None) -> Tuple[float, str]:
        return parse_response(self.generate_fn(build_prompt(features, decoded_fields, signal_names)))

    def predict_proba_attack(self, feature_rows: List[List[float]]) -> List[float]:
        return [self.score_one(row)[0] for row in feature_rows]

    def calibrate_threshold(self, normal_feature_rows: List[List[float]],
                            max_false_alarm_rate: float) -> float:
        """Same discipline as v1/v2/ml_baseline/TimingCNN: calibrated on
        NORMAL validation scores only, never on attack data."""
        scores = self.predict_proba_attack(normal_feature_rows)
        self.calibration_windows = len(normal_feature_rows)
        self.threshold = choose_threshold(scores, max_false_alarm_rate)
        return self.threshold


class DefenderLLMDefender:
    """Same score_window(window) -> DefenderOutput interface as
    defender.defender.Defender, defender.ml_baseline.MLBaselineDefender and
    defender.timing_cnn.TimingCNNDefender, so it drops into the exact same
    evaluation machinery unchanged. stage1/stage2 are v2's ALREADY-FROZEN
    models, borrowed purely as ml_baseline's feature extractor -- never
    refit. decoder/signal_names are optional (see module docstring's
    SIGNAL-NAME HOOK) -- omit both to reproduce the plain six-number
    prompt, exactly what defender.crossval_defender_llm evaluates on ROAD."""

    def __init__(self, stage1: Stage1Model, stage2: Stage2Model,
                model: DefenderLLMModel, decoder: Optional[Decoder] = None,
                signal_names: Optional[Dict[str, str]] = None,
                model_version: str = "defender_llm_v1"):
        if not (isinstance(stage1, Stage1Model) and stage1.fitted):
            raise ValueError("DefenderLLMDefender needs a fitted Stage1Model")
        if not (isinstance(stage2, Stage2Model) and stage2.fitted):
            raise ValueError("DefenderLLMDefender needs a fitted Stage2Model")
        self.stage1 = stage1
        self.stage2 = stage2
        self.model = model
        self.decoder = decoder
        self.signal_names = signal_names or {}
        self.model_version = model_version

    def score_window(self, window: TrafficWindow) -> DefenderOutput:
        start = time.perf_counter()
        features = extract_features(self.stage1, self.stage2, window)
        decoded_fields = latest_decoded_values(self.decoder, window) if self.decoder else None
        score, reason = self.model.score_one(features, decoded_fields, self.signal_names)
        latency_ms = (time.perf_counter() - start) * 1000
        decision = make_decision(score, self.model.threshold)
        return DefenderOutput(
            window_id=window.window_id, attack_score=score,
            threshold=self.model.threshold, decision=decision,
            evidence=f"DefenderLLM ({self.model.model_name}): {reason}",
            model_version=self.model_version, latency_ms=latency_ms,
        )
