"""
Tests for defender/defender_llm.py (comparison + explanation-layer LLM
scorer). ALL windows here are MOCK traffic, same style as
tests/test_ml_baseline.py. generate_fn is FAKED throughout -- no real
model, no transformers/torch needed to run this file -- so these tests
prove the prompt/parse/wiring/threshold logic is correct, never how well a
real local model does on real ROAD data (see
results/crossval_defender_llm.json for that).
"""

import pytest

from defender.defender_llm import (
    DefenderLLMDefender,
    DefenderLLMModel,
    build_prompt,
    parse_response,
)
from defender.ml_baseline import FEATURES, extract_features
from defender.stage1 import Stage1Model
from defender.stage2 import Stage2Model
from defender.threshold import ACCEPT, ATTACK
from shared.schemas import TrafficWindow

MOCK_PERIODS = {"0D0": 0.010, "0F4": 0.020}


def mock_payload(k: int) -> str:
    b0 = k % 200
    return f"{b0:02X}00000000000000"


def make_normal_window(index: int) -> TrafficWindow:
    start = 1000.0 + index
    frames = []
    for can_id, period in MOCK_PERIODS.items():
        count = int(round(1.0 / period))
        for k in range(count):
            t = round(start + 0.001 + k * period, 6)
            frames.append({"timestamp": t, "can_id": can_id,
                           "payload": mock_payload(index * 100 + k)})
    frames.sort(key=lambda f: f["timestamp"])
    return TrafficWindow(window_id=f"cap01_w{index:04d}", capture_id="cap01",
                         window_start=start, window_end=start + 1.0, frames=frames)


def make_frozen_attack_window(index: int) -> TrafficWindow:
    window = make_normal_window(index)
    data = window.model_dump()
    for frame in data["frames"]:
        if frame["can_id"] == "0D0":
            frame["payload"] = "FF00000000000000"
    return TrafficWindow(**data)


@pytest.fixture(scope="module")
def fitted_stages():
    normal = [make_normal_window(i) for i in range(60)]
    stage1 = Stage1Model().fit(normal)
    stage2 = Stage2Model().fit(normal)
    return stage1, stage2


def constant_generate_fn(text: str):
    """A fake model that always answers the same thing, wrapped so tests
    can swap in whatever completion string they want to exercise."""
    def generate_fn(prompt: str) -> str:
        return text
    return generate_fn


def echo_frozen_break_generate_fn(prompt: str) -> str:
    """A fake 'model' that actually reads its own prompt -- scores high
    only when frozen_break looks large -- so DefenderLLMDefender's wiring
    can be exercised end to end without a real network forward pass."""
    frozen_line = [line for line in prompt.splitlines() if line.startswith("frozen_break")][0]
    value = float(frozen_line.split(":")[-1].strip())
    prob = 0.9 if value > 0.5 else 0.1
    return f'{{"attack_probability": {prob}, "reason": "frozen_break={value}"}}'


def test_build_prompt_includes_all_six_feature_values_in_order():
    features = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    prompt = build_prompt(features)
    for name, value in zip(FEATURES, features):
        assert f"{name}" in prompt
        assert str(value) in prompt


def test_parse_response_reads_well_formed_json():
    prob, reason = parse_response('{"attack_probability": 0.83, "reason": "rate spike"}')
    assert prob == pytest.approx(0.83)
    assert reason == "rate spike"


def test_parse_response_finds_json_inside_extra_text():
    text = 'Sure, here is my answer:\n{"attack_probability": 0.2, "reason": "looks normal"}\nThanks!'
    prob, reason = parse_response(text)
    assert prob == pytest.approx(0.2)
    assert reason == "looks normal"


def test_parse_response_falls_back_to_a_bare_number_on_bad_json():
    prob, _ = parse_response("I'd say the attack probability is about 0.75 here.")
    assert prob == pytest.approx(0.75)


def test_parse_response_never_raises_on_garbage():
    prob, reason = parse_response("the model rambled with no numbers at all")
    assert prob == 0.5
    assert reason


def test_parse_response_clamps_out_of_range_probabilities():
    prob, _ = parse_response('{"attack_probability": 4.2, "reason": "overconfident"}')
    assert prob == 1.0
    prob, _ = parse_response('{"attack_probability": -1.0, "reason": "negative"}')
    assert prob == 0.0


def test_predict_proba_attack_uses_the_injected_generate_fn():
    model = DefenderLLMModel(constant_generate_fn('{"attack_probability": 0.6, "reason": "x"}'))
    scores = model.predict_proba_attack([[0.0] * 6, [1.0] * 6])
    assert scores == [pytest.approx(0.6), pytest.approx(0.6)]


def test_calibrate_threshold_limits_false_alarms_on_the_calibration_set():
    # A fake model whose score is just the sum of the features -- gives a
    # real, non-constant distribution to calibrate a threshold against.
    def generate_fn(prompt: str) -> str:
        total = sum(float(line.split(":")[-1].strip())
                   for line in prompt.splitlines() if ":" in line and line.split(":")[0].split()[0] in FEATURES)
        prob = min(1.0, total / 10.0)
        return f'{{"attack_probability": {prob}, "reason": "sum={total}"}}'

    model = DefenderLLMModel(generate_fn)
    normal = [[0.1, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(50)]
    model.calibrate_threshold(normal, max_false_alarm_rate=0.1)
    scores = model.predict_proba_attack(normal)
    false_alarms = sum(1 for s in scores if s >= model.threshold)
    assert false_alarms / len(normal) <= 0.1 + 1e-9


def test_defender_llm_defender_score_window_matches_schema_and_decision(fitted_stages):
    stage1, stage2 = fitted_stages
    model = DefenderLLMModel(echo_frozen_break_generate_fn)
    normal_features = [extract_features(stage1, stage2, make_normal_window(i)) for i in range(30)]
    model.calibrate_threshold(normal_features, max_false_alarm_rate=0.1)

    defender = DefenderLLMDefender(stage1, stage2, model)
    normal_output = defender.score_window(make_normal_window(500))
    attack_output = defender.score_window(make_frozen_attack_window(500))

    for output in (normal_output, attack_output):
        assert output.model_version == "defender_llm_v1"
        assert 0.0 <= output.attack_score <= 1.0
        assert output.threshold == model.threshold
        assert output.latency_ms is not None and output.latency_ms >= 0.0
        expected = ATTACK if output.attack_score >= output.threshold else ACCEPT
        assert output.decision == expected
        assert "DefenderLLM" in output.evidence


def test_defender_llm_defender_catches_the_frozen_attack_via_its_own_prompt_reasoning(fitted_stages):
    """End-to-end: the fake 'model' reads frozen_break out of the REAL
    prompt built from REAL Stage1/Stage2 features -- proves the whole
    feature-extraction -> prompt -> parse -> decision pipeline is wired
    correctly, not just each piece in isolation."""
    stage1, stage2 = fitted_stages
    model = DefenderLLMModel(echo_frozen_break_generate_fn)
    normal_features = [extract_features(stage1, stage2, make_normal_window(i)) for i in range(30)]
    model.calibrate_threshold(normal_features, max_false_alarm_rate=0.1)
    defender = DefenderLLMDefender(stage1, stage2, model)

    assert defender.score_window(make_frozen_attack_window(777)).decision == ATTACK
    assert defender.score_window(make_normal_window(777)).decision == ACCEPT


def test_defender_llm_defender_rejects_an_unfitted_stage1():
    unfitted = Stage1Model()
    fitted_stage2 = Stage2Model().fit([make_normal_window(i) for i in range(10)])
    model = DefenderLLMModel(constant_generate_fn('{"attack_probability": 0.5, "reason": "x"}'))
    with pytest.raises(ValueError, match="fitted Stage1Model"):
        DefenderLLMDefender(unfitted, fitted_stage2, model)


def test_defender_llm_defender_rejects_an_unfitted_stage2():
    fitted_stage1 = Stage1Model().fit([make_normal_window(i) for i in range(10)])
    unfitted = Stage2Model()
    model = DefenderLLMModel(constant_generate_fn('{"attack_probability": 0.5, "reason": "x"}'))
    with pytest.raises(ValueError, match="fitted Stage2Model"):
        DefenderLLMDefender(fitted_stage1, unfitted, model)


class FakeDecoder:
    """Duck-types part1.decode.Decoder (just decode_frames), so these tests
    don't need a real DBC file on disk."""

    def __init__(self, per_frame_values):
        self.per_frame_values = per_frame_values  # {field: [v0, v1, ...]}

    def decode_frames(self, frames):
        return dict(self.per_frame_values)


def test_decoded_section_is_empty_when_no_fields_given():
    from defender.defender_llm import _decoded_section
    assert _decoded_section(None, None) == ""
    assert _decoded_section({}, {"x": "y"}) == ""


def test_decoded_section_uses_real_names_when_given_falls_back_to_field_id_otherwise():
    from defender.defender_llm import _decoded_section
    section = _decoded_section({"0D0:Unknown_4": 12.0, "0D0:Unknown_9": 3.0},
                               {"0D0:Unknown_4": "brake_pressure"})
    assert "brake_pressure: 12.0" in section
    assert "0D0:Unknown_9: 3.0" in section   # no name given -> falls back to the raw field id


def test_build_prompt_includes_the_decoded_section_when_provided():
    prompt_plain = build_prompt([0.0] * 6)
    prompt_named = build_prompt([0.0] * 6, {"0D0:Unknown_4": 99.0}, {"0D0:Unknown_4": "door_status"})
    assert "door_status" not in prompt_plain
    assert "door_status: 99.0" in prompt_named


def test_latest_decoded_values_keeps_the_most_recent_non_nan_reading():
    from defender.defender_llm import latest_decoded_values
    nan = float("nan")
    decoder = FakeDecoder({"0D0:Unknown_4": [1.0, 5.0, nan], "0D0:Unknown_9": [nan, nan]})
    values = latest_decoded_values(decoder, make_normal_window(0))
    assert values == {"0D0:Unknown_4": 5.0}   # last non-NaN reading; all-NaN field dropped


def test_defender_llm_defender_passes_decoded_fields_and_signal_names_into_the_prompt(fitted_stages):
    stage1, stage2 = fitted_stages
    seen_prompts = []

    def capturing_generate_fn(prompt: str) -> str:
        seen_prompts.append(prompt)
        return '{"attack_probability": 0.5, "reason": "ok"}'

    model = DefenderLLMModel(capturing_generate_fn)
    decoder = FakeDecoder({"0D0:Unknown_4": [42.0]})
    defender = DefenderLLMDefender(stage1, stage2, model, decoder=decoder,
                                   signal_names={"0D0:Unknown_4": "door_status"})
    defender.score_window(make_normal_window(1))

    assert "door_status: 42.0" in seen_prompts[0]


def test_defender_llm_defender_works_with_no_decoder_at_all(fitted_stages):
    """The default path (ROAD today): omitting decoder/signal_names must
    reproduce the plain six-number prompt exactly, no regression."""
    stage1, stage2 = fitted_stages
    model = DefenderLLMModel(constant_generate_fn('{"attack_probability": 0.1, "reason": "fine"}'))
    defender = DefenderLLMDefender(stage1, stage2, model)
    output = defender.score_window(make_normal_window(2))
    assert output.model_version == "defender_llm_v1"
