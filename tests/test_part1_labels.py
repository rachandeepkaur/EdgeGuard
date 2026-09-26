"""
Tests for part1/labels.py: private frame and window labels.

Run from the EdgeGuard folder with:
    python -m pytest tests/test_part1_labels.py -v
"""

import pytest

from part1.labels import InjectionRule, injection_rule, is_injected, mask_match, window_label
from part1.split_manifest import CaptureEntry

FAB = InjectionRule("fabrication", "speed", "0D0", (8_000_000, 12_000_000), "XXXXXXXXXXFFXXXX")
MASQ = InjectionRule("masquerade", "speed", "0D0", (8_000_000, 12_000_000), "XXXXXXXXXXFFXXXX")
FUZZ = InjectionRule("fuzzing", "fuzzing", None, (8_000_000, 12_000_000), "FFFFFFFFFFFFFFFF")
NONE = InjectionRule("ambient", None, None, None, None)


def test_mask_match_uses_x_as_wildcard():
    assert mask_match("12345678AAFF0000", "XXXXXXXXXXFFXXXX")
    assert not mask_match("12345678AAFE0000", "XXXXXXXXXXFFXXXX")
    assert not mask_match("12", "XXXXXXXXXXFFXXXX")                 # wrong length


def test_masquerade_labels_every_target_frame_inside_the_interval():
    assert is_injected(MASQ, (9_000_000, "0D0", "0000000000000000"))
    assert not is_injected(MASQ, (9_000_000, "0F4", "0000000000FF0000"))   # other ID
    assert not is_injected(MASQ, (13_000_000, "0D0", "0000000000FF0000"))  # after


def test_fabrication_also_needs_the_payload_mask():
    assert is_injected(FAB, (9_000_000, "0D0", "0000000000FF0000"))
    assert not is_injected(FAB, (9_000_000, "0D0", "0000000000000000"))    # a real frame


def test_interval_is_closed_in_integer_microseconds():
    assert is_injected(MASQ, (8_000_000, "0D0", "00"))
    assert is_injected(MASQ, (12_000_000, "0D0", "00"))
    assert not is_injected(MASQ, (12_000_001, "0D0", "00"))


def test_fuzzing_labels_by_payload_over_the_whole_capture():
    assert is_injected(FUZZ, (1_000, "7AB", "FFFFFFFFFFFFFFFF"))           # outside interval
    assert not is_injected(FUZZ, (9_000_000, "7AB", "FFFFFFFFFFFFFFFE"))


def test_ambient_and_accelerator_have_no_injected_frames():
    assert not is_injected(NONE, (9_000_000, "0D0", "FFFFFFFFFFFFFFFF"))
    accel = InjectionRule("accelerator", "accelerator", None, None, None)
    assert not is_injected(accel, (9_000_000, "0D0", "FFFFFFFFFFFFFFFF"))


def test_window_label_counts_injected_frames_and_overlap():
    frames = [(11_500_000, "0D0", "0000000000FF0000"), (11_600_000, "0D0", "0000000000000000")]
    label = window_label(FAB, "cap07_w0011", 11_000_000, 12_000_000, frames)
    assert label.is_attack and label.injected_frames == 1
    assert label.family == "speed" and label.target == "0D0"
    assert label.injection_interval == [8.0, 12.0]
    assert label.interval_overlap_s == pytest.approx(1.0)
    partial = window_label(FAB, "cap07_w0012", 11_500_000, 12_500_000, frames[:1])
    assert partial.interval_overlap_s == pytest.approx(0.5)


def test_window_without_injected_frames_is_normal():
    label = window_label(MASQ, "cap07_w0001", 1_000_000, 2_000_000, [(1_500_000, "0D0", "00")])
    assert not label.is_attack and label.injected_frames == 0 and label.family is None


def test_injection_rule_from_manifest_and_road_metadata():
    entry = CaptureEntry(name="speed_attack_1", capture_id="cap07", folder="attacks",
                         group="development", kind="fabrication", family="speed", fold=1)
    rule = injection_rule(entry, {"speed_attack_1": {
        "injection_id": "0xd0", "injection_interval": [42.009204, 66.449011],
        "injection_data_str": "XXXXXXXXXXFFXXXX"}})
    assert rule.target == "0D0"                                   # 0xd0 -> padded, upper
    assert rule.interval_us == (42_009_204, 66_449_011)           # exact microseconds


def test_injection_rule_requires_metadata_for_targeted_attacks():
    entry = CaptureEntry(name="speed_attack_1", capture_id="cap07", folder="attacks",
                         group="development", kind="masquerade", family="speed", fold=1)
    with pytest.raises(ValueError, match="needs injection_id"):
        injection_rule(entry, {"speed_attack_1": {}})
