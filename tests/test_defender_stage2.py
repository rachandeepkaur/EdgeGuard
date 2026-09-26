"""
Tests for defender/stage2.py.

Run from the EdgeGuard folder with:
    python -m pytest tests/test_defender_stage2.py -v

ALL windows here are MOCK traffic made up for testing. They imitate the
shape of the ROAD masquerade families (a forced/frozen byte or an
out-of-range value) but are not real data, so these tests prove the
logic works, not how well it does on ROAD.
"""

import pytest

from defender.stage1 import Stage1Model
from defender.stage2 import Stage2Model
from defender.threshold import ATTACK, ACCEPT, choose_threshold, make_decision
from shared.schemas import TrafficWindow

# MOCK normal traffic: ID -> period in seconds, and a payload generator
MOCK_PERIODS = {"0D0": 0.010, "0F4": 0.020}


def mock_payload(can_id: str, k: int) -> str:
    """Deterministic, varying MOCK payload (0D0 byte0 cycles, 0F4 pair0 cycles)."""
    if can_id == "0D0":
        b0 = k % 200          # varies a lot, like a real signal or counter
        return f"{b0:02X}00000000000000"
    pair = (1000 + k * 7) % 4000     # varies, e.g. a 2-byte speed-like value
    b0, b1 = pair >> 8, pair & 0xFF
    return f"{b0:02X}{b1:02X}0000000000000"[:16]


def make_normal_window(index: int) -> TrafficWindow:
    """MOCK 1-second window with varying payloads, normal timing."""
    start = 1000.0 + index
    frames = []
    for can_id, period in MOCK_PERIODS.items():
        count = int(round(1.0 / period))
        for k in range(count):
            t = round(start + 0.001 + k * period, 6)
            frames.append({"timestamp": t, "can_id": can_id,
                           "payload": mock_payload(can_id, index * 100 + k)})
    frames.sort(key=lambda f: f["timestamp"])
    return TrafficWindow(window_id=f"cap01_w{index:04d}", capture_id="cap01",
                         window_start=start, window_end=start + 1.0, frames=frames)


def with_payload_override(window: TrafficWindow, can_id: str, payload: str) -> TrafficWindow:
    """MOCK masquerade: return a COPY where every frame of can_id has this payload."""
    data = window.model_dump()
    for frame in data["frames"]:
        if frame["can_id"] == can_id:
            frame["payload"] = payload
    return TrafficWindow(**data)


@pytest.fixture(scope="module")
def model():
    return Stage2Model().fit([make_normal_window(i) for i in range(40)])


@pytest.fixture(scope="module")
def normal_validation_scores(model):
    return [model.score(make_normal_window(i)).score for i in range(100, 130)]


# ---------- MOCK attacks (masquerade-style) ----------------------------
def frozen_byte_window():
    """0D0 byte0 held at a value it never reaches in training -> out_of_range AND frozen."""
    w = make_normal_window(200)
    return with_payload_override(w, "0D0", "FF00000000000000")


def frozen_but_in_range_window():
    """0D0 byte0 held constant at a value seen in training -> frozen_break only."""
    w = make_normal_window(201)
    return with_payload_override(w, "0D0", "0500000000000000")


def large_jump_window():
    """One 0D0 frame jumps far beyond any training jump, then returns."""
    w = make_normal_window(202)
    data = w.model_dump()
    d0_frames = [f for f in data["frames"] if f["can_id"] == "0D0"]
    d0_frames[len(d0_frames) // 2]["payload"] = "C800000000000000"   # 200, far from neighbours
    return TrafficWindow(**data)


def timing_only_attack_window():
    """Extra unknown-ID frames: Stage 1's job, NOT Stage 2's. Payload content is untouched."""
    data = make_normal_window(203).model_dump()
    data["frames"].append({"timestamp": data["window_start"] + 0.5, "can_id": "7FF",
                           "payload": "FFFFFFFFFFFFFFFF"})
    return TrafficWindow(**data)


# ---------- Normal behaviour ------------------------------------------
def test_scores_are_between_zero_and_one(model):
    for w in [make_normal_window(300), frozen_byte_window(), large_jump_window()]:
        result = model.score(w)
        assert 0.0 <= result.score < 1.0
        assert set(result.sub_scores) == {"out_of_range", "frozen_break", "large_jump"}


def test_normal_window_has_no_payload_evidence(model):
    result = model.score(make_normal_window(5))       # a training window
    assert "No payload anomaly" in result.evidence


def test_scoring_is_deterministic(model):
    w = frozen_byte_window()
    assert model.score(w).score == model.score(w).score


# ---------- Attacks score above all normal windows ----------------------
def test_out_of_range_value_detected(model, normal_validation_scores):
    """Forcing 0D0 byte0 out of range also forces its byte-pair fields out of
    range; either evidence is correct, since the score is the max over all
    fields. This test only checks the attack is caught and attributed to 0D0."""
    result = model.score(frozen_byte_window())
    assert result.score > max(normal_validation_scores)
    assert "0D0" in result.evidence


def test_frozen_in_range_value_detected(model, normal_validation_scores):
    result = model.score(frozen_but_in_range_window())
    assert result.score > max(normal_validation_scores)
    assert "frozen_break" in result.evidence


def test_large_jump_detected(model, normal_validation_scores):
    result = model.score(large_jump_window())
    assert result.score > max(normal_validation_scores)
    assert "large_jump" in result.evidence


def test_timing_only_attack_is_a_known_limit(model):
    """An attack that only adds an unknown ID (no payload change) is Stage 1's job."""
    original = make_normal_window(203)
    assert model.score(timing_only_attack_window()).score == model.score(original).score


# ---------- Byte-pair (2-byte / 16-bit) detection -----------------------
def test_two_byte_value_out_of_range_detected(model, normal_validation_scores):
    """0F4's first pair (bytes 0-1) held far outside its trained 1000-5000 range."""
    w = with_payload_override(make_normal_window(204), "0F4", "FFFF000000000000")
    result = model.score(w)
    assert result.score > max(normal_validation_scores)
    assert "pair0" in result.evidence


# ---------- Works with threshold.py and alongside Stage 1 ----------------
def test_end_to_end_with_threshold(model, normal_validation_scores):
    threshold = choose_threshold(normal_validation_scores, 0.0)
    for s in normal_validation_scores:
        assert make_decision(s, threshold) == ACCEPT
    for w in [frozen_byte_window(), frozen_but_in_range_window(), large_jump_window()]:
        assert make_decision(model.score(w).score, threshold) == ATTACK


def test_stage1_and_stage2_are_complementary(model):
    """Stage 1 catches timing-only attacks that Stage 2 does not, and vice versa."""
    stage1 = Stage1Model().fit([make_normal_window(i) for i in range(40)])
    timing_attack = timing_only_attack_window()
    payload_attack = frozen_byte_window()

    assert stage1.score(timing_attack).score > stage1.score(make_normal_window(0)).score
    assert model.score(timing_attack).score == model.score(make_normal_window(203)).score

    assert model.score(payload_attack).score > model.score(make_normal_window(0)).score


# ---------- Safety -------------------------------------------------------
def test_scoring_does_not_modify_window(model):
    w = frozen_byte_window()
    before = w.model_dump()
    model.score(w)
    assert w.model_dump() == before


def test_lowercase_ids_match_uppercase(model):
    w = make_normal_window(301)
    data = w.model_dump()
    for frame in data["frames"]:
        frame["can_id"] = frame["can_id"].lower()
    lower = TrafficWindow(**data)
    assert model.score(lower).score == model.score(w).score


# ---------- Invalid use ----------------------------------------------------
def test_score_before_fit_gives_clear_error():
    with pytest.raises(RuntimeError, match="not trained"):
        Stage2Model().score(make_normal_window(0))


def test_fit_rejects_empty_list():
    with pytest.raises(ValueError, match="at least 2"):
        Stage2Model().fit([])


def test_fit_rejects_non_window():
    with pytest.raises(TypeError, match="expected TrafficWindow"):
        Stage2Model().fit([make_normal_window(0), {"frames": []}])


def test_score_rejects_dictionary(model):
    with pytest.raises(TypeError, match="needs a TrafficWindow"):
        model.score({"frames": []})


def test_unseen_can_id_does_not_crash(model):
    """A CAN ID never seen in training: Stage 2 skips it (Stage 1's job)."""
    data = make_normal_window(302).model_dump()
    data["frames"].append({"timestamp": data["window_start"] + 0.5, "can_id": "7FF",
                           "payload": "0000000000000000"})
    result = model.score(TrafficWindow(**data))
    assert 0.0 <= result.score <= 1.0


# ---------- Save / load ------------------------------------------------
def test_save_and_load_give_identical_scores(model, tmp_path):
    path = tmp_path / "stage2_v2.json"
    model.save(path, "v2")
    loaded = Stage2Model.load(path, "v2")
    for w in [make_normal_window(400), frozen_byte_window(), large_jump_window()]:
        assert loaded.score(w).score == model.score(w).score


def test_load_refuses_other_model_version(model, tmp_path):
    path = tmp_path / "stage2_v2.json"
    model.save(path, "v2")
    with pytest.raises(ValueError, match="Model version mismatch"):
        Stage2Model.load(path, "v3")


def test_cannot_save_untrained_model(tmp_path):
    with pytest.raises(RuntimeError, match="untrained"):
        Stage2Model().save(tmp_path / "x.json", "v2")


# ---------- Ranges learned from full captures (real-data fix) ----------
def _frames_of(windows):
    for w in windows:
        for f in w.frames:
            yield (f.timestamp, f.can_id, f.payload)


def test_full_capture_ranges_prevent_false_alarm():
    """A field that is constant in the SAMPLED windows but changes elsewhere in
    the full capture must not be flagged when that normal change appears."""
    def window_with(value, index):
        frames = [{"timestamp": 1000.0 + index + k * 0.02, "can_id": "371",
                   "payload": f"{value:02X}" + "00" * 7} for k in range(40)]
        return TrafficWindow(window_id=f"cap01_w{index:04d}", capture_id="cap01",
                             window_start=1000.0 + index, window_end=1001.0 + index,
                             frames=frames)

    sampled = [window_with(5, i) for i in range(10)]            # byte0 always 5 in samples
    full_capture = sampled + [window_with(6, 50)]               # rare normal change to 6
    rare_but_normal = window_with(6, 99)

    windows_only = Stage2Model().fit(sampled)
    with_full = Stage2Model().fit(sampled, range_captures=[_frames_of(full_capture)])

    top = 10 / 11
    assert windows_only.score(rare_but_normal).score > top      # false alarm before the fix
    assert with_full.score(rare_but_normal).score <= top        # no false alarm after


def test_jumps_do_not_cross_capture_boundaries():
    """The last frame of one capture and the first of the next are not a 'jump'."""
    def frames(value):
        return [(1.0 + k, "0D0", f"{value:02X}" + "00" * 7) for k in range(5)]
    model = Stage2Model().fit([make_normal_window(0), make_normal_window(1)],
                              range_captures=[frames(10), frames(200)])
    assert model.field_max_jump.get("0D0|byte0", 0) < 190


# ---------- Watch-list --------------------------------------------------
def test_watch_list_ignores_unwatched_ids(normal_validation_scores):
    """0D0 attacked, but only 0F4 is watched -> Stage 2 does not see it."""
    watched = Stage2Model(watch_ids=["0F4"]).fit([make_normal_window(i) for i in range(40)])
    attack = frozen_byte_window()                       # attacks 0D0
    assert "No payload anomaly" in watched.score(attack).evidence


def test_watch_list_still_catches_watched_id():
    watched = Stage2Model(watch_ids=["0D0"]).fit([make_normal_window(i) for i in range(40)])
    validation = [watched.score(make_normal_window(i)).score for i in range(100, 130)]
    assert watched.score(frozen_byte_window()).score > max(validation)


def test_watch_ids_are_normalized():
    from defender.stage2 import normalize_can_id
    assert normalize_can_id("0xd0") == "0D0"
    assert normalize_can_id("0x6e0") == "6E0"
    assert normalize_can_id("d0") == "0D0"
    assert Stage2Model(watch_ids=["0xd0", "0D0"]).watch_ids == ["0D0"]


def test_bad_watch_ids_rejected():
    with pytest.raises(ValueError, match="hexadecimal"):
        Stage2Model(watch_ids=["speed"])
    with pytest.raises(ValueError, match="empty"):
        Stage2Model(watch_ids=[])


def test_watch_list_saved_and_loaded(tmp_path):
    model = Stage2Model(watch_ids=["0D0"]).fit([make_normal_window(i) for i in range(10)])
    model.save(tmp_path / "s2.json", "v2")
    loaded = Stage2Model.load(tmp_path / "s2.json", "v2")
    assert loaded.watch_ids == ["0D0"]
    assert loaded.score(frozen_byte_window()).score == model.score(frozen_byte_window()).score


# ---------- v3 hardening: "rate" frozen check ---------------------------
def _counter_window(index, freeze_counter=False):
    """MOCK 0D0: byte0 = rolling counter (changes EVERY frame);
    bytes 1-2 = slow 16-bit signal, constant within a window but very
    different between windows (wide range)."""
    start = 1000.0 + index
    slow = (index * 1500) % 60000
    frames = []
    for k in range(50):
        counter = 0x3A if freeze_counter else (index * 50 + k) * 8 % 256
        payload = f"{counter:02X}{slow >> 8:02X}{slow & 0xFF:02X}" + "00" * 5
        frames.append({"timestamp": round(start + 0.001 + k * 0.02, 6),
                       "can_id": "0D0", "payload": payload})
    return TrafficWindow(window_id=f"cap01_w{index:04d}", capture_id="cap01",
                         window_start=start, window_end=start + 1.0, frames=frames)


def test_rate_mode_catches_frozen_counter():
    model = Stage2Model(frozen_mode="rate").fit([_counter_window(i) for i in range(40)])
    validation = [model.score(_counter_window(i)).score for i in range(100, 130)]
    result = model.score(_counter_window(200, freeze_counter=True))
    assert result.score > max(validation)
    assert "frozen_break" in result.evidence and "0D0" in result.evidence


def test_rate_mode_no_false_alarm_on_normal_slow_signal():
    """The slow field is constant in every normal window: freezing it is normal."""
    model = Stage2Model(frozen_mode="rate").fit([_counter_window(i) for i in range(40)])
    assert "No payload anomaly" in model.score(_counter_window(150)).evidence


def test_frozen_mode_saved_and_old_files_default_to_width(tmp_path):
    import json as _json
    model = Stage2Model(frozen_mode="rate").fit([_counter_window(i) for i in range(10)])
    model.save(tmp_path / "s2.json", "v3")
    assert Stage2Model.load(tmp_path / "s2.json", "v3").frozen_mode == "rate"
    record = _json.loads((tmp_path / "s2.json").read_text(encoding="utf-8"))
    del record["frozen_mode"]
    del record["frozen_counts"]
    (tmp_path / "old.json").write_text(_json.dumps(record), encoding="utf-8")
    assert Stage2Model.load(tmp_path / "old.json", "v3").frozen_mode == "width"


def test_rejects_unknown_frozen_mode():
    with pytest.raises(ValueError, match="frozen_mode"):
        Stage2Model(frozen_mode="magic")

# ---------------------------------------------------------------------
# bound_percentile: percentile out-of-range bounds instead of strict min/max
# ---------------------------------------------------------------------
def _percentile_window(index: int, byte0_values):
    """MOCK window: 0D0 byte0 takes exactly these values, one frame each."""
    start = 1000.0 + index
    frames = [{"timestamp": round(start + 0.01 + k * 0.02, 6), "can_id": "0D0",
              "payload": f"{v:02X}00000000000000"} for k, v in enumerate(byte0_values)]
    return TrafficWindow(window_id=f"cap01_w{index:04d}", capture_id="cap01",
                         window_start=start, window_end=start + 1.0, frames=frames)


def _percentile_training_windows():
    """19 windows with byte0 spread over roughly [10, 90], plus ONE window
    with a single spurious frame at 250 -- a rare-but-normal extreme that
    would otherwise widen the learned range far past where byte0 usually is."""
    normal = [_percentile_window(i, [10 + (i * 7 + k * 3) % 81 for k in range(5)])
             for i in range(19)]
    outlier = _percentile_window(19, [10, 20, 250, 30, 40])
    return normal + [outlier]


def test_bound_percentile_none_keeps_the_full_training_range():
    model = Stage2Model(watch_ids=["0D0"], bound_percentile=None).fit(
        _percentile_training_windows())
    assert model.field_range["0D0|byte0"] == (10, 250)
    assert model.field_bounds["0D0|byte0"] == (10.0, 250.0)


def test_bound_percentile_narrows_out_of_range_bounds_but_not_field_range():
    train = _percentile_training_windows()
    model = Stage2Model(watch_ids=["0D0"], bound_percentile=10.0).fit(train)
    # field_range (used by frozen_break) is UNCHANGED: still the true min/max.
    assert model.field_range["0D0|byte0"] == (10, 250)
    # field_bounds (used by out_of_range) is narrowed away from the outlier.
    assert model.field_bounds["0D0|byte0"] == pytest.approx((17.9, 78.1))


def test_bound_percentile_flags_a_value_the_full_range_would_accept():
    train = _percentile_training_windows()
    no_percentile = Stage2Model(watch_ids=["0D0"], bound_percentile=None).fit(train)
    p10 = Stage2Model(watch_ids=["0D0"], bound_percentile=10.0).fit(train)
    probe = _percentile_window(100, [150])   # inside [10,250], outside [17.9,78.1]

    assert no_percentile.score(probe).sub_scores["out_of_range"] == 0.0
    assert p10.score(probe).sub_scores["out_of_range"] == pytest.approx(0.9047619047619048)


def test_bound_percentile_does_not_change_frozen_break_strength():
    """frozen_break's strength uses the FULL training width, never the
    percentile-narrowed bounds -- a change aimed at out_of_range must not
    make frozen_break more sensitive as a side effect."""
    train = _percentile_training_windows()
    no_percentile = Stage2Model(watch_ids=["0D0"], bound_percentile=None).fit(train)
    p10 = Stage2Model(watch_ids=["0D0"], bound_percentile=10.0).fit(train)
    frozen_probe = _percentile_window(101, [42] * 5)

    assert (no_percentile.score(frozen_probe).sub_scores["frozen_break"]
           == p10.score(frozen_probe).sub_scores["frozen_break"])


@pytest.mark.parametrize("bad", [0, 50, -1, 51, 100])
def test_rejects_out_of_bounds_bound_percentile(bad):
    with pytest.raises(ValueError, match="bound_percentile"):
        Stage2Model(bound_percentile=bad)


def test_bound_percentile_saved_and_old_files_default_to_none(tmp_path):
    import json as _json
    train = _percentile_training_windows()
    model = Stage2Model(watch_ids=["0D0"], bound_percentile=10.0).fit(train)
    model.save(tmp_path / "s2.json", "v4")

    loaded = Stage2Model.load(tmp_path / "s2.json", "v4")
    assert loaded.bound_percentile == 10.0
    assert loaded.field_bounds == model.field_bounds

    record = _json.loads((tmp_path / "s2.json").read_text(encoding="utf-8"))
    del record["bound_percentile"]
    del record["field_bounds"]
    (tmp_path / "old.json").write_text(_json.dumps(record), encoding="utf-8")
    old_loaded = Stage2Model.load(tmp_path / "old.json", "v4")
    assert old_loaded.bound_percentile is None
    assert old_loaded.field_bounds == old_loaded.field_range


def test_capture_range_stats_and_merge_keep_raw_values_sorted():
    """capture_range_stats() sorts per capture; merge_range_stats() must keep
    the merge sorted (k-way merge), since fit() relies on that ordering for
    percentile bounds without re-sorting the whole thing every fit."""
    from defender.stage2 import merge_range_stats

    model = Stage2Model(watch_ids=["0D0"])
    capture_a = [(float(i), "0D0", f"{v:02X}00000000000000")
                for i, v in enumerate([50, 10, 90, 30])]
    capture_b = [(float(i), "0D0", f"{v:02X}00000000000000")
                for i, v in enumerate([20, 80, 5])]
    stats_a = model.capture_range_stats(capture_a)
    stats_b = model.capture_range_stats(capture_b)
    assert stats_a[3]["0D0|byte0"] == sorted(stats_a[3]["0D0|byte0"])

    merged = merge_range_stats([stats_a, stats_b])
    assert merged[3]["0D0|byte0"] == sorted(merged[3]["0D0|byte0"])
    assert merged[3]["0D0|byte0"] == sorted(stats_a[3]["0D0|byte0"] + stats_b[3]["0D0|byte0"])
