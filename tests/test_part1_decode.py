"""
Tests for part1/decode.py: DBC signals from raw payload bytes.

Run from the EdgeGuard folder with:
    python -m pytest tests/test_part1_decode.py -v
"""

import math

import pytest

from part1.decode import Decoder, Signal, decode_value, parse_dbc, verify_against_csv
from shared.schemas import Frame


def test_parse_dbc_reads_layout(fake_road):
    dbc = parse_dbc(fake_road.data_dir / "signal_extractions" / "DBC" / "anonymized.dbc")
    assert sorted(dbc) == ["0D0", "0F4"]                       # 208, 244 -> canonical hex
    assert dbc["0D0"][1] == Signal("Unknown_1", 40, 8, intel=True, signed=False)


def test_motorola_and_intel_bit_orders():
    payload = "0102030405060708"
    assert decode_value(payload, Signal("m", 7, 16, intel=False, signed=False)) == 0x0102
    assert decode_value(payload, Signal("i", 0, 16, intel=True, signed=False)) == 0x0201
    # Motorola field crossing a byte boundary: start bit 3, 8 bits wide
    # -> low nibble of byte 0 (1) then high nibble of byte 1 (0) = 0x10
    assert decode_value(payload, Signal("x", 3, 8, intel=False, signed=False)) == 0x10


def test_single_bit_flag_and_signed_values():
    assert decode_value("0000000000080000", Signal("flag", 43, 1, False, False)) == 1.0
    assert decode_value("0000000000F70000", Signal("flag", 43, 1, False, False)) == 0.0
    assert decode_value("FF", Signal("s", 7, 8, intel=False, signed=True)) == -1.0


def test_short_payload_gives_nan_not_a_guess():
    assert math.isnan(decode_value("01", Signal("late", 40, 8, True, False)))


def test_decoder_keys_are_id_and_signal_name(fake_road):
    decoder = Decoder.from_data_dir(fake_road.data_dir, watch_ids=["0xd0"])
    frames = [Frame(timestamp=0.1, can_id="0D0", payload="07000000000F0000"),
              Frame(timestamp=0.2, can_id="0F4", payload="0102030405060708"),
              Frame(timestamp=0.3, can_id="0D0", payload="08000000000F0000")]
    out = decoder.decode_frames(frames)
    assert out == {"0D0:Unknown_0": [7.0, 8.0], "0D0:Unknown_1": [15.0, 15.0],
                   "0D0:Unknown_2": [1.0, 1.0]}                # 0F4 is not watched


def test_watching_an_id_missing_from_the_dbc_is_an_error(fake_road):
    with pytest.raises(ValueError, match="no entry"):
        Decoder.from_data_dir(fake_road.data_dir, watch_ids=["123"])


def test_verify_against_csv_skips_ids_without_dbc_entry(fake_road, tmp_path):
    decoder = Decoder.from_data_dir(fake_road.data_dir)
    frames = [Frame(timestamp=0.0, can_id="FFF", payload="00"),                 # not in DBC
              Frame(timestamp=0.1, can_id="0F4", payload="0102030405060708")]
    csv = tmp_path / "x.csv"
    csv.write_text("Label,Time,ID,Signal_1_of_ID\n0,0.1,244,258\n", encoding="utf-8")
    assert verify_against_csv(decoder, frames, csv) == \
        {"checked_rows": 1, "mismatches": 0, "first_mismatch": None}
    csv.write_text("Label,Time,ID,Signal_1_of_ID\n0,0.1,244,259\n", encoding="utf-8")
    assert verify_against_csv(decoder, frames, csv)["mismatches"] == 1
