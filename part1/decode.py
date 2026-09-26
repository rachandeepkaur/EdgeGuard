"""
Part 1 -- decode DBC signals straight from raw payload bytes (build plan S3/S4:
"optional parsed features", "out-of-range decoded values, flatlined values").

ROAD ships an anonymised DBC (signal_extractions/DBC/anonymized.dbc): bit
layout only. Every signal is named Unknown_N with range [0|0], so NOTHING here
knows what a signal means ("speed", "reverse light"); do not add code that
assumes it. What decoding buys is that each flag or value is checked ON ITS OWN
instead of hidden inside a byte with other signals: a reverse-light attack
flips one bit, which a whole-byte range check cannot see.

Bit layout (same rules as the verified decoder in ~/Downloads/road, which
matches all 664 signals of the published signal-extraction CSVs bit for bit):
    @1  Intel (little-endian): start bit is the LSB, bits count upwards
    @0  Motorola (big-endian): start bit is the MSB; the next bit is start-1,
        wrapping from bit 0 of a byte to bit 7 of the next byte (+15)
    bit b lives in byte b // 8, bit b % 8 (LSB = 0)
    values are raw integers (factor 1, offset 0, the DBC's own scaling)

decoded_signals keys are "<CAN ID>:<signal name>", e.g. "0D0:Unknown_4", with
one value per frame of that ID in frame order. Missing data is explicit:
    - an ID with no frames in the window has no keys
    - a payload too short to hold a signal gives NaN for that frame
    - a watched ID that is not in the DBC is an error at construction
"""

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from part1.fleet_simulator import canonical_can_id

DBC_RELATIVE_PATH = Path("signal_extractions") / "DBC" / "anonymized.dbc"

_BO = re.compile(r"^BO_\s+(\d+)\s+(\S+)\s*:\s*(\d+)")
_SG = re.compile(r"^\s*SG_\s+(\S+)\s*:\s*(\d+)\|(\d+)@(\d)([-+])")


@dataclass(frozen=True)
class Signal:
    name: str
    start: int
    width: int
    intel: bool      # @1 little-endian; @0 Motorola
    signed: bool

    def bit_positions(self) -> List[int]:
        """Message bit indices, most significant first."""
        if self.intel:
            return list(range(self.start + self.width - 1, self.start - 1, -1))
        out, b = [], self.start
        for _ in range(self.width):
            out.append(b)
            b = b + 15 if b % 8 == 0 else b - 1
        return out


def parse_dbc(path) -> Dict[str, List[Signal]]:
    """CAN ID (canonical, e.g. '0D0') -> its signals in DBC order."""
    out: Dict[str, List[Signal]] = {}
    current = None
    for line in Path(path).expanduser().read_text(encoding="utf-8").splitlines():
        match = _BO.match(line)
        if match:
            current = canonical_can_id(int(match.group(1)))
            out[current] = []
            continue
        match = _SG.match(line)
        if match and current is not None:
            out[current].append(Signal(name=match.group(1), start=int(match.group(2)),
                                       width=int(match.group(3)),
                                       intel=match.group(4) == "1",
                                       signed=match.group(5) == "-"))
    if not out:
        raise ValueError(f"{path}: no messages found; is this a DBC file?")
    return out


def decode_value(payload: str, signal: Signal) -> float:
    """Raw value of one signal in one payload (hex). NaN if the payload is too short."""
    data = bytes.fromhex(payload)
    value = 0
    for b in signal.bit_positions():
        if b // 8 >= len(data):
            return math.nan
        value = (value << 1) | ((data[b // 8] >> (b % 8)) & 1)
    if signal.signed and value >= 1 << (signal.width - 1):
        value -= 1 << signal.width
    return float(value)


class Decoder:
    """Decodes the signals of a set of watched CAN IDs."""

    def __init__(self, dbc_path, watch_ids: Optional[Iterable[str]] = None):
        dbc = parse_dbc(dbc_path)
        if watch_ids is None:
            self.signals = dbc
        else:
            watch = sorted({canonical_can_id(i) for i in watch_ids})
            missing = [i for i in watch if i not in dbc]
            if missing:
                raise ValueError(f"watched IDs {missing} have no entry in {dbc_path}; "
                                 f"they cannot be decoded")
            self.signals = {i: dbc[i] for i in watch}

    @classmethod
    def from_data_dir(cls, data_dir, watch_ids=None) -> "Decoder":
        return cls(Path(data_dir).expanduser() / DBC_RELATIVE_PATH, watch_ids)

    def decode_frames(self, frames: Sequence) -> Dict[str, List[float]]:
        """Frames with .can_id / .payload (e.g. a TrafficWindow's) ->
        decoded_signals dict (see the module docstring)."""
        out: Dict[str, List[float]] = {}
        for frame in frames:
            can_id = frame.can_id.upper()
            for signal in self.signals.get(can_id, ()):
                out.setdefault(f"{can_id}:{signal.name}", []).append(
                    decode_value(frame.payload, signal))
        return out


def verify_against_csv(decoder: Decoder, log_frames: Iterable, csv_path,
                       max_rows: int = 200_000) -> dict:
    """Check the decoder against ROAD's signal-extraction CSV (columns read by
    NAME: Label, Time, ID (decimal), Signal_1_of_ID ...). The CSV rows follow
    the log's frame order but leave out IDs that have no DBC entry (e.g. the
    FFF filler), so those log frames are skipped here too. The decoder must
    cover every DBC message (watch_ids=None). Returns {checked, mismatches, first}."""
    checked = mismatches = 0
    first = None
    in_dbc = (f for f in log_frames if f.can_id.upper() in decoder.signals)
    with open(Path(csv_path).expanduser(), newline="", encoding="utf-8") as fh:
        rows = csv.DictReader(fh)
        signal_cols = [c for c in rows.fieldnames if c.startswith("Signal_")]
        for frame, row in zip(in_dbc, rows):
            if checked >= max_rows:
                break
            can_id = canonical_can_id(int(row["ID"]))
            if frame.can_id.upper() != can_id:
                raise ValueError(f"log and CSV are out of step at row {checked}: "
                                 f"{frame.can_id} vs {can_id}")
            for signal, col in zip(decoder.signals.get(can_id, ()), signal_cols):
                expected = row[col]
                if expected == "":
                    continue
                if decode_value(frame.payload, signal) != float(expected):
                    mismatches += 1
                    first = first or (checked, can_id, signal.name, expected)
            checked += 1
    return {"checked_rows": checked, "mismatches": mismatches, "first_mismatch": first}
