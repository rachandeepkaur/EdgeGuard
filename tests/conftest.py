"""
Shared FAKE ROAD folder for tests. Never touches the real dataset.

make_fake_road(tmp_path) writes small ROAD-format logs for every capture kind,
ROAD-style capture_metadata.json files, a tiny DBC, and a matching split
manifest, and returns a FakeRoad with the paths and the Manifest.

Traffic: 0F4 every 50 ms from exactly t = 0 (so elapsed time matches the
interval, as in ROAD) and 0D0 every 20 ms (byte 0 counts 0..199, byte 5 = 00,
small non-negative jitter). Attacks inside [8.0, 11.99] s set 0D0 byte 5 to
FF, the ROAD mask XXXXXXXXXXFFXXXX:
    fabrication  an EXTRA 0D0 frame with byte 5 = FF next to each real one
    masquerade   the real 0D0 frames themselves get byte 5 = FF
    fuzzing      extra frames on unknown IDs with payload FFFFFFFFFFFFFFFF
    accelerator  normal traffic, no injected frames
"""

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from part1.split_manifest import CaptureEntry, Manifest

T0 = 1110000000.0
SECONDS = 20.0
INTERVAL = [8.0, 11.99]      # windows 8..11 hold the attack
MASK = "XXXXXXXXXXFFXXXX"

# name, group, kind, family, twin, fold
CAPTURES = [
    ("ambient_a", "train", "ambient", None, None, None),
    ("ambient_b", "train", "ambient", None, None, None),
    ("ambient_c", "train", "ambient", None, None, None),
    ("ambient_d", "validation", "ambient", None, None, None),
    ("ambient_e", "validation", "ambient", None, None, None),
    ("ambient_f", "final_test", "ambient", None, None, None),
    ("speed_attack_1", "development", "fabrication", "speed", "speed_attack_1_masquerade", 1),
    ("speed_attack_1_masquerade", "development", "masquerade", "speed", "speed_attack_1", 1),
    ("speed_attack_2", "development", "fabrication", "speed", "speed_attack_2_masquerade", 2),
    ("speed_attack_2_masquerade", "development", "masquerade", "speed", "speed_attack_2", 2),
    ("fuzzing_attack_1", "development", "fuzzing", "fuzzing", None, 1),
    ("speed_attack_3", "final_test", "fabrication", "speed", "speed_attack_3_masquerade", None),
    ("speed_attack_3_masquerade", "final_test", "masquerade", "speed", "speed_attack_3", None),
    ("accelerator_attack_drive_1", "separate", "accelerator", "accelerator", None, None),
]

# 0D0: Unknown_0 = byte 0 (Motorola 7|8), Unknown_1 = byte 5 (Intel 40|8),
#      Unknown_2 = bit 3 of byte 5 (a one-bit flag)
DBC = """VERSION ""

BO_ 208 MSG_0xd0: 8 X
 SG_ Unknown_0 : 7|8@0+ (1,0) [0|0] "" X
 SG_ Unknown_1 : 40|8@1+ (1,0) [0|0] "" X
 SG_ Unknown_2 : 43|1@0+ (1,0) [0|0] "" X

BO_ 244 MSG_0xf4: 8 X
 SG_ Unknown_0 : 7|16@0+ (1,0) [0|0] "" X
"""


def d0(k: int, byte5: str = "00") -> str:
    return f"{k % 200:02X}00000000{byte5}0000"


def traffic(kind: str, seed: int = 0):
    """(elapsed seconds, 'ID#PAYLOAD') lines of one fake capture."""
    lines = []
    lo, hi = INTERVAL
    for k in range(int(SECONDS / 0.02)):
        t = round(k * 0.02 + ((seed * 7 + k * 3) % 5) * 0.0002 + 0.001, 6)
        inside = lo <= t <= hi
        lines.append((t, "0D0#" + d0(k, "FF" if kind == "masquerade" and inside else "00")))
        if kind == "fabrication" and inside:
            lines.append((t, "0D0#" + d0(k, "FF")))
    for k in range(int(SECONDS / 0.05)):
        lines.append((round(k * 0.05, 6), "0F4#0102030405060708"))
    if kind == "fuzzing":
        for k in range(20):
            lines.append((round(lo + k * 0.1 + 0.001, 6), f"{0x700 + k:03X}#FFFFFFFFFFFFFFFF"))
    lines.sort()
    return lines


def write_log(path: Path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f"({T0 + t:.6f}) can0 {rest}" for t, rest in lines) + "\n",
                    encoding="utf-8")


def road_metadata_for(name, kind):
    if kind == "ambient":
        return {"elapsed_sec": SECONDS, "on_dyno": True}
    if kind == "accelerator":
        return {"injection_id": None, "injection_interval": None, "injection_data_str": None,
                "modified": False}
    if kind == "fuzzing":
        return {"injection_id": "XXX", "injection_interval": INTERVAL,
                "injection_data_str": "FFFFFFFFFFFFFFFF", "modified": False}
    return {"injection_id": "0xd0", "injection_interval": INTERVAL, "injection_data_str": MASK,
            "modified": kind == "masquerade"}


@dataclass
class FakeRoad:
    data_dir: Path
    manifest_path: Path
    manifest: Manifest


def make_fake_road(tmp_path) -> FakeRoad:
    data_dir = tmp_path / "road"
    metadata = {"ambient": {}, "attacks": {}}
    entries = []
    for number, (name, group, kind, family, twin, fold) in enumerate(CAPTURES, start=1):
        folder = "ambient" if kind == "ambient" else "attacks"
        write_log(data_dir / folder / f"{name}.log", traffic(kind, seed=number))
        metadata[folder][name] = road_metadata_for(name, kind)
        entries.append(CaptureEntry(name=name, capture_id=f"cap{number:02d}", folder=folder,
                                    group=group, kind=kind, family=family, twin=twin, fold=fold))
    for folder, records in metadata.items():
        (data_dir / folder / "capture_metadata.json").write_text(json.dumps(records))
    dbc = data_dir / "signal_extractions" / "DBC" / "anonymized.dbc"
    dbc.parent.mkdir(parents=True)
    dbc.write_text(DBC, encoding="utf-8")
    manifest = Manifest(vehicle_id="veh01", captures=entries)
    manifest_path = tmp_path / "split_manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return FakeRoad(data_dir, manifest_path, manifest)


@pytest.fixture
def fake_road(tmp_path) -> FakeRoad:
    return make_fake_road(tmp_path)
