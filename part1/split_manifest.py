"""
Part 1 -- the frozen split manifest (build plan S6, Block 1).

part1/split_manifest.json is THE split: every capture's neutral id, group
and kind. Training, validation, the dev check, cross-validation and the final
evaluation all read it; no code builds a split from a glob, a seed or a
hard-coded list.

Groups (build plan S6)
    train        normal drives: fit the Defender
    validation   normal drives: choose the threshold
    development  ROAD attack recordings _1 and _2: attack development, v2
                 candidates, cross-validation folds (fold = the index)
    final_test   _3 recordings, single-recording types (coolant) and the test
                 drives: read ONCE, by the final v1-vs-v2 comparison
    separate     accelerator captures: no injected frames, so they are neither
                 attacks nor clean normal traffic; reported on their own

Rules the manifest enforces when it is loaded
    - every capture appears once, with a unique neutral id (cap01, cap02, ...)
    - a fabrication capture and its _masquerade twin share a group
    - only development captures carry a cross-validation fold
    - capture_path() refuses final_test / separate captures unless the caller
      explicitly says it is the final evaluation

How it was built (python -m part1.split_manifest build ...)
    ambient drives   copied from ~/Downloads/road/splits.json (schema 2), whose
                     assignments come from measured regime coverage (docs/02)
    attack captures  index rule: _1/_2 -> development, _3 -> final_test,
                     single recordings -> final_test, accelerators -> separate.
                     Same as splits.json, except fuzzing_1/_2: splits.json keeps
                     all fuzzing in test because that project trains Stage 1 on
                     synthetic fuzzing. We do not, so the index rule applies.
    ids              cap01.. in the order of fleet_simulator.discover_captures
                     (ambient sorted, then attacks sorted)

The manifest holds names and kinds, so it belongs to the evaluator side: the
Defender never reads it and only ever sees neutral ids.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

MANIFEST_PATH = Path(__file__).with_name("split_manifest.json")
GROUPS = ("train", "validation", "development", "final_test", "separate")
EVAL_ONLY_GROUPS = frozenset({"final_test", "separate"})
Kind = Literal["ambient", "fabrication", "masquerade", "fuzzing", "accelerator"]

_ID = re.compile(r"^cap\d{2,}$")
_ATTACK_SUFFIX = re.compile(r"_attack(?:_(?:drive|reverse))?(?:_(\d+))?$")


class CaptureEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str                     # ROAD file name without .log (PRIVATE)
    capture_id: str               # neutral, e.g. "cap07"
    folder: Literal["ambient", "attacks"]
    group: Literal["train", "validation", "development", "final_test", "separate"]
    kind: Kind
    family: Optional[str] = None  # attack family, e.g. "max_speedometer"; None for ambient
    twin: Optional[str] = None    # the other half of a fabrication/masquerade pair
    fold: Optional[int] = None    # cross-validation fold (development only)


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    vehicle_id: str = Field(min_length=1)
    built_from: Dict[str, str] = Field(default_factory=dict)
    rules: List[str] = Field(default_factory=list)
    captures: List[CaptureEntry]

    @model_validator(mode="after")
    def check(self):
        names = [c.name for c in self.captures]
        ids = [c.capture_id for c in self.captures]
        if len(set(names)) != len(names):
            raise ValueError("a capture appears more than once in the manifest")
        if len(set(ids)) != len(ids):
            raise ValueError("two captures share a capture_id")
        by_name = {c.name: c for c in self.captures}
        for c in self.captures:
            if not _ID.match(c.capture_id):
                raise ValueError(f"{c.name}: capture_id {c.capture_id!r} is not neutral (capNN)")
            if (c.fold is not None) != (c.group == "development" and c.kind != "ambient"):
                raise ValueError(f"{c.name}: only development attack captures have a fold")
            if c.twin is not None:
                twin = by_name.get(c.twin)
                if twin is None:
                    raise ValueError(f"{c.name}: twin {c.twin} is not in the manifest")
                if twin.group != c.group:
                    raise ValueError(
                        f"{c.name} ({c.group}) and its twin {c.twin} ({twin.group}) are one "
                        f"recording and must share a split")
        return self

    # ---------- lookups ---------------------------------------------------
    def entry(self, name: str) -> CaptureEntry:
        for c in self.captures:
            if c.name == name:
                return c
        raise KeyError(f"{name} is not in the split manifest")

    def names(self, *groups: str, folder: Optional[str] = None,
              kinds: Optional[set] = None) -> List[str]:
        for g in groups:
            if g not in GROUPS:
                raise ValueError(f"unknown group {g!r}; expected one of {GROUPS}")
        return [c.name for c in self.captures
                if c.group in groups and (folder is None or c.folder == folder)
                and (kinds is None or c.kind in kinds)]

    def development_folds(self) -> Dict[int, List[str]]:
        folds: Dict[int, List[str]] = {}
        for c in self.captures:
            if c.fold is not None:
                folds.setdefault(c.fold, []).append(c.name)
        return dict(sorted(folds.items()))

    def capture_path(self, data_dir, name: str, final_evaluation: bool = False) -> Path:
        """Path of one capture. Refuses final_test / separate captures unless
        final_evaluation=True, so development code cannot read them by accident."""
        c = self.entry(name)
        if c.group in EVAL_ONLY_GROUPS and not final_evaluation:
            raise ValueError(
                f"{name} is a {c.group} capture: only the final evaluation may read it "
                f"(pass final_evaluation=True there, nowhere else)")
        return Path(data_dir).expanduser() / c.folder / f"{name}.log"


def load_manifest(path=MANIFEST_PATH) -> Manifest:
    return Manifest.model_validate_json(Path(path).expanduser().read_text(encoding="utf-8"))


def load_road_metadata(data_dir) -> dict:
    """ROAD's capture_metadata.json, ambient and attacks merged, by name."""
    merged = {}
    for folder in ("ambient", "attacks"):
        path = Path(data_dir).expanduser() / folder / "capture_metadata.json"
        if path.exists():
            merged.update(json.loads(path.read_text(encoding="utf-8")))
    if not merged:
        raise FileNotFoundError(f"no capture_metadata.json under {data_dir}/ambient or attacks")
    return merged


# ---------------------------------------------------------------------
# Building the manifest (run once; the result is committed)
# ---------------------------------------------------------------------
def classify(name: str, meta: dict):
    """(kind, family, index, twin) of one ROAD capture."""
    if name.startswith("ambient_"):
        return "ambient", None, None, None
    base = name[: -len("_masquerade")] if name.endswith("_masquerade") else name
    match = _ATTACK_SUFFIX.search(base)
    if not match:
        raise ValueError(f"cannot derive the attack family of {name!r}")
    family = base[: match.start()]
    index = int(match.group(1)) if match.group(1) else None
    target = meta.get("injection_id")
    if target is None:
        return "accelerator", family, index, None
    if str(target).upper() == "XXX":
        return "fuzzing", family, index, None
    if meta.get("modified"):
        return "masquerade", family, index, base
    return "fabrication", family, index, name + "_masquerade"


def build_manifest(splits_path, data_dir, vehicle_id: str = "veh01") -> Manifest:
    splits = json.loads(Path(splits_path).expanduser().read_text(encoding="utf-8"))
    if splits.get("schema_version") != 2:
        raise ValueError(f"{splits_path}: expected splits.json schema_version 2")
    ambient_group = {}
    for source, group in (("train", "train"), ("val", "validation"), ("test", "final_test")):
        for name in splits[source]["ambient"]:
            ambient_group[name] = group

    metadata = load_road_metadata(data_dir)
    ambient = sorted(n for n in metadata if n.startswith("ambient_"))
    attacks = sorted(n for n in metadata if not n.startswith("ambient_"))
    entries = []
    for number, name in enumerate(ambient + attacks, start=1):
        kind, family, index, twin = classify(name, metadata[name])
        if kind == "ambient":
            if name not in ambient_group:
                raise ValueError(f"{name} is not assigned in {splits_path}")
            group, fold = ambient_group[name], None
        elif kind == "accelerator":
            group, fold = "separate", None
        elif index in (1, 2):
            group, fold = "development", index
        else:
            group, fold = "final_test", None
        if twin is not None and twin not in metadata:
            twin = None          # a fabrication capture without a masquerade version
        entries.append(CaptureEntry(
            name=name, capture_id=f"cap{number:02d}",
            folder="ambient" if kind == "ambient" else "attacks",
            group=group, kind=kind, family=family, twin=twin, fold=fold))
    return Manifest(
        vehicle_id=vehicle_id,
        built_from={"ambient_groups": "~/Downloads/road/splits.json (schema_version 2)",
                    "attack_groups": "index rule: _1/_2 development, _3 and single "
                                     "recordings final_test, accelerator separate",
                    "capture_ids": "discover_captures order: ambient sorted, then attacks sorted"},
        rules=["split by capture, never by window",
               "a fabrication capture and its _masquerade twin share a group",
               "thresholds are chosen on validation normal drives only",
               "final_test and separate captures are read once, by the final evaluation",
               "development attack captures carry a cross-validation fold = their index"],
        captures=entries)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build or show the frozen split manifest.")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="build the manifest from splits.json + ROAD metadata")
    build.add_argument("--splits", required=True, help="path to splits.json (schema_version 2)")
    build.add_argument("--data-dir", required=True, help="ROAD folder with ambient/ attacks/")
    build.add_argument("--out", default=str(MANIFEST_PATH))
    show = sub.add_parser("show", help="print how many captures each group holds")
    show.add_argument("--manifest", default=str(MANIFEST_PATH))
    args = parser.parse_args(argv)

    if args.command == "build":
        manifest = build_manifest(args.splits, args.data_dir)
        Path(args.out).write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        manifest = load_manifest(args.manifest)
    for group in GROUPS:
        entries = [c for c in manifest.captures if c.group == group]
        kinds = {}
        for c in entries:
            kinds[c.kind] = kinds.get(c.kind, 0) + 1
        print(f"  {group:12s} {len(entries):3d} captures  {kinds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
