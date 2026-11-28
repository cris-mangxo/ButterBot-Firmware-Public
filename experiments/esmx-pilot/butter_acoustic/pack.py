#!/usr/bin/env python3
"""Prepare local R36 searches for the shared CI32 PocketSphinx decoder."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import struct


MODEL = ("feat.params", "mdef", "noisedict", "mixture_weights", "transition_matrices", "means", "variances")
ALIAS = "mantequillabb"
BUTTER = "mantequilla"
FIELDS = (
    "keywordMinimum", "onsetSlack", "articleLead", "endSlack", "tailSlack",
    "floorPerFrame", "contrastPerFrame", "oppositionKeywordMinimum",
)
TIMING_FIELDS = {"onsetSlack", "articleLead", "endSlack", "tailSlack"}


def seal(path):
    return sha256(path.read_bytes()).hexdigest()


def dictionary(path):
    entries = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        word, pronunciation = line.split(maxsplit=1)
        if word in entries:
            raise ValueError(f"Duplicate dictionary entry in {path.name}")
        entries[word] = " ".join(pronunciation.split())
    return entries


def merge(clock, butter):
    original, extra = dictionary(clock), dictionary(butter)
    if (original.get(BUTTER) != "m a_7 n t e k i Z a" or
            extra.get(BUTTER) != "m a n t e k i_7 Z a" or ALIAS in original or ALIAS in extra):
        raise ValueError("Unexpected Hora or Mantequilla pronunciation")
    extra[ALIAS] = extra.pop(BUTTER)
    for word, pronunciation in extra.items():
        if word in original and original[word] != pronunciation:
            raise ValueError(f"Dictionary collision: {word}")
        original[word] = pronunciation
    return "".join(f"{word} {pron}\n" for word, pron in sorted(original.items()))


def rewrite(path):
    return re.sub(r"\bmantequilla\b", ALIAS, path.read_text(encoding="utf-8"))


def bytes_array(name, value):
    data = value.encode("utf-8")
    rows = [", ".join(str(x) for x in data[pos:pos + 24]) for pos in range(0, len(data), 24)]
    return f"inline constexpr std::uint8_t {name}[]{{\n    " + ",\n    ".join(rows) + "\n};\n"


def raw_text(name, value):
    if ')bb37"' in value:
        raise ValueError("Unexpected raw string delimiter")
    return f'inline constexpr char {name}[] = R"bb37({value})bb37";\n'


def gate_fields(path):
    if path is None:
        return ""
    gate = json.loads(path.read_text(encoding="utf-8"))
    if set(gate) != set(FIELDS):
        raise ValueError("An experimental gate must specify each field")
    for key, value in gate.items():
        if type(value) is not int or not (-10000 <= value <= 10000):
            raise ValueError(f"Invalid gate value: {key}")
        if key in TIMING_FIELDS and not 0 <= value <= 100:
            raise ValueError(f"Invalid gate timing: {key}")
    return ".enabled = true, " + ", ".join(f".{key} = {gate[key]}" for key in FIELDS)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("clock-model", "r36-searches", "out"):
        p.add_argument("--" + flag, type=Path, required=True)
    p.add_argument("--gate", type=Path, help="explicit experimental thresholds; omit to collect evidence only")
    a = p.parse_args()
    for name in ("means", "variances"):
        tail = (a.clock_model / name).read_bytes().split(b"endhdr\n", 1)[1]
        expect = (0x11223344, 90, 1, 32, 39, 90 * 32 * 39)
        if struct.unpack_from("<6I", tail) != expect:
            raise ValueError("Expected the 32-density CI32 acoustic profile")
    sources = {name: a.r36_searches / name for name in ("butter.dict", "focus.kws", "affirm.gram", "oppose.gram")}
    clock_dict = a.clock_model / "clock.dict"
    merged = merge(clock_dict, sources["butter.dict"])
    keywords = rewrite(sources["focus.kws"])
    affirmative = rewrite(sources["affirm.gram"])
    opposing = rewrite(sources["oppose.gram"])
    settings = gate_fields(a.gate)

    a.out.mkdir(parents=True, exist_ok=False)
    for file in (*MODEL, "clock.dict"):
        (a.out / file).symlink_to((a.clock_model / file).resolve(strict=True))
    (a.out / "butter-clock.dict").write_text(merged, encoding="utf-8")
    (a.out / "butter.kws").write_text(keywords, encoding="utf-8")
    header = a.out / "ButterModelAssets.h"
    header.write_text(
        '#pragma once\n#include "Audio/EsMxPilot/ButterAcoustic.h"\n'
        "namespace EsMx::Pilot::ButterModelAssets {\n"
        + bytes_array("Dictionary", merged)
        + bytes_array("Keywords", keywords)
        + raw_text("Affirmative", affirmative)
        + raw_text("Opposing", opposing)
        + f"inline constexpr ButterGate Gate{{{settings}}};\n"
        + "}\n",
        encoding="utf-8",
    )
    receipt = {
        "ci32": {name: seal(a.clock_model / name) for name in MODEL},
        "clock_dictionary_sha256": seal(clock_dict),
        "r36": {name: seal(path) for name, path in sources.items()},
        "header_sha256": seal(header),
        "experimental_gate": json.loads(a.gate.read_text()) if a.gate else None,
    }
    (a.out / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
