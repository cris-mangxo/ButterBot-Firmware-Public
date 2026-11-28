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
VALIDATED_R37 = {
    "affirm.gram": "94a618d4214c5c76502b6b9b3126392358b87ffff9bc9c281cc3b56b2fec46d4",
    "oppose.gram": "2769232ffefc49d5b2eb2981b128ae28fbc51ba1eda2fcbe99e43aa3ef8a2903",
    "butter.dict": "b9d5d263751c19011f0a9632f527be3b513e9c7a681dc49882dae6d6bf9757b6",
    "focus.kws": "cb94341b84a3eaf3524d87f14e480ea579e37eb153f72e6851e882292eeeaeb9",
}
VALIDATED_PROFILE = {
    "feat.params": "6ad725f620b3f0c1d82bba878498630e8be4fa1f0b0bf035a7fe9ffb36f5a735",
    "mdef": "0296ecd8d4220fc6e361c599536a50b34653d95bb7b6fe30cc38d14cf4c0a5af",
    "means": "82fb3cf8b8c0655c5e6929b8ac2c9612d7a01f70a3609fc5c6158e87224ff093",
    "mixture_weights": "667fa282b96e747e38815be24ba5c4f45ed78d166f381e784f6ea28872425cae",
    "noisedict": "f9f30e70167dfb43605a1d68f86e15c0874655055e0488e34251789b668cf819",
    "transition_matrices": "c42587630dde772ef349fea9c050cdd8adcb360b919cfc460c3a18749b08ec24",
    "variances": "e504372628bcabba2b810e3af563f834a90a014ab22fb3f459d60359ce941e32",
}
VALIDATED_CLOCK_DICTIONARY = "68b218e5fbd2f381944cea5bf33b4267905bde66bec022a17d7a84ee1cf46ccf"


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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("clock-model", "r36-searches", "out"):
        p.add_argument("--" + flag, type=Path, required=True)
    p.add_argument("--validation", action="store_true", help="enable the frozen experimental R37 filter")
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
    clock_seal = seal(clock_dict)
    profile_seal = {name: seal(a.clock_model / name) for name in MODEL}
    r37_seal = {name: seal(path) for name, path in sources.items()}
    if a.validation and (profile_seal != VALIDATED_PROFILE or r37_seal != VALIDATED_R37
                         or clock_seal != VALIDATED_CLOCK_DICTIONARY):
        raise ValueError("The experimental R37 profile requires the sealed CI32-Q8 and locative grammar")

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
        + f"inline constexpr bool ValidationEnabled = {str(a.validation).lower()};\n"
        + "}\n",
        encoding="utf-8",
    )
    receipt = {
        "ci32": profile_seal,
        "clock_dictionary_sha256": clock_seal,
        "searches": r37_seal,
        "header_sha256": seal(header),
        "validation_enabled": a.validation,
    }
    (a.out / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
