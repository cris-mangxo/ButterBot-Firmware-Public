#!/usr/bin/env python3
"""Build local, verified PCM fixtures for the ESP32-S3 recognition bench."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import struct

DISPOSITIONS = {
    "Accepted", "NoMatch", "InvalidAudio", "FactVeto", "UnconfirmedDice",
    "PhraseRejected", "DiceNegationVeto", "DiceNegationConflict", "DiceFollowUp",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("cases", "model", "header"):
        parser.add_argument("--" + key, required=True, type=Path)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text())
    model = json.loads(args.model.read_text())["ci32"]
    if not cases or len({row["id"] for row in cases}) != len(cases):
        raise ValueError("Empty or duplicate fixture IDs")
    for name in ("means", "variances"):
        if not re.fullmatch(r"[a-f0-9]{64}", model[name]):
            raise ValueError("Invalid model digest")
    rows = []
    with args.header.open("x") as out:
        out.write('#pragma once\n#include "Audio/EsMxPilot/Recognition.h"\n#include <span>\n'
                  'namespace EsMx::Pilot {\n')
        for index, case in enumerate(cases):
            if (not re.fullmatch(r"[A-Za-z0-9_.-]+", case["id"])
                    or case["intent"] not in (0, 1, 3, 5, 6) or case["upstream"] not in DISPOSITIONS):
                raise ValueError("Invalid case descriptor")
            pcm = (args.cases.parent / case["pcm"]).read_bytes()
            if not 0 < len(pcm) <= 128_000 or len(pcm) % 2 or sha256(pcm).hexdigest() != case["sha256"]:
                raise ValueError(f"Unexpected PCM: {case['id']}")
            samples = struct.unpack(f"<{len(pcm) // 2}h", pcm)
            out.write(f"inline constexpr int16_t Audio{index}[] = {{\n")
            for offset in range(0, len(samples), 24):
                line = ", ".join(map(str, samples[offset:offset + 24]))
                out.write("    " + line + ",\n")
            out.write("};\n")
            rows.append(f'{{"{case["id"]}", Audio{index}, {case["intent"]}, TurnDisposition::{case["upstream"]}}}')
        out.write("struct TargetCase {const char* name; std::span<const int16_t> samples; "
                  "unsigned intent; TurnDisposition upstream;};\ninline constexpr TargetCase TargetCases[]{\n")
        out.write(",\n".join(rows))
        out.write("\n};\n}\n")
        out.write(f'inline constexpr char ClockTargetMeansSha[] = "{model["means"]}";\n')
        out.write(f'inline constexpr char ClockTargetVariancesSha[] = "{model["variances"]}";\n')


if __name__ == "__main__":
    main()
