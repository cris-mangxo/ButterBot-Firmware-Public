"""Build the local target fixtures from the sealed Heroico PCM and model report."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import struct

CASES = (
    ("T01-388", "raw", 5, "NoMatch"),
    ("T01-388", "retained", 5, "NoMatch"),
    ("T02-388", "retained", 5, "NoMatch"),
    ("T03-388", "retained", 0, "DiceNegationConflict"),
    ("T05-388", "retained", 0, "NoMatch"),
    ("T06-388", "retained", 5, "NoMatch"),
    ("C08-550", "retained", 0, "NoMatch"),
)


def prepare(panel, pcm_dir, model_report, output):
    indexed = {row["id"]: row for row in json.loads(panel.read_text())["results"]}
    model = json.loads(model_report.read_text())
    with output.open("x") as file:
        file.write('#pragma once\n#include "Audio/EsMxPilot/Recognition.h"\n#include <span>\n'
                   'namespace EsMx::Pilot {\n')
        for i, (key, view, _, _) in enumerate(CASES):
            path = pcm_dir / f"{key}.{view}.pcm"
            pcm = path.read_bytes()
            if sha256(pcm).hexdigest() != indexed[key][view]["pcm_sha256"]:
                raise ValueError(f"Unexpected sealed audio: {path.name}")
            values = struct.unpack(f"<{len(pcm) // 2}h", pcm)
            file.write(f"inline constexpr int16_t Audio{i}[] = {{\n")
            for offset in range(0, len(values), 26):
                file.write("    " + ", ".join(map(str, values[offset:offset + 26])) + ",\n")
            file.write("};\n")
        file.write("struct TargetCase {const char* name; std::span<const int16_t> samples;"
                   " unsigned intent; TurnDisposition upstream;};\n"
                   "inline constexpr TargetCase TargetCases[]{\n")
        for i, (key, view, expected, disposition) in enumerate(CASES):
            file.write(f'{{"{key}.{view}", Audio{i}, {expected}, TurnDisposition::{disposition}}},\n')
        file.write("};\n}\n")
        for asset in ("Means", "Variances"):
            digest = model["model_files"][asset.lower()]
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise ValueError(f"Invalid expected asset digest: {asset}")
            file.write(f'inline constexpr char ClockTarget{asset}Sha[] = "{digest}";\n')


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for key in ("panel", "pcm", "model-report", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    prepare(args.panel, args.pcm, args.model_report, args.output)
