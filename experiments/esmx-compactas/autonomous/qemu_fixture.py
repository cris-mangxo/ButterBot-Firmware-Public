"""Generate test-only Ald PCM for an ESP32-S3 QEMU build."""

import argparse
import json
from pathlib import Path
import re
import struct

from pilot import read_pcm


def load(path):
    return json.loads(path.read_text("utf-8"))["cases"]


def pick(rows, category, novel=False):
    case = next(row for row in rows if row["category"] == category
                and not row["overlong"] and row["decisions"]["word_gate"] == row["expected"]
                and (not novel or row["decisions"]["compact"] == 0 and row["expected"] == 6))
    if not re.fullmatch(r"test-[0-9]{4}(?:-[0-9])?", case["id"]):
        raise ValueError("ID de audio Ald inválido")
    return case


def write(args):
    tests = load(args.test)
    replay = load(args.replay)
    selected = [(pick(tests, category, novel=(category == "positive")), args.corpus)
                for category in ("positive", "other_die", "negated")]
    selected += [(pick(replay, "legacy"), args.legacy),
                 (pick(replay, "original_fact_veto"), args.older_wide)]
    with args.header.open("x", encoding="utf-8") as output:
        output.write('#pragma once\n#include <cstdint>\n#include <span>\n')
        output.write('struct PcmCase { const char* name; int expected; std::span<const int16_t> samples; };\n')
        for index, (case, folder) in enumerate(selected):
            raw = read_pcm(folder / "audio" / (case["id"] + ".wav"))
            samples = struct.unpack(f"<{len(raw) // 2}h", raw)
            output.write(f'inline constexpr int16_t Turn{index}[] = {{\n')
            for start in range(0, len(samples), 24):
                output.write("    " + ", ".join(map(str, samples[start:start + 24])) + ",\n")
            output.write("};\n")
        output.write("inline constexpr PcmCase WordPcmCases[] = {\n")
        for index, (case, _) in enumerate(selected):
            output.write(f'    {{"{case["category"]}", {case["expected"]}, Turn{index}}},\n')
        output.write("};\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("test", "replay", "corpus", "legacy", "older-wide", "header"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    write(parser.parse_args())


if __name__ == "__main__":
    main()
