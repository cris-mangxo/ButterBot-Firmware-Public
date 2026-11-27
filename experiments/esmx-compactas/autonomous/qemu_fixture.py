"""Generate test-only PCM for an ESP32-S3 QEMU build."""

import argparse
import json
from pathlib import Path
import re
import struct

from pilot import read_pcm
from public_orders import EXPANDED_FIXTURE, NATURAL_FIXTURE, ORDERED_FIXTURE, verified_cases


def load(path):
    return json.loads(path.read_text("utf-8"))["cases"]


def pick(rows, category, novel=False):
    case = next(row for row in rows if row["category"] == category
                and not row["overlong"] and row["decisions"]["word_gate"] == row["expected"]
                and (not novel or row["decisions"]["compact"] == 0 and row["expected"] == 6))
    if not re.fullmatch(r"test-[0-9]{4}(?:-[0-9])?", case["id"]):
        raise ValueError("ID de audio Ald inválido")
    return case


def new_hit(rows, reference, candidate):
    case = next((row for row in rows if not row["overlong"] and row["expected"] == 6
                 and row["decisions"].get(reference) == 0 and row["decisions"].get(candidate) == 6), None)
    if case is None:
        raise ValueError(f"El informe debe incluir una toma D6 que solo reconozca {candidate}")
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
    selected = [(case["category"], case["expected"], folder / "audio" / (case["id"] + ".wav"))
                for case, folder in selected]
    if args.guard_control:
        # The expanded panel already includes other dice. Keep the regression
        # vetoes and the legacy action within QEMU's one-megabyte app partition.
        selected[:2] = [("continuous_false_positive", 0, args.guard_control)]
    elif args.continuous_hit:
        rows, folder, reference, candidate = tests, args.corpus, "word_adapt", "word_full"
        case = new_hit(rows, reference, candidate)
        selected[0] = ("continuous_hit", 6, folder / "audio" / (case["id"] + ".wav"))
    elif args.supplement_hit or args.adapted_hit:
        rows, folder, reference, candidate = (
            (tests, args.corpus, "word_pair", "word_adapt") if args.adapted_hit
            else (replay, args.legacy, "word_gate", "word_pair")
        )
        case = new_hit(rows, reference, candidate)
        selected[0] = ("adapted_hit" if args.adapted_hit else "supplement_hit", 6, folder / "audio" / (case["id"] + ".wav"))
    if args.public_dir:
        fixture = (NATURAL_FIXTURE if args.natural_public else ORDERED_FIXTURE if args.ordered_public
                   else EXPANDED_FIXTURE if args.expanded_public else None)
        public = [(case["id"], case["expected"], wav)
                  for case, wav in verified_cases(args.public_dir, fixture)]
        if args.public_id:
            identifiers = set(args.public_id)
            if not identifiers <= {name for name, _, _ in public}:
                raise ValueError("Un ID no pertenece al panel humano")
            public = [row for row in public if row[0] in identifiers]
        selected += public
    with args.header.open("x", encoding="utf-8") as output:
        output.write('#pragma once\n#include <cstdint>\n#include <span>\n')
        output.write('struct PcmCase { const char* name; int expected; std::span<const int16_t> samples; };\n')
        for index, (_, _, wav) in enumerate(selected):
            raw = read_pcm(wav)
            samples = struct.unpack(f"<{len(raw) // 2}h", raw)
            output.write(f'inline constexpr int16_t Turn{index}[] = {{\n')
            for start in range(0, len(samples), 24):
                output.write("    " + ", ".join(map(str, samples[start:start + 24])) + ",\n")
            output.write("};\n")
        output.write("inline constexpr PcmCase WordPcmCases[] = {\n")
        for index, (name, expected, _) in enumerate(selected):
            output.write(f'    {{"{name}", {expected}, Turn{index}}},\n')
        output.write("};\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("test", "replay", "corpus", "legacy", "older-wide", "header"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--public-dir", type=Path)
    panel = parser.add_mutually_exclusive_group()
    panel.add_argument("--expanded-public", action="store_true")
    panel.add_argument("--ordered-public", action="store_true")
    panel.add_argument("--natural-public", action="store_true")
    parser.add_argument("--public-id", action="append", help="Include only the specified public cases")
    positive = parser.add_mutually_exclusive_group()
    positive.add_argument("--supplement-hit", action="store_true")
    positive.add_argument("--adapted-hit", action="store_true")
    positive.add_argument("--continuous-hit", action="store_true")
    positive.add_argument("--guard-control", type=Path)
    args = parser.parse_args()
    if (args.expanded_public or args.ordered_public or args.natural_public or args.public_id) and not args.public_dir:
        parser.error("Seleccionar tomas humanas requiere --public-dir")
    write(args)


if __name__ == "__main__":
    main()
