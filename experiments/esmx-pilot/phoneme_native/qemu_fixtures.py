"""Export only deterministic and sealed synthetic cases to a private emulator header."""

import argparse
import json
from pathlib import Path

import torch

from compare import ATOL, RTOL, cache_cases, frozen_model, oracle, synthetic_cases
from export_bank import write_array


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("header", type=Path)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model, report = frozen_model(args.checkpoint)
    cases = [*synthetic_cases(), *cache_cases(args.cache)]
    with args.header.open("x") as out:
        out.write('#pragma once\n#include "Kernel.h"\n#include <string_view>\n'
                  'namespace EsMx::PhonemeNative::Test {\n')
        out.write(f'inline constexpr std::string_view BankSha = "{report["weights_sha256"]}";\n')
        out.write(f'inline constexpr float AbsoluteTolerance = {ATOL}f;\n'
                  f'inline constexpr float RelativeTolerance = {RTOL}f;\n')
        names = []
        for i, (name, frames) in enumerate(cases):
            expected, logits = oracle(model, frames)
            out.write(f"inline constexpr struct {{\n"
                      f"    std::array<int8_t, {frames.size}> frames;\n"
                      f"    std::array<float, {logits.size}> logits;\n"
                      f"}} Case{i}{{\n")
            write_array(out, "frames", frames.flatten(), 4)
            write_array(out, "logits", logits, 4)
            out.write("};\n")
            names.append(f'{{"{name}", Case{i}.frames, Case{i}.logits, {expected}}}')
        out.write("struct Fixture {\n"
                  "    std::string_view name;\n"
                  "    std::span<const int8_t> frames;\n"
                  "    std::span<const float> logits;\n"
                  "    uint8_t intentId;\n"
                  "};\ninline constexpr Fixture Cases[]{\n")
        out.write("".join("    " + initializer + ",\n" for initializer in names))
        out.write("};\n}\n")
    print(json.dumps({"case_count": len(cases), "weight_sha256": report["weights_sha256"]}))


if __name__ == "__main__":
    main()
