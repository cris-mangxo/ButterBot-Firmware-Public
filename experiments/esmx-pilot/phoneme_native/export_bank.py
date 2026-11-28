"""Export the sealed Student checkpoint as a typed C++ bank. Keep the generated header local."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


WIDTH = 28
LABELS = 32
INTENTS = 5
SCHEMA = "bb-phoneme-student-1"
BANK_BYTES = {28: 9180, 96: 50320}


def layers(width):
    return [
        ("input", (width, 20, 5)),
        *((f"blocks.{i}.{key}", (width, inputs, taps)) for i in range(3)
          for key, inputs, taps in (("temporal", 1, 5), ("channel", width, 1))),
        ("phones", (LABELS, width, 1)),
        ("intent", (INTENTS, width * 5)),
        ("negation", (1, width * 5)),
    ]


def read_bank(directory):
    report = json.loads((directory / "report.json").read_text())
    weights_path = directory / "student-int8-weights.npz"
    checksum = sha256(weights_path.read_bytes()).hexdigest()
    width = report.get("width", WIDTH)
    if (report["schema"] != SCHEMA or report["labels"] != LABELS
            or report["weights_sha256"] != checksum or type(width) is not int or width not in BANK_BYTES):
        raise ValueError("Frozen Student checkpoint does not match its report")

    ordered = {}
    names = set()
    with np.load(weights_path, allow_pickle=False) as bank:
        for layer, shape in layers(width):
            weight, scale, bias = layer + ".weight", layer + ".weight.scale", layer + ".bias"
            names.update((weight, scale, bias))
            weights, scales, biases = bank[weight], bank[scale], bank[bias]
            scale_shape = (shape[0],) + (1,) * (len(shape) - 1)
            if (weights.dtype != np.int8 or weights.shape != shape or np.min(weights) < -127
                    or scales.dtype != np.float32 or scales.shape != scale_shape
                    or biases.dtype != np.float32 or biases.shape != (shape[0],)
                    or not np.all(np.isfinite(scales)) or not np.all(scales > 0)
                    or not np.all(np.isfinite(biases))):
                raise ValueError(f"Invalid Student tensor {layer}")
            ordered[layer] = (weights.ravel(), scales.ravel(), biases)
        if len(bank.files) != len(names) or set(bank.files) != names:
            raise ValueError("Unexpected Student weight names")
    size = sum(array.nbytes for tensors in ordered.values() for array in tensors)
    if size != BANK_BYTES[width]:
        raise ValueError(f"Unexpected bank size: {size}")
    return checksum, ordered, size, width


def write_array(out, key, data, spaces):
    width = 18 if data.dtype == np.int8 else 6
    format_value = str if data.dtype == np.int8 else lambda item: float(item).hex() + "f"
    prefix = " " * spaces
    out.write(f"{prefix}.{key} = {{\n")
    for index in range(0, len(data), width):
        section = ", ".join(map(format_value, data[index:index + width]))
        out.write(f"{prefix}    {section},\n")
    out.write(f"{prefix}}},\n")


def write_layer(out, member, values, spaces):
    prefix = " " * spaces
    out.write(f"{prefix}.{member} = {{\n")
    for key, data in zip(("weights", "scales", "bias"), values, strict=True):
        write_array(out, key, data, spaces + 4)
    out.write(f"{prefix}}},\n")


def write_header(out, checksum, bank, width):
    out.write('#pragma once\n#include "Kernel.h"\n\n'
              'namespace EsMx::PhonemeNative {\n')
    out.write(f'static_assert(Width == {width}, "Native width does not match the checkpoint");\n')
    out.write(f'inline constexpr char FrozenSha256[] = "{checksum}";\n')
    out.write('inline constexpr Bank Frozen{\n')
    write_layer(out, "input", bank["input"], 4)
    out.write('    .blocks = {{\n')
    for index in range(3):
        out.write("        {\n")
        for member in ("temporal", "channel"):
            write_layer(out, member, bank[f"blocks.{index}.{member}"], 12)
        out.write("        },\n")
    out.write("    }},\n")
    for member in ("phones", "intent", "negation"):
        write_layer(out, member, bank[member], 4)
    out.write("};\n}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("header", type=Path)
    args = parser.parse_args()
    checksum, bank, size, width = read_bank(args.checkpoint)
    with args.header.open("x") as out:
        write_header(out, checksum, bank, width)
    print(json.dumps({"weight_sha256": checksum, "bank_bytes": size, "width": width}))


if __name__ == "__main__":
    main()
