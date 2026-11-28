"""Prepare a local compile-time asset header from sealed CI32-Q8 parameters."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import struct

import numpy as np


SMALL = {
    "mdef": "Definition", "mixture_weights": "Mixture", "transition_matrices": "Transitions",
    "noisedict": "Noise", "feat.params": "Features",
}


def verified(path, expected):
    content = path.read_bytes()
    if sha256(content).hexdigest() != expected:
        raise ValueError(f"Unexpected asset fingerprint: {path.name}")
    return content


def checked_quantized(directory, file, reference, geometry):
    data = verified(directory / file, reference["model_files"][file])
    header, tail = data.split(b"endhdr\n", 1)
    prefix_len = len(header) + len(b"endhdr\n") + 24
    magic, states, streams, densities, depth, count = struct.unpack_from("<6I", tail)
    if (not header.startswith(b"s3\nversion 1.0\n") or (magic, states, streams, depth) != (0x11223344, 90, 1, 39)
            or densities not in (12, 16, 24, 32) or count != states * densities * depth):
        raise ValueError(f"Unexpected S3 encoder geometry: {file}")
    packed = verified(directory / (file + ".uint8"), geometry[file]["sha256"])
    if len(packed) != depth * 8 + count:
        raise ValueError(f"Unexpected code geometry: {file}")
    origin, scale = np.frombuffer(packed[:depth * 8], dtype="<f4").reshape(2, depth)
    code = np.frombuffer(packed[depth * 8:], dtype=np.uint8).reshape(-1, depth)
    reconstructed = (origin + scale * code).astype("<f4").tobytes()
    if not np.isfinite(origin).all() or not (np.isfinite(scale) & (scale > 0)).all():
        raise ValueError(f"Invalid quantization parameters: {file}")
    if data[prefix_len:] != reconstructed:
        raise ValueError(f"Quantized assets differ from the desktop reference: {file}")
    return data[:prefix_len], packed, densities


def write_array(file, name, content):
    file.write(f"inline constexpr uint8_t {name}[] = {{\n")
    for offset in range(0, len(content), 24):
        file.write("    " + ", ".join(str(x) for x in content[offset:offset + 24]) + ",\n")
    file.write("};\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("directory", "dictionary", "reference", "geometry", "header"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    geometry = json.loads(args.geometry.read_text())
    assets = {name: verified(args.directory / filename, reference["model_files"][filename])
              for filename, name in SMALL.items()}
    assets["Dictionary"] = verified(args.dictionary, reference["subset_sha256"])
    model_densities = set()
    for filename, name in (("means", "Means"), ("variances", "Variances")):
        prefix, packed, densities = checked_quantized(args.directory, filename, reference, geometry)
        assets[name + "Prefix"], assets[name + "Q8"] = prefix, packed
        model_densities.add(densities)
    if len(model_densities) != 1:
        raise ValueError("Means and variances use different density counts")
    with args.header.open("x") as result:
        result.write("#pragma once\n#include <cstdint>\nnamespace EsMx::Pilot::ClockModelAssets {\n")
        result.write(f"inline constexpr unsigned Densities = {densities};\n")
        for name, content in assets.items():
            write_array(result, name, content)
        result.write("}\n")
