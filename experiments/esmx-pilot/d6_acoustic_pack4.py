"""Measure four-bit storage for a generated 90-state CI32-Q8 acoustic model."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import struct

import numpy as np


def repack(src, out, mode):
    data = src.read_bytes()
    header, body = data.split(b"endhdr\n", 1)
    start = len(header) + 7 + 24
    dims = struct.unpack_from("<6I", body)
    first, states, streams, densities, depth, count = dims
    if ((first, states, streams, densities, depth) != (0x11223344, 90, 1, 32, 39)
            or count != states * densities * depth or len(body) != 24 + count * 4):
        raise ValueError(dims)
    raw = np.frombuffer(body[24:], "<f4").reshape(states, densities, depth)
    if not np.isfinite(raw).all():
        raise ValueError(f"Nonfinite S3 parameters: {src.name}")
    axis = (0, 1) if mode == "global" else 1
    lo = raw.min(axis=axis, keepdims=True)
    gap = (raw.max(axis=axis, keepdims=True) - lo) / 15.0
    gap[gap == 0] = 1
    codes = np.rint((raw - lo) / gap).clip(0, 15).astype(np.uint8)
    floats = (lo + gap * codes).astype("<f4")
    prefix = data[:start]
    out.write_bytes(prefix + floats.tobytes())
    flat = codes.reshape(-1)
    blob = lo.astype("<f4").tobytes() + gap.astype("<f4").tobytes() + (flat[::2] | (flat[1::2] << 4)).tobytes()
    out.with_suffix(".q4").write_bytes(blob)
    return {"wire_bytes": len(blob), "sha256": sha256(blob).hexdigest(),
            "max_error": float(np.abs(floats - raw).max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["global", "state"])
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir()
    metrics = {}
    for name in ("feat.params", "mdef", "mixture_weights", "transition_matrices", "noisedict"):
        shutil.copyfile(args.source / name, args.output / name)
    for name in ("means", "variances"):
        metrics[name] = repack(args.source / name, args.output / name, args.mode)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
