"""Check the pinned local experiment headers and copy them to an external build directory."""

import argparse
import hashlib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    manifest = Path(__file__).with_name("model-assets.sha256")
    headers = {}
    for row in manifest.read_text().splitlines():
        expected, relative_path = row.split()
        source = args.experiment_root / relative_path
        contents = source.read_bytes()
        if hashlib.sha256(contents).hexdigest() != expected:
            parser.exit(1, f"SHA-256 mismatch: {relative_path}\n")
        headers[source.name] = contents

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, contents in headers.items():
        (args.output_dir / name).write_bytes(contents)


if __name__ == "__main__":
    main()
