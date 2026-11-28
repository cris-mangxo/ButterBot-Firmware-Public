"""Check a frozen keyword model on seven es-MX controls and two supplementary turns."""

import argparse
import json
from pathlib import Path

from capture_replay import pcm_from
from keyword_lab import BANK_KIND, decode, digest, encode, environment, file_digest, invoke, raw_scores, read_model, recipe


CONTROLS_SHA256 = "03e23ff84f35516233d424f3ed7e4ea4690b4f5a558b46c8d93119e5eff6a244"
SUPPLEMENTARY_SHA256 = "f35dda633891910750db88b0c21316738af9db7f5d993aea6f03b81db78d3ed7"


def recordings(manifest, key, expected_hash, filename, label, split):
    if file_digest(manifest) != expected_hash:
        raise ValueError(f"{split} manifest SHA-256 mismatch")
    for row in json.loads(manifest.read_text())[key]:
        path = (manifest.parent / row[filename]).resolve(strict=True)
        if path.parent != manifest.parent:
            raise ValueError(f"invalid path in {split}")
        yield row["id"], split, label, pcm_from(path, row["wav_sha256"])


def run(args):
    config = json.loads((args.bank / "frozen.json").read_text())
    weights = args.bank / "keyword-host.npz"
    if (config["kind"] != BANK_KIND or config["recipe"] != recipe() or config["environment"] != environment()
            or config["features_sha256"] != file_digest(args.features)
            or config["weight_sha256"] != file_digest(weights)):
        raise ValueError("frozen keyword experiment does not match its manifest")
    model = read_model(weights)
    limits = {int(key): value for key, value in config["calibration"].items()}
    panels = (
        recordings(args.controls, "cases", CONTROLS_SHA256, "filename", 0, "ciempiess_esmx_control"),
        recordings(args.supplementary, "clips", SUPPLEMENTARY_SHA256, "file", 3, "fsi_es419_supplementary"),
    )
    rows = []
    for cases in panels:
        for identifier, split, label, pcm in cases:
            checksum = digest(pcm)
            if checksum in config["pcm_hash_to_adjustment_id"] or not 0 < len(pcm) <= 128_000:
                raise ValueError("public sample overlaps adjustment or exceeds four seconds")
            row = {"id": identifier, "split": split, "pcm_sha256": checksum, "expected": label}
            for mode, option in (("raw", None), ("retained_capture", "--pcm-with-hangover")):
                offered = pcm if option is None else invoke(args.capture, pcm, option)
                baseline = int(invoke(args.recognizer, offered)) if offered else 0
                candidate = decode(raw_scores(model, encode(args.features, offered)), limits)
                row[mode] = {"samples": len(offered) // 2, "baseline": baseline,
                             "candidate": candidate, "combined": baseline or candidate}
            rows.append(row)
    print(json.dumps({"weight_sha256": config["weight_sha256"], "rows": rows}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("controls", "supplementary", "bank", "features", "recognizer", "capture"):
        parser.add_argument(f"--{name}", type=lambda path: Path(path).resolve(strict=True), required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
