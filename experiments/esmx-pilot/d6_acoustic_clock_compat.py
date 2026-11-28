"""Check acoustic-model replacement against the attested complete clock panel."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from subprocess import run

from d6_acoustic_provenance import clock_reference, model_hashes
from time_lab import file_digest


def prediction(binary, model, pcm):
    result = run([binary], input=pcm, check=True, capture_output=True, timeout=10,
                 env=dict(os.environ, BB_ESMX_CLOCK_ACOUSTIC_DIR=str(model)))
    return int(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("attested", "pcm", "binary", "old", "new", "baseline-evidence"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    reference = clock_reference(args.baseline_evidence, args.old, args.binary)
    candidate = model_hashes(args.new, args.new / "clock.dict")
    if candidate["dictionary"] != reference["files"]["dictionary"]:
        raise ValueError("Use the unchanged clock.dict alongside the expanded candidate model")
    attestation_sha = file_digest(args.attested)
    attestation = json.loads(args.attested.read_text())
    if (attestation["panel_sha256"] != reference["heroico_panel_sha256"]
            or attestation["binary_sha256"]["host"] != reference["binary"]["host"]):
        raise ValueError("Panel or test binary differs from the published reference")
    rows = []
    for case in attestation["heroico"]:
        for view, witness in case["views"].items():
            pcm = (args.pcm / f"{case['id']}.{view}.pcm").read_bytes()
            if sha256(pcm).hexdigest() != witness["pcm_sha256"]:
                raise ValueError("PCM differs from attestation")
            old = prediction(args.binary, args.old, pcm)
            if old != witness["intent"]:
                raise ValueError("Baseline differs from attestation")
            new = prediction(args.binary, args.new, pcm)
            rows.append(dict(id=case["id"], view=view, old=old, new=new))
    if len(rows) != 36:
        raise ValueError("Expected 18 sealed utterances in two views")
    print(json.dumps({"attested_replay_sha256": attestation_sha, "reference": reference, "candidate": candidate,
                      "evaluated": len(rows), "changed": [item for item in rows if item["old"] != item["new"]]},
                     indent=2))


if __name__ == "__main__":
    main()
