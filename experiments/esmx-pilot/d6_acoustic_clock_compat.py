"""Check acoustic-model replacement against the attested complete clock panel."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from subprocess import run


def prediction(binary, model, pcm):
    result = run([binary], input=pcm, check=True, capture_output=True, timeout=10,
                 env=dict(os.environ, BB_ESMX_CLOCK_ACOUSTIC_DIR=str(model)))
    return int(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("attested", "pcm", "binary", "old", "new"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for case in json.loads(args.attested.read_text())["heroico"]:
        for view, witness in case["views"].items():
            pcm = (args.pcm / f"{case['id']}.{view}.pcm").read_bytes()
            if sha256(pcm).hexdigest() != witness["pcm_sha256"]:
                raise ValueError("PCM differs from attestation")
            old = prediction(args.binary, args.old, pcm)
            if old != witness["intent"]:
                raise ValueError("Baseline differs from attestation")
            new = prediction(args.binary, args.new, pcm)
            rows.append(dict(id=case["id"], view=view, old=old, new=new))
    print(json.dumps({"evaluated": len(rows), "changed": [item for item in rows if item["old"] != item["new"]]},
                     indent=2))


if __name__ == "__main__":
    main()
