"""Compare fixed D6 decisions with reduced evidence from the production algorithms."""

import argparse
from hashlib import file_digest, sha256
import json
from pathlib import Path
import subprocess

from capture_replay import evaluate, pcm_from, recordings


CAPTURE_CASES = {"public-03", "public-37", "public-81", "public-88", "public-89",
                 "test-0379", "d6"}


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def check(stages, pcm, expected):
    trace = evaluate(stages, pcm)
    if trace["final"] != expected:
        raise ValueError("stage probe diverged from the independent pilot executable")
    return trace


def captured(stages, capture, pcm, retain):
    mode = ("--nominal-hangover",) if retain else ()
    verdict = evaluate(capture, pcm, *mode)
    result = {"observed": verdict["observed"], "reason": verdict["reason"]}
    if verdict["reason"] == "classified":
        mode = "--pcm-with-hangover" if retain else "--pcm"
        captured_pcm = subprocess.check_output([capture, mode], input=pcm, timeout=10)
        if not captured_pcm or len(captured_pcm) % 2:
            raise ValueError("probe returned invalid PCM")
        result["pcm_sha256"] = sha256(captured_pcm).hexdigest()
        result["stages"] = check(stages, captured_pcm, verdict["observed"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stages", "recognizer", "capture", "experiment-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for name, split, category, expected, wav, pin in recordings(args.experiment_root):
        pcm = pcm_from(wav, pin)
        stages = check(args.stages, pcm, evaluate(args.recognizer, pcm))
        row = {"id": name, "split": split, "category": category, "expected": expected,
               "wav_sha256": digest(wav), "raw": stages}
        if name in CAPTURE_CASES:
            row["immediate"] = captured(args.stages, args.capture, pcm, False)
            row["hold"] = captured(args.stages, args.capture, pcm, True)
        rows.append(row)
    coverage = {row["id"] for row in rows if "hold" in row}
    if coverage != CAPTURE_CASES:
        raise ValueError("panel is missing a required capture case")
    print(json.dumps({
        "executables": {name: digest(getattr(args, name))
                        for name in ("stages", "recognizer", "capture")},
        "rows": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
