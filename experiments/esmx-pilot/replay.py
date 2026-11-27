"""Replay existing local panels through the same recognizer compiled into the pilot."""

import argparse
import json
from pathlib import Path
import subprocess
import wave


PUBLIC_PANELS = (
    "autonomous-v2/public-holdout", "autonomous-v5/expanded-test", "autonomous-v6/ordered-complete",
    "autonomous-v7/natural-public-final", "autonomous-v8/esmx-publish", "autonomous-v9/holdout-direct",
    "autonomous-v10/heldout-final", "autonomous-v11/new-panel", "autonomous-v12/table-panel",
    "autonomous-v13/mention-panel", "autonomous-v14/board-games-panel", "autonomous-v15/full-turns-panel",
)


def evaluate(executable, wav):
    with wave.open(str(wav)) as recording:
        if (recording.getframerate(), recording.getnchannels(), recording.getsampwidth(),
                recording.getcomptype()) != (16000, 1, 2, "NONE"):
            raise ValueError(f"{wav.name}: expected 16 kHz mono PCM16")
        samples = recording.readframes(recording.getnframes())
    invocation = subprocess.run([executable], input=samples, stdout=subprocess.PIPE, check=True, timeout=10)
    return int(invocation.stdout)


def panel(base, index):
    report = base / "autonomous-v18" / f"native-public-{index}.json"
    return json.loads(report.read_text())["cases"]


def replay(executable, base):
    results = []
    for index, folder in enumerate(PUBLIC_PANELS):
        for previous in panel(base, index):
            if previous["overlong"]:
                continue
            name = previous["id"]
            path = base / folder / f"{name}.wav"
            results.append({
                "id": name, "panel": "public", "category": previous["category"],
                "expected": previous["expected"], "observed": evaluate(executable, path),
            })

    report = base / "autonomous-v18" / "native-replay.json"
    for previous in json.loads(report.read_text())["cases"]:
        if previous["overlong"]:
            continue
        name = previous["id"]
        folder = "d6-round1" if previous["category"] == "original_fact_veto" else "phrases-final"
        path = base / folder / "audio" / f"{name}.wav"
        results.append({
            "id": name, "panel": "synthetic", "category": previous["category"],
            "expected": previous["expected"], "observed": evaluate(executable, path),
        })
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recognizer", type=Path)
    parser.add_argument("experiment_root", type=Path)
    args = parser.parse_args()
    results = replay(args.recognizer.resolve(strict=True), args.experiment_root)
    print(json.dumps(results, indent=2))
    indexed = {(row["panel"], row["id"]): row["observed"] for row in results}
    required = {("public", "public-03"): 6, ("public", "public-37"): 0,
                ("public", "public-81"): 6, ("synthetic", "test-0379"): 0}
    if any(indexed.get(case) != label for case, label in required.items()):
        parser.exit(1, "required regression failed\n")
    if any(row["panel"] == "public" and row["expected"] == 0 and row["observed"] != 0
           for row in results):
        parser.exit(1, "public negative control activated\n")


if __name__ == "__main__":
    main()
