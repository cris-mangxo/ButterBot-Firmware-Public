"""Replay a synthetic affirmative D6 followed by a delayed cancellation."""

from array import array
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

from capture_replay import evaluate, pcm_from


def samples(pcm):
    values = array("h")
    values.frombytes(pcm)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def without_zeros(values):
    start = next((i for i, value in enumerate(values) if value), len(values))
    end = next((i for i, value in enumerate(reversed(values)) if value), len(values))
    return values[start:len(values) - end]


def as_pcm(values):
    if sys.byteorder != "little":
        values.byteswap()
    return values.tobytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_recognizer", type=Path)
    parser.add_argument("affirmation", type=Path)
    parser.add_argument("cancellation", type=Path)
    args = parser.parse_args()
    capture = args.capture_recognizer.resolve(strict=True)
    affirmative = without_zeros(samples(pcm_from(args.affirmation)))
    cancel = without_zeros(samples(pcm_from(args.cancellation)))
    if not affirmative or not cancel:
        parser.error("synthetic control must contain a spoken waveform")
    arms = {"immediate": (), "nominal": ("--nominal-hangover",)}
    fixed = {name: evaluate(capture, as_pcm(affirmative[:]), *flags) for name, flags in arms.items()}
    cases = []
    for delay_ms in (250, 880, 950):
        quiet = array("h", [0]) * (16 * delay_ms)
        for divisor in (1, 4, 16):
            waveform = affirmative + quiet + array("h", (int(x / divisor) for x in cancel))
            for arm, flags in arms.items():
                measured = evaluate(capture, as_pcm(waveform[:]), *flags)
                cases.append({"arm": arm, "delay_ms": delay_ms, "suffix_divisor": divisor, **measured})
    print(json.dumps({
        "source": "two synthesized Ald turns, PCM16 mono at 16 kHz",
        "affirmation_sha256": sha256(args.affirmation.read_bytes()).hexdigest(),
        "cancellation_sha256": sha256(args.cancellation.read_bytes()).hexdigest(),
        "affirmative_control": fixed,
        "cancellations": cases,
    }, indent=2))
    if any(result["route"] != "dice6_factory" for result in fixed.values()):
        parser.exit(1, "the affirmative D6 control was not dispatched\n")
    if any(case["route"] != "none" for case in cases):
        parser.exit(1, "a delayed synthetic cancellation caused dispatch\n")


if __name__ == "__main__":
    main()
