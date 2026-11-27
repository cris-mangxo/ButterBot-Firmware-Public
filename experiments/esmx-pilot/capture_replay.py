#!/usr/bin/env python3
"""Compare raw PCM inference and the actual pilot capture with a simulated VAD."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import wave

from replay import PUBLIC_PANELS, panel


DEVELOPMENT_D6 = {"public-03", "public-37", "public-61", "public-81"}
RESERVED_D6 = {"public-88", "public-89"}
ALIASES = {
    "butter": 1, "joke": 3, "time": 5, "d6": 6, "veto-d6": 0, "veto-fact": 0,
}


def recordings(base):
    for index, folder in enumerate(PUBLIC_PANELS):
        for case in panel(base, index):
            if case["overlong"]:
                continue
            name = case["id"]
            split = "public"
            if name in DEVELOPMENT_D6:
                split = "public_development"
            elif name in RESERVED_D6:
                split = "public_reserved_d6"
            yield (name, split, case["category"], case["expected"],
                   base / folder / f"{name}.wav", None)

    reports = base / "autonomous-v18" / "native-replay.json"
    for case in json.loads(reports.read_text())["cases"]:
        if case["overlong"]:
            continue
        name = case["id"]
        folder = "d6-round1" if case["category"] == "original_fact_veto" else "phrases-final"
        yield (name, "synthetic_panel", case["category"], case["expected"],
               base / folder / "audio" / f"{name}.wav", None)

    for name, expected in ALIASES.items():
        path = base / "primer-candidato" / "synthetic" / f"{name}.wav"
        yield name, "synthetic_aliases", "alias", expected, path, None

    voices = base / "primer-candidato-voces"
    for case in json.loads((voices / "manifest.json").read_text())["clips"]:
        if case["locale"] != "es-MX" or case["split"] != "heldout" or case["intent"] != "hora":
            raise ValueError(f"unexpected reserved voice metadata for {case['id']}")
        yield (case["id"], "public_reserved_time", "time", ALIASES["time"],
               voices / case["file"], case["wav_sha256"])


def pcm_from(wav, expected_hash=None):
    if expected_hash is not None and sha256(wav.read_bytes()).hexdigest() != expected_hash:
        raise ValueError(f"{wav.name}: does not match the pinned recording")
    with wave.open(str(wav)) as recording:
        metadata = (recording.getframerate(), recording.getnchannels(),
                    recording.getsampwidth(), recording.getcomptype())
        if metadata != (16000, 1, 2, "NONE"):
            raise ValueError(f"{wav.name}: expected 16 kHz mono PCM16")
        return recording.readframes(recording.getnframes())


def evaluate(executable, pcm, *arguments):
    verdict = subprocess.run([executable, *arguments], input=pcm, stdout=subprocess.PIPE,
                             check=True, timeout=10)
    return json.loads(verdict.stdout)


def run(raw, capture, base):
    rows = []
    for name, split, category, expected, wav, digest in recordings(base):
        pcm = pcm_from(wav, digest)
        rows.append({"id": name, "split": split, "category": category, "expected": expected,
                     "raw": evaluate(raw, pcm), **evaluate(capture, pcm),
                     "nominal_hold": evaluate(capture, pcm, "--nominal-hangover")["observed"]})
    d6_pcm = pcm_from(base / "primer-candidato" / "synthetic" / "d6.wav")
    interruption = evaluate(capture, d6_pcm, "--interrupted")
    return rows, interruption


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_recognizer", type=Path)
    parser.add_argument("capture_recognizer", type=Path)
    parser.add_argument("experiment_root", type=Path)
    args = parser.parse_args()
    rows, interruption = run(args.raw_recognizer.resolve(strict=True),
                             args.capture_recognizer.resolve(strict=True), args.experiment_root)
    summary = {
        "capture": "production PilotCapture and recognizer with host FreeRTOS stubs",
        "input": "16 kHz mono PCM16, fixed 20 ms frames, RMS gate 180, up to 240 ms preroll",
        "nominal_hold": "same RMS gate with 1000 ms of simulated VAD hold; not ESP-SR output",
        "cases": rows,
        "interrupted_synthetic_d6": interruption,
    }
    print(json.dumps(summary, indent=2))

    synthetic = {row["id"]: row for row in rows if row["split"] == "synthetic_aliases"}
    if synthetic["d6"]["raw"] != ALIASES["d6"] or synthetic["d6"]["observed"] != ALIASES["d6"]:
        parser.exit(1, "the positive D6 control did not establish the interrupted prefix\n")
    if any(synthetic[name][mode] != 0 for name in ("veto-d6", "veto-fact")
           for mode in ("observed", "nominal_hold")):
        parser.exit(1, "a complete negated synthetic order activated\n")
    if interruption["observed"] != 0 or not interruption["interrupted"]:
        parser.exit(1, "the interrupted positive prefix was not safely discarded\n")


if __name__ == "__main__":
    main()
