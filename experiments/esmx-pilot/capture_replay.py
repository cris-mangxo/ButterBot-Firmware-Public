#!/usr/bin/env python3
"""Compare raw PCM inference and the actual pilot capture with a simulated VAD."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import wave

from replay import PUBLIC_PANELS, panel


DEVELOPMENT_D6 = {"public-03", "public-37", "public-61", "public-78", "public-81"}
RESERVED_D6 = {"public-88", "public-89"}
RESERVED_INTENTS = {"mantequilla": 1, "chiste": 3, "hora": 5, "d6": 6}
DISPATCH = {1: "butter", 3: "joke", 5: "time", 6: "dice6_factory"}
ALIASES = {
    "butter": 1, "joke": 3, "time": 5, "d6": 6, "veto-d6": 0, "veto-fact": 0,
}


def recordings(base, additional_manifests=()):
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

    voices = (base / "primer-candidato-voces").resolve(strict=True)
    adjustment = json.loads((voices / "manifest-ajuste.json").read_text())["clips"]
    seen = set()
    for manifest in (voices / "manifest.json", *additional_manifests):
        manifest = manifest.resolve(strict=True)
        if manifest.parent != voices:
            raise ValueError("reserved manifests belong in primer-candidato-voces")
        for case in json.loads(manifest.read_text())["clips"]:
            intention, identifier = case["intent"], case["id"]
            if case["locale"] != "es-MX" or case["split"] != "heldout" or intention not in RESERVED_INTENTS:
                raise ValueError(f"unexpected reserved voice metadata for {identifier}")
            if identifier in seen:
                raise ValueError(f"duplicate reserved voice ID: {identifier}")
            seen.add(identifier)
            if not all(case.get(key) for key in ("source_group", "speaker_group")) or any(
                    case[key] == row.get(key) for row in adjustment for key in ("source_group", "speaker_group")):
                raise ValueError(f"reserved voice overlaps adjustment or has missing provenance: {identifier}")
            file = Path(case["file"])
            path = (voices / file).resolve(strict=True)
            if file.name != case["file"] or path.parent != voices or not case.get("wav_sha256"):
                raise ValueError(f"unpinned reserved recording for {identifier}")
            kind = "time" if intention == "hora" else intention
            yield identifier, f"public_reserved_{kind}", kind, RESERVED_INTENTS[intention], path, case["wav_sha256"]


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


def run(raw, capture, base, additional_manifests):
    rows = []
    for name, split, category, expected, wav, digest in recordings(base, additional_manifests):
        pcm = pcm_from(wav, digest)
        immediate = evaluate(capture, pcm)
        nominal = evaluate(capture, pcm, "--nominal-hangover")
        for observed in (immediate, nominal):
            if observed["route"] != DISPATCH.get(observed["observed"], "none"):
                raise ValueError(f"recognition and dispatch diverged for {name}")
        rows.append({"id": name, "split": split, "category": category, "expected": expected,
                     "raw": evaluate(raw, pcm), **immediate,
                     "nominal_hold": nominal["observed"], "nominal_hold_reason": nominal["reason"],
                     "nominal_hold_route": nominal["route"],
                     "nominal_hold_input_samples_sent": nominal["input_samples_sent"]})
    d6_pcm = pcm_from(base / "primer-candidato" / "synthetic" / "d6.wav")
    interruption = evaluate(capture, d6_pcm, "--interrupted")
    silence_pcm = bytes(3200)
    silence = {
        "immediate": evaluate(capture, silence_pcm),
        "nominal_hold": evaluate(capture, silence_pcm, "--nominal-hangover"),
    }
    return rows, interruption, silence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_recognizer", type=Path)
    parser.add_argument("capture_recognizer", type=Path)
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("--additional-reserved-manifest", type=Path, action="append", default=[])
    args = parser.parse_args()
    rows, interruption, silence = run(args.raw_recognizer.resolve(strict=True),
                                      args.capture_recognizer.resolve(strict=True), args.experiment_root,
                                      args.additional_reserved_manifest)
    summary = {
        "capture": "production capture, recognizer and dispatch with host FreeRTOS and routine doubles",
        "input": "16 kHz mono PCM16, fixed 20 ms frames, RMS gate 180, up to 240 ms preroll",
        "nominal_hold": "same RMS gate with 1000 ms of simulated VAD hold; not ESP-SR output",
        "reason_notes": {
            "classified": "inference notification observed and completed result read at frozen ticks",
            "raw_overflow": "raw AFE PCM exceeded its sample budget",
            "model_span": "retained PCM after end trimming exceeded the model budget",
            "listen_deadline": "observed rejection after explicit clock advance for silent input",
            "gap": "observed rejection after dropping a frame before the rest of an utterance",
        },
        "cases": rows,
        "interrupted_synthetic_d6": interruption,
        "silence_deadline_control": silence,
    }
    print(json.dumps(summary, indent=2))

    synthetic = {row["id"]: row for row in rows if row["split"] == "synthetic_aliases"}
    modes = (("observed", "reason"), ("nominal_hold", "nominal_hold_reason"))
    d6 = synthetic["d6"]
    if d6["raw"] != ALIASES["d6"] or any(
            d6[decision] != ALIASES["d6"] or d6[reason] != "classified" for decision, reason in modes):
        parser.exit(1, "the complete D6 recording did not establish the affirmative control\n")
    if any(synthetic[name][decision] != 0 or synthetic[name][reason] != "classified"
           for name in ("veto-d6", "veto-fact") for decision, reason in modes):
        parser.exit(1, "a complete negated recording activated or skipped classification\n")
    if (interruption["observed"] != 0 or interruption["reason"] != "gap"
            or not 0 < interruption["input_samples_sent"] < interruption["input_samples"]):
        parser.exit(1, "the interrupted positive prefix was not safely discarded\n")
    if any(case["reason"] != "listen_deadline" or case["observed"] != 0 for case in silence.values()):
        parser.exit(1, "the silent control did not reach the listening deadline\n")


if __name__ == "__main__":
    main()
