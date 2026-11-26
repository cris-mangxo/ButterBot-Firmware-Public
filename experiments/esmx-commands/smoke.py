"""Run all checked-in text cases and optionally synthesize and recognize the audio cases."""

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import time
import wave

from recognize import route, transcribe, transcript_from_json


def synthesize(path, text, voice, folder):
    aiff = folder / "voice.aiff"
    subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True, timeout=20)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(path)],
                   check=True, timeout=20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router", required=True, type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--voice", default="Paulina")
    parser.add_argument("--whisper", default="whisper-cli")
    args = parser.parse_args()

    router = args.router.resolve(strict=True)
    cases = json.loads(Path(__file__).with_name("cases.json").read_text("utf-8"))
    checks = [
        {"id": identifier, "expected": expected, "actual": route(router, text).get("intent")}
        for identifier, text, expected in cases["audio"] + cases["extra_text"]
    ]
    for identifier, transcript in (
        ("empty-segments", transcript_from_json({"transcription": []})),
        ("multiple-segments", transcript_from_json({"transcription": [
            {"text": "Pásame la mantequilla."}, {"text": "Mejor no."}
        ]})),
        ("long-transcript", " " * 128 + "Dime un chiste"),
    ):
        checks.append({"id": identifier, "expected": None, "actual": route(router, transcript).get("intent")})

    audio_checks = []
    if args.model is not None:
        model = args.model.resolve(strict=True)
        with tempfile.TemporaryDirectory(prefix="esmx-smoke-") as work:
            folder = Path(work)
            for identifier, text, expected in cases["audio"] + [["silence", None, None]]:
                wav = folder / f"{identifier}.wav"
                if text is None:
                    with wave.open(str(wav), "wb") as silent:
                        silent.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                        silent.writeframes(b"\0\0" * 16000)
                else:
                    synthesize(wav, text, args.voice, folder)
                    (folder / "voice.aiff").unlink()
                started = time.monotonic()
                decision = route(router, transcribe(wav, model, args.whisper))
                audio_checks.append({
                    "id": identifier, "expected": expected, "actual": decision.get("intent"),
                    "mac_inference_and_route_ms": round((time.monotonic() - started) * 1000),
                })

    print(json.dumps({"text": checks, "audio": audio_checks, "synthetic_voice": args.voice if audio_checks else None},
                     indent=2))
    # Audio mismatches make this experimental run fail and remain in its report.
    return int(any(row["expected"] != row["actual"] for row in checks + audio_checks))


if __name__ == "__main__":
    raise SystemExit(main())
