"""Compare a frozen compact model with eight new Ald templates."""

import argparse
import hashlib
import json
import re
from pathlib import Path
import subprocess
import unicodedata

from pilot import FEATURE_BYTES, feature, render, score, write_header

ALD_SHA256 = "6ca4b9bc22b3fa19792c212822186fa0f34cb73c1bdced6be83e4038aa946e10"
BASELINE_SHA256 = "c43bf408791c1311bdfed32fa7625899a0111b718a7b26498793b6daf4cebf16"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(text):
    folded = unicodedata.normalize("NFKD", text.casefold())
    return " ".join("".join(c if c.isalnum() else " " for c in folded
                            if not unicodedata.combining(c)).split())


def render_ald(python, model, out, identifier, text, speed):
    if text is None:
        return render(out, identifier, "", None, 0)
    original = out / f"{identifier}.original.wav"
    wav = out / f"{identifier}.wav"
    subprocess.run(
        ["/usr/bin/sandbox-exec", "-p", "(version 1) (allow default) (deny network*)",
         python, "-m", "piper", "--model", model, "--output-file", original,
         "--noise-scale", "0", "--noise-w-scale", "0", "--length-scale", str(speed),
         text],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=25,
    )
    subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
         "-i", original, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav],
        check=True, timeout=10,
    )
    return wav


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("program", "baseline", "piper-python", "ald", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("ald_cases.json"))
    args = parser.parse_args()
    if digest(args.baseline) != BASELINE_SHA256 or digest(args.ald) != ALD_SHA256:
        parser.error("El modelo compacto o Ald difiere del experimento fijado")
    program = args.program.resolve(strict=True)
    fixture = args.cases
    cases = json.loads(fixture.read_text("utf-8"))
    enrollment_texts = {canonical(text) for _, text, _ in cases["enroll"]}
    evaluation_texts = {canonical(text) for _, _, text, _ in cases["evaluate"] if text is not None}
    identifiers = [name for name, *_ in cases["evaluate"]]
    if (len(identifiers) != len(set(identifiers))
            or any(not re.fullmatch(r"[a-z][a-z0-9-]*", name)
                   or name.startswith("enroll-") for name in identifiers)):
        parser.error("ID inválido o duplicado en la evaluación")
    if enrollment_texts & evaluation_texts:
        parser.error("Frases de alta y evaluación compartidas")

    args.out.mkdir(mode=0o700)
    audio = args.out / "audio"
    audio.mkdir(mode=0o700)
    raw = args.baseline.read_bytes()
    step = 1 + FEATURE_BYTES
    templates = [(raw[i], raw[i + 1:i + step]) for i in range(0, len(raw), step)]
    if len(raw) != 16 * step or len(cases["enroll"]) != 8:
        parser.error("Se requieren 16 plantillas de base y ocho altas")
    for index, (label, text, speed) in enumerate(cases["enroll"]):
        wav = render_ald(args.piper_python, args.ald, audio, f"enroll-{index}", text, speed)
        templates.append((label, feature(program, wav)))

    candidate = args.out / "templates.bin"
    with candidate.open("xb") as target:
        for label, vector in templates:
            target.write(bytes([label]) + vector)
    write_header(args.out / "pilot_model.h", templates)

    rows = []
    for identifier, expected, text, speed in cases["evaluate"]:
        wav = render_ald(args.piper_python, args.ald, audio, identifier, text, speed)
        rows.append({
            "id": identifier, "expected": expected,
            "baseline": score(program, args.baseline, wav),
            "candidate": score(program, candidate, wav),
            "wav_sha256": digest(wav),
        })
    print(json.dumps({
        "voice": "Piper Ald es-MX", "enrollment_templates": len(templates) - 16,
        "baseline_sha256": digest(args.baseline), "candidate_sha256": digest(candidate),
        "fixture_sha256": digest(fixture), "candidate_bytes": candidate.stat().st_size,
        "synthetic_results": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
