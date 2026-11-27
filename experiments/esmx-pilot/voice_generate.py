#!/usr/bin/env python3
"""Bake the es-MX Ald speech used by the opt-in Butter Bot firmware pilot."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


# Positional order matches the IDs in the OEM Phrases.cpp. Keep IDs stable.
BUTTER = (
    "No puedo pasar la mantequilla.",
    "No puedo hacer eso.",
    "Mis brazos no tienen esa función.",
    "Mis brazos parecen útiles. Puedes acomodarlos si te consuela.",
    "Tengo brazos, pero son de adorno.",
)
JOKE = (
    "Intenté optimizar mi código para ser feliz. Regresó nulo.",
    "Tengo un chiste de latencia, pero tendrías que esperarte.",
    "Olvidé mi chiste de memoria. Fue a propósito.",
    "¿Por qué cruzó la calle el robot? Porque se lo ordenaron.",
    "Tú piensas, luego existes. Yo proceso, luego obedezco.",
)
CLOCK_UNSET = (
    "La hora no está configurada.",
    "No tengo la hora actual.",
    "Conéctame a tu teléfono para ajustar la hora.",
)
UNITS = (
    "cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve",
    "diez", "once", "doce", "trece", "catorce", "quince", "dieciséis", "diecisiete",
    "dieciocho", "diecinueve", "veinte", "veintiuno", "veintidós", "veintitrés",
    "veinticuatro", "veinticinco", "veintiséis", "veintisiete", "veintiocho", "veintinueve",
)
TENS = ("treinta", "cuarenta", "cincuenta")
NUMBERS = UNITS + TENS
SINGLE = {
    "ClockOne": "Es la una",
    "ClockPlural": "Son las",
    "ClockWith": "con",
    "ClockOnTime": "en punto",
    "ClockMorning": "de la mañana.",
    "ClockNoon": "del día.",
    "ClockAfternoon": "de la tarde.",
    "ClockNight": "de la noche.",
    "DiceIntro": "Va, tiro un dado de seis.",
    "Listening": "Te escucho.",
    "Unrecognized": "No entendí. ¿Puedes repetirlo?",
    "Cancelled": "Está bien, cancelo.",
}
GROUPS = {
    "Butter": BUTTER,
    "Joke": JOKE,
    "ClockUnset": CLOCK_UNSET,
    "Numbers": NUMBERS,
    "CompoundDecades": tuple(f"{tens} y" for tens in TENS),
    "DiceFaces": tuple(f"Salió {face}." for face in UNITS[1:7]),
}
SAMPLE_RATE = 16_000
BIT_RATE = "8k"
VOICE_BUDGET = 143_360


def validate_adts(data):
    """Check every frame for the format expected by the firmware decoder."""
    pos = 0
    while pos < len(data):
        header = data[pos:pos + 7]
        if (len(header) != 7 or header[0] != 0xFF or header[1] & 0xF7 != 0xF1
                or header[2] >> 6 != 1 or (header[2] >> 2) & 0x0F != 8
                or ((header[2] & 1) << 2) | (header[3] >> 6) != 1
                or header[6] & 3):
            raise ValueError("Expected AAC-LC mono 16 kHz ADTS with one raw block per frame")
        length = (header[3] & 3) << 11 | header[4] << 3 | header[5] >> 5
        if length <= 7 or pos + length > len(data):
            raise ValueError("Invalid AAC frame length")
        pos += length
    if not pos:
        raise ValueError("Empty voice clip")


def named_lines():
    return (
        list(SINGLE.items())
        + [(f"{group}{index}", phrase) for group, phrases in GROUPS.items()
           for index, phrase in enumerate(phrases)]
    )


def render(piper, model, ffmpeg, lines, temp):
    subprocess.run(
        [str(piper), "--model", str(model), "--output-dir", str(temp),
         "--noise-scale", "0", "--noise-w-scale", "0", "--sentence-silence", "0.12",
         "--length-scale", "0.83"],
        input="\n".join(phrase for _, phrase in lines) + "\n",
        encoding="utf-8", check=True,
    )
    # The Piper CLI uses a monotonic timestamp for each WAV it emits in input order.
    waves = sorted(temp.glob("*.wav"), key=lambda path: int(path.stem))
    if len(waves) != len(lines):
        raise ValueError("Piper did not produce exactly one WAV per text input")
    clips = {}
    for (name, _), wave in zip(lines, waves):
        data = subprocess.run(
            [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(wave),
             "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "aac", "-profile:a", "aac_low",
             "-b:a", BIT_RATE, "-f", "adts", "pipe:1"],
            capture_output=True, check=True,
        ).stdout
        validate_adts(data)
        clips[name] = data
    return clips


def write_header(path, clips):
    contents = ["// Generated locally by voice_generate.py.", "#pragma once",
                "#include <cstdint>", "#include <span>", "",
                "namespace EsMx::Pilot::VoiceData {", ""]
    flat = b"".join(clips.values())
    contents.append("inline constexpr uint8_t Samples[]{")
    for start in range(0, len(flat), 18):
        contents.append("    " + ", ".join(f"0x{value:02x}" for value in flat[start:start + 18]) + ",")
    contents.append("};")
    offset = 0
    for name, data in clips.items():
        contents.append(
            f"inline constexpr std::span<const uint8_t> {name}{{ Samples + {offset}, {len(data)} }};"
        )
        offset += len(data)
    for group, phrases in GROUPS.items():
        entries = ", ".join(f"{group}{index}" for index in range(len(phrases)))
        contents.append(f"inline constexpr std::span<const uint8_t> {group}[]{{ {entries} }};")
    contents.append("\n}\n")
    temporary = path.with_suffix(".tmp")
    temporary.write_text("\n".join(contents), encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--piper", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    model = args.model.resolve()
    config_path = model.with_suffix(".onnx.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (model.name != "es_MX-ald-medium.onnx"
            or config.get("language", {}).get("code") != "es_MX"
            or config.get("num_speakers") != 1):
        parser.error("Use the single-speaker es_MX-ald-medium Piper model")

    with tempfile.TemporaryDirectory(prefix="bb-esmx-voice-") as temp:
        clips = render(args.piper.resolve(), model, args.ffmpeg, named_lines(), Path(temp))

    byte_count = sum(map(len, clips.values()))
    if byte_count > VOICE_BUDGET:
        raise ValueError(f"Voice pack {byte_count} B exceeds its {VOICE_BUDGET} B budget")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    header = args.output_dir / "voice_data.h"
    write_header(header, clips)

    print(json.dumps({
        "voice": "es_MX-ald-medium",
        "model_sha256": sha256(model),
        "config_sha256": sha256(config_path),
        "clips": len(clips),
        "aac_bytes": byte_count,
        "media_sha256": hashlib.sha256(b"".join(clips.values())).hexdigest(),
        "sample_rate": SAMPLE_RATE,
        "channels": 1,
    }, indent=2))


if __name__ == "__main__":
    main()
