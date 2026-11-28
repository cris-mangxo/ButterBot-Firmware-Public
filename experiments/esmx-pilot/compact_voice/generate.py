#!/usr/bin/env python3
"""Generate the local clip catalogue for the experimental Flite-free firmware."""

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import tempfile


SAMPLE_RATE = 16_000
BYTE_BUDGET = 2_000_000
GROUPS = {
    "Startup": "Hola.",
    "CameraFailure": "Falló la cámara.",
    "MotorBoardFailure": "Falló la tarjeta de los motores.",
    "ShuttingDown": "Me voy a apagar.",
    "TurningOff": "Hasta luego. Me voy a apagar.",
    "BatteryCritical": "La batería se agotó. Me voy a apagar.",
    "BatteryLow": "Me queda poca batería.",
    "BatteryCharging": "Estoy cargando.",
    "BatteryChargingFull": "Ya terminé de cargar.",
    "SettingsFrontSensorOff": "Apagué el sensor frontal.",
    "SettingsFrontSensorOn": "Encendí el sensor frontal.",
    "SettingsBottomSensorOff": "Apagué el sensor inferior.",
    "SettingsBottomSensorReduced": "Reduje la sensibilidad del sensor inferior.",
    "SettingsBottomSensorOn": "Encendí el sensor inferior.",
    "SettingsActivityOff": "Apagué mi actividad automática.",
    "SettingsActivityChill": "Me quedo en modo tranquilo.",
    "SettingsActivityActive": "Me quedo en modo activo.",
    "SettingsActivityHyper": "Me quedo en modo hiperactivo.",
    "SettingsAskActivity": "¿Qué nivel de actividad quieres? Apagado, tranquilo, activo o hiperactivo.",
    "ListeningStart": "Sí.",
    "ListeningTimeout": "No te entendí.",
    "ListenAbort": "Está bien, cancelo.",
    "EightBallListening": "Te escucho.",
    "EightBallNoQuestion": "Hazme una pregunta.",
    "EightBall": ("Sí.", "No.", "Tal vez.", "Es poco probable.", "Es muy probable.",
                  "Podría responder, pero prefiero dejarlo en duda.", "No tengo respuesta para eso."),
    "DiceRollAskCount": "¿Cuántos dados tiro?",
    "DiceRollAskType": "¿De cuántas caras quieres el dado?",
    "DiceRollCountTimeout": "No escuché cuántos dados quieres.",
    "DiceRollCountInvalid": "No entendí cuántos dados quieres.",
    "DiceRollTypeTimeout": "No escuché de cuántas caras.",
    "DiceRollTypeInvalid": "No reconozco ese tipo de dado.",
    "GasModuleMissing": "No detecto el módulo de gas.",
    "PIRModuleMissing": "No detecto el módulo de movimiento.",
    "PhoneNotConnected": "No estoy conectado a tu teléfono.",
    "PhoneNotPlaying": "No hay música sonando.",
    "FaceNoOwner": "Todavía no tengo un dueño registrado.",
    "IRListEmpty": "No tengo acciones infrarrojas guardadas.",
    "DanceStart": "Vamos a bailar.",
    "DanceInterlude": "Aquí va el siguiente paso.",
    "DanceStopped": "Ya terminé de bailar.",
    "DanceInterrupted": "Dejé de bailar.",
    "RoutineInterrupted": "Algo interrumpió mi rutina. Me detengo.",
}
SMALL = (
    "cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve",
    "diez", "once", "doce", "trece", "catorce", "quince", "dieciséis", "diecisiete",
    "dieciocho", "diecinueve", "veinte", "veintiuno", "veintidós", "veintitrés",
    "veinticuatro", "veinticinco", "veintiséis", "veintisiete", "veintiocho", "veintinueve",
)
TENS = ("treinta", "cuarenta", "cincuenta", "sesenta", "setenta", "ochenta", "noventa")
HUNDREDS = ("ciento", "doscientos", "trescientos", "cuatrocientos", "quinientos",
            "seiscientos", "setecientos", "ochocientos", "novecientos")
SINGLE = {
    "And": "y", "Thousand": "mil", "HundredExact": "cien", "Roll": "Tiro",
    "OneBeforeNoun": "un", "Die": "dado de", "Dice": "dados de", "Sides": "caras",
}
NUMBER_GROUPS = {"Small": SMALL, "Tens": TENS, "Hundreds": HUNDREDS}


def read_factory(source):
    fixed = {}
    counts = {}
    document = source.read_text(encoding="utf-8")
    mappings = dict(re.findall(r"m\[static_cast<size_t>\(Phrase::(\w+)\)\]\s*=\s*(\w+);", document))
    for name, translation in GROUPS.items():
        bank = re.search(
            rf"std::array\s+{mappings[name]}\s*=\s*std::to_array<Phrases::PhraseOutput>\(\s*\{{(.*?)\}}\s*\);",
            document, re.DOTALL,
        )
        if bank is None:
            raise ValueError(f"Missing factory bank {name}")
        variants = [ast.literal_eval(entry) for entry in
                    re.findall(r'^\s*\{\s*("(?:[^"\\]|\\.)*")\s*,', bank.group(1), re.MULTILINE)]
        translated = (translation,) * len(variants) if isinstance(translation, str) else translation
        if not variants or len(variants) != len(translated):
            raise ValueError(f"Translation count differs for {name}")
        counts[name] = len(variants)
        for old, new in zip(variants, translated, strict=True):
            if old in fixed and fixed[old] != new:
                raise ValueError(f"Ambiguous factory phrase in {name}")
            fixed[old] = new
    return fixed, counts


def render(args, texts, temp):
    subprocess.run([
        str(args.piper), "--model", str(args.model), "--output-dir", str(temp),
        "--noise-scale", "0", "--noise-w-scale", "0", "--sentence-silence", "0.02",
        "--length-scale", "0.83",
    ], input="\n".join(texts) + "\n", encoding="utf-8", check=True,
       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    waves = sorted(temp.glob("*.wav"), key=lambda path: int(path.stem))
    if len(waves) != len(texts):
        raise ValueError("Piper did not emit exactly one WAV per catalogue entry")
    audio = {}
    for text, wave in zip(texts, waves, strict=True):
        clip = subprocess.run([
            str(args.ffmpeg), "-nostdin", "-v", "error", "-i", str(wave), "-ac", "1",
            "-ar", str(SAMPLE_RATE), "-c:a", "pcm_mulaw", "-f", "mulaw", "pipe:1",
        ], capture_output=True, check=True).stdout
        if not 0 < len(clip) < 10 * SAMPLE_RATE:
            raise ValueError("Generated clip length is invalid")
        audio[text] = clip
    return audio


def write_header(path, fixed, audio):
    payload = b"".join(audio.values())
    if len(payload) > BYTE_BUDGET:
        raise ValueError(f"Generated voice is above the {BYTE_BUDGET}-byte budget")
    offsets = {}
    with path.open("x", encoding="utf-8") as header:
        header.write('#pragma once\n#include "Audio/CompactSpeech.h"\n'
                     'namespace EsMx::CompactSpeech::Bank {\ninline constexpr uint8_t Samples[]{\n')
        for begin in range(0, len(payload), 32):
            header.write("    " + ", ".join(str(value) for value in payload[begin:begin + 32]) + ",\n")
        header.write("};\n")
        offset = 0
        for index, (text, clip) in enumerate(audio.items()):
            offsets[text] = f"Clip{index}"
            header.write(f"inline constexpr Clip Clip{index}{{Samples + {offset}, {len(clip)}}};\n")
            offset += len(clip)
        for name, text in SINGLE.items():
            header.write(f"inline constexpr Clip {name} = {offsets[text]};\n")
        for name, texts in NUMBER_GROUPS.items():
            entries = ", ".join(offsets[text] for text in texts)
            header.write(f"inline constexpr Clip {name}[]{{{entries}}};\n")
        header.write("inline constexpr TextClip Fixed[]{\n")
        for old, new in fixed.items():
            header.write(f"    {{{json.dumps(old, ensure_ascii=False)}, {offsets[new]}}},\n")
        header.write("};\n}\n")
    return len(payload)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("phrases", "piper", "model", "ffmpeg", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    config_path = args.model.with_suffix(".onnx.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (args.model.name != "es_MX-ald-medium.onnx"
            or config.get("language", {}).get("code") != "es_MX"
            or config.get("num_speakers") != 1):
        parser.error("Use the single-speaker es_MX-ald-medium Piper model")
    fixed, groups = read_factory(args.phrases)
    texts = list(dict.fromkeys([*fixed.values(), *SINGLE.values(),
                               *(word for words in NUMBER_GROUPS.values() for word in words)]))
    with tempfile.TemporaryDirectory(prefix="bb-compact-voice-") as temp:
        audio = render(args, texts, Path(temp))
    args.out.mkdir(parents=True, exist_ok=False)
    header = args.out / "compact_voice_bank.h"
    total = write_header(header, fixed, audio)
    receipt = {
        "codec": "G.711 mu-law 16 kHz mono", "clip_bytes": total,
        "clips": len(audio), "factory_keys": len(fixed), "groups": groups,
        "bank_sha256": digest(header), "model_sha256": digest(args.model),
        "model_config_sha256": digest(config_path),
        "factory_sha256": digest(args.phrases),
    }
    (args.out / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: receipt[key] for key in ("clips", "clip_bytes", "factory_keys", "bank_sha256")}))


if __name__ == "__main__":
    main()
