"""Reproduce a compact template smoke test with synthetic Spanish voices."""

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import wave

FEATURE_BYTES = 480
MAX_COST = 6200
MIN_GAP = 600
TRAIN_VOICES = ["Paulina", "Eddy (Español (México))"]
DEV_VOICE = "Shelley (Español (México))"
CHECK_VOICE = "Flo (Español (México))"


def read_pcm(wav):
    with wave.open(str(wav), "rb") as source:
        if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (1, 2, 16000):
            raise ValueError("Se requiere WAV PCM mono de 16 bits y 16 kHz")
        frames = source.getnframes()
        if not 0 < frames <= 64000:
            raise ValueError("Duración fuera de rango")
        payload = source.readframes(frames)
        if len(payload) != frames * 2:
            raise ValueError("WAV incompleto")
        return payload


def feature(program, wav):
    result = subprocess.run([program, "features"], input=read_pcm(wav), capture_output=True, timeout=10)
    result.check_returncode()
    if len(result.stdout) != FEATURE_BYTES:
        raise ValueError("Firma acústica incompleta")
    return result.stdout


def score(program, model, wav):
    result = subprocess.run([program, "score", model, str(MAX_COST), str(MIN_GAP)],
                            input=read_pcm(wav), capture_output=True, timeout=10)
    if result.returncode == 3:
        return {"label":0, "no_feature":True}
    result.check_returncode()
    return json.loads(result.stdout)


def render(folder, name, voice, text, rate):
    wav = folder / f"{name}.wav"
    if text is None:
        with wave.open(str(wav), "wb") as target:
            target.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            target.writeframes(bytes(32000))
        return wav
    aiff = folder / f"{name}.aiff"
    subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", aiff, text], check=True, timeout=20)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", aiff, wav], check=True, timeout=15)
    return wav


def write_header(path, templates):
    with path.open("x", encoding="utf-8") as header:
        header.write('#pragma once\n#include "Compact.h"\n\nusing namespace EsMx::Kws;\n')
        header.write('inline constexpr Prototype PilotModel[] = {\n')
        for label, features in templates:
            ints = [i if i < 128 else i - 256 for i in features]
            header.write(f'    {{static_cast<Intent>({label}), Signature{{.frames = {{{{\n')
            for start in range(0, FEATURE_BYTES, 20):
                header.write('        {' + ', '.join(map(str, ints[start:start+20])) + '},\n')
            header.write('    }}}},\n')
        header.write(f'}};\ninline constexpr Limits PilotLimits{{{MAX_COST}, {MIN_GAP}}};\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--negative-dir", type=Path,
                        help="WAV públicos opcionales previamente verificados como ajenos al vocabulario")
    args = parser.parse_args()
    program = args.program.resolve(strict=True)
    args.out.mkdir(mode=0o700)
    model = args.out / "templates.bin"
    prompts = json.loads(Path(__file__).with_name("prompts.json").read_text("utf-8"))
    with tempfile.TemporaryDirectory(prefix="compact-kws-") as scratch:
        folder = Path(scratch)
        templates = []
        for speaker, voice in enumerate(TRAIN_VOICES):
            for index, (label, text) in enumerate(prompts):
                if text is not None:
                    wav = render(folder, f"train-{speaker}-{index}", voice, text, 175)
                    templates.append((label, feature(program, wav)))
        with model.open("xb") as output:
            for label, features in templates:
                output.write(bytes([label]) + features)
        write_header(args.out / "pilot_model.h", templates)

        rows = []
        # Dev uses a different voice, speed and phrasings for the negative controls.
        dev_prompts = prompts[:6] + [
            [0,"No quiero que me pases la mantequilla."],
            [0,"Dime dónde está la salida."],
            [0,"No me digas la hora."],
            [0,None],
        ]
        for index, (expected, text) in enumerate(dev_prompts):
            wav = render(folder, f"dev-{index}", DEV_VOICE, text, 155)
            rows.append({"id":index, "expected":expected, "decision":score(program, model, wav)})
        check = []
        check_prompts = [
            [1,"Dame la mantequilla."],
            [2,"Tú pasa la mantequilla."],
            [3,"Échame un chiste."],
            [4,"Dime un dato curioso."],
            [5,"Dime la hora."],
            [6,"Tira un dado de seis."],
            [0,"No me digas ningún chiste."],
            [0,"La puerta de la casa está cerrada."],
            [0,"No tires el dado de seis."],
            [0,None],
        ]
        for index, (expected, text) in enumerate(check_prompts):
            wav = render(folder, f"check-{index}", CHECK_VOICE, text, 165)
            check.append({"id":index, "expected":expected, "decision":score(program, model, wav)})
        public = []
        if args.negative_dir is not None:
            for wav in sorted(args.negative_dir.glob("*.wav")):
                public.append({"id":wav.stem, "expected":0, "decision":score(program, model, wav)})
    print(json.dumps({
        "train_synthetic_voices":TRAIN_VOICES, "dev_synthetic_voice":DEV_VOICE,
        "template_count":len(templates), "bytes":model.stat().st_size,
        "limits":{"max_cost":MAX_COST, "min_gap":MIN_GAP},
        "synthetic_dev":rows, "synthetic_check_voice":CHECK_VOICE,
        "synthetic_check":check, "public_negatives":public,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
