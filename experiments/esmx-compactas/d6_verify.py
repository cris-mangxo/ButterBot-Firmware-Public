"""Check unchanged baseline decisions and rejection of malformed D6 models."""

import argparse
import json
from pathlib import Path
import struct
import subprocess
from tempfile import TemporaryDirectory
import wave

from ald_round import BASELINE_SHA256, digest
from d6_guard import MAX_FRAMES
from pilot import MAX_COST, MIN_GAP, read_pcm, score


def check_parity(reference, program, baseline, corpora):
    checked = 0
    for corpus in corpora:
        manifest = json.loads((corpus / "manifest.json").read_text("utf-8"))
        for sample in manifest["samples"]:
            if sample["split"] != "test":
                continue
            wav = corpus / "audio" / (sample["id"] + ".wav")
            with wave.open(str(wav)) as stream:
                if stream.getnframes() > MAX_FRAMES:
                    continue
            if score(reference, baseline, wav) != score(program, baseline, wav):
                raise ValueError(f"El comparador base cambió en {sample['id']}")
            checked += 1
    if not checked:
        raise ValueError("Faltan clips de evaluación de la base")
    return checked


def check_invalid(program, baseline, anchor, wav):
    valid = anchor.read_bytes()
    pcm = read_pcm(wav)
    command = [program, "anchor", baseline, str(MAX_COST), str(MIN_GAP)]
    control = subprocess.run([*command, anchor], input=pcm, capture_output=True, timeout=10, check=True)
    if json.loads(control.stdout)["label"] != 6:
        raise ValueError("El control positivo del verificador no produjo D6")
    invalid = (b"", valid + b"\0", valid[:-4] + struct.pack("<I", 0))
    with TemporaryDirectory(prefix="d6-invalid-model-") as temp:
        model = Path(temp) / "invalid.bin"
        for raw in invalid:
            model.write_bytes(raw)
            result = subprocess.run([*command, model], input=pcm, capture_output=True, timeout=10)
            if result.returncode != 2 or result.stdout:
                raise ValueError("Se aceptó un verificador binario inválido")
    return len(invalid)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "program", "baseline", "anchor", "positive-wav"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, action="append", required=True)
    args = parser.parse_args()
    if digest(args.baseline) != BASELINE_SHA256:
        parser.error("La base no coincide con el modelo congelado")
    print(json.dumps({
        "baseline_exact_parity": check_parity(args.reference, args.program, args.baseline, args.corpus),
        "invalid_anchor_models_rejected": check_invalid(args.program, args.baseline, args.anchor, args.positive_wav),
    }))


if __name__ == "__main__":
    main()
