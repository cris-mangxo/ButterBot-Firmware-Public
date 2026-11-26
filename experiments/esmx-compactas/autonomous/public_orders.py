"""Cut fixed continuous human utterances from local copies of public RSS audio."""

import argparse
import json
from pathlib import Path
import re
import subprocess

from ald_round import digest
from pilot import read_pcm

FIXTURE = Path(__file__).with_name("public_orders.json")


def specification():
    data = json.loads(FIXTURE.read_text("utf-8"))
    declared = set()
    for name, source in data["sources"].items():
        if not re.fullmatch(r"[a-z]+(?:-[a-z]+)*", name) or not re.fullmatch(r"[a-f0-9]{64}", source["sha256"]):
            raise ValueError("Fuente pública inválida")
    for case in data["cases"]:
        name = case["id"]
        start, end = case["start_ms"], case["end_ms"]
        if (not re.fullmatch(r"public-[0-9]{2}", name) or name in declared
                or case["source"] not in data["sources"] or case["expected"] not in (0, 6)
                or start < 0 or not 0 < end - start <= 4000):
            raise ValueError("Caso humano inválido")
        declared.add(name)
    return data


def extract(sources, out):
    fixture = specification()
    paths = {key: sources / (key + ".mp3") for key in fixture["sources"]}
    for key, path in paths.items():
        if digest(path) != fixture["sources"][key]["sha256"]:
            raise ValueError(f"El MP3 publicado de {key} difiere de la copia fijada")
    out.mkdir(mode=0o700)
    rows = []
    for case in fixture["cases"]:
        wav = out / (case["id"] + ".wav")
        subprocess.run(
            ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
             "-ss", f'{case["start_ms"]/1000:.3f}', "-i", paths[case["source"]],
             "-t", f'{(case["end_ms"]-case["start_ms"])/1000:.3f}',
             "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav],
            check=True, capture_output=True, timeout=30,
        )
        read_pcm(wav)
        rows.append(dict(case, wav_sha256=digest(wav)))
    (out / "manifest.json").write_text(json.dumps({
        "public_fixture_sha256": digest(FIXTURE), "cases": rows,
    }, indent=2) + "\n", encoding="utf-8")


def verified_cases(folder):
    evidence = json.loads((folder / "manifest.json").read_text("utf-8"))
    if evidence["public_fixture_sha256"] != digest(FIXTURE):
        raise ValueError("El corpus humano difiere de la anotación fijada")
    for case, reference in zip(evidence["cases"], specification()["cases"], strict=True):
        if {key: value for key, value in case.items() if key != "wav_sha256"} != reference:
            raise ValueError("Cambió la anotación previa al ensayo")
        wav = folder / (reference["id"] + ".wav")
        if digest(wav) != case["wav_sha256"]:
            raise ValueError("Cambió la toma pública")
        yield case, wav


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    extract(args.sources, args.out)


if __name__ == "__main__":
    main()
