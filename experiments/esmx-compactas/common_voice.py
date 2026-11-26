"""Evaluate a verified Mozilla Common Voice es-MX test archive as outside-command speech."""

import argparse
import csv
from io import StringIO
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import tempfile

from ald_round import canonical, digest
from pilot import score

DATASETS = {
    "female": (
        "cmr3zunmk00flnt07p7ai9p43", 1767747525,
        "8c01733fd724d5bb14ed9a65753bcb4fbc87834eb532cbb988eff1c2747080f1",
    ),
    "male": (
        "cmr3jjevj0041nt07vb3sm35d", 2172664927,
        "111c9189f9bd15be34ef52e60c17490aa5f37e2514115eee4582c17aca90fb29",
    ),
}
EXCLUDED_WORDS = set(
    "mantequilla untar chiste chistes broma bromas dato datos curioso curiosos hora horas "
    "horario reloj relojes dado dados caras lados lanzar lanza lanzaos tirar tira avienta "
    "seis cuatro ocho tiempo".split()
)
MAX_MEMBER_BYTES = 2 * 1024 * 1024


def read_test_split(archive):
    with tarfile.open(archive, "r|*") as entries:
        for member in entries:
            name = PurePosixPath(member.name)
            if (name.name == "test.tsv" and member.isfile()
                    and member.size <= MAX_MEMBER_BYTES):
                with entries.extractfile(member) as stream:
                    text = stream.read().decode("utf-8-sig")
                rows = list(csv.DictReader(StringIO(text), delimiter="\t"))
                return name.parent, rows
    raise ValueError("No se encontró test.tsv acotado en el archivo verificado")


def select_examples(rows, limit):
    speakers = set()
    clips = set()
    # Sort by clip path so that the sample is reproducible and independent of
    # any prior ordering by speaker. Never export these IDs or the transcripts.
    for row in sorted(rows, key=lambda r: r["path"]):
        path = row["path"]
        person = row["client_id"]
        words = set(canonical(row["sentence"]).split())
        if (PurePosixPath(path).name != path or not path.endswith(".mp3")
                or not person or not words or words & EXCLUDED_WORDS
                or person in speakers or path in clips):
            continue
        speakers.add(person)
        clips.add(path)
        if len(clips) == limit:
            break
    if len(clips) != limit:
        raise ValueError("La partición no tiene suficientes controles distintos")
    return clips


def tally(summary, label):
    if label not in range(7):
        raise ValueError("El clasificador devolvió una etiqueta inválida")
    summary["rejected" if label == 0 else "false_accepts"] += 1
    if label:
        key = str(label)
        summary["false_accepts_by_label"][key] = summary["false_accepts_by_label"].get(key, 0) + 1


def evaluate_selected(archive, prefix, clips, classifiers):
    selected = {prefix / "clips" / clip for clip in clips}
    limit = len(selected)
    summary = {name: {"rejected": 0, "false_accepts": 0, "false_accepts_by_label": {}}
               for name in classifiers}
    with tempfile.TemporaryDirectory(prefix="cv-esmx-eval-") as temporary:
        mp3 = Path(temporary) / "clip.mp3"
        wav = Path(temporary) / "clip.wav"
        with tarfile.open(archive, "r|*") as members:
            for member in members:
                name = PurePosixPath(member.name)
                if name not in selected:
                    continue
                if not member.isfile() or not 0 < member.size <= MAX_MEMBER_BYTES:
                    raise ValueError("Tipo o tamaño de clip fuera de rango")
                with members.extractfile(member) as stream:
                    mp3.write_bytes(stream.read())
                subprocess.run(
                    ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
                     "-i", mp3, "-t", "4", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav],
                    check=True, capture_output=True, timeout=12,
                )
                for key, classifier in classifiers.items():
                    tally(summary[key], classifier(wav)["label"])
                wav.unlink()
                selected.remove(name)
                if not selected:
                    break
    if selected:
        raise ValueError("El archivo no contiene todos los clips de la muestra")
    return {"split": "test", "outside_command_trials": limit, "distinct_speakers": limit,
            "max_seconds_per_clip": 4, "results": summary}


def evaluate(archive, program, models, limit):
    prefix, rows = read_test_split(archive)
    clips = select_examples(rows, limit)
    classifiers = {key: lambda wav, model=model: score(program, model, wav)
                   for key, model in models.items()}
    return evaluate_selected(archive, prefix, clips, classifiers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    for name in ("archive", "program", "baseline", "candidate"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--alternate", type=Path)
    parser.add_argument("--limit", type=int, choices=range(1, 33), default=16, metavar="1..32")
    args = parser.parse_args()
    dataset_id, size, checksum = DATASETS[args.dataset]
    if args.archive.stat().st_size != size or digest(args.archive) != checksum:
        parser.error("El archivo no coincide con los bytes y SHA-256 de Mozilla")
    models = {"baseline": args.baseline, "candidate": args.candidate}
    if args.alternate is not None:
        models["alternate"] = args.alternate
    metrics = evaluate(args.archive, args.program.resolve(strict=True), models, args.limit)
    print(json.dumps({"dataset_id": dataset_id, "archive_sha256": checksum,
                      "model_sha256": {key: digest(model) for key, model in models.items()},
                      **metrics}, indent=2))


if __name__ == "__main__":
    main()
