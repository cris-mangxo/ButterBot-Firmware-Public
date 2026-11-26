"""Compare the compact bank with a local speech-to-text fallback."""

import argparse
from collections import Counter
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
from tempfile import TemporaryDirectory
import time

from ald_round import ALD_SHA256, BASELINE_SHA256, digest
from common_voice import DATASETS, MAX_MEMBER_BYTES
from d6_guard import LOCAL_CV_SHA256, guarded_score
from phrases import too_long
from phrases_human import read_tables, select
from pilot import score
from render_cases import FIXTURE, FIXTURES, cases

BANK_SHA256 = "ca40780ad1ee096f211aa59d3d1d7e410bbeb3f2879c0c3f23e76633e589d7ca"
MODELS = {
    "base": ("422f1ae452ade6f30a004d7e5c6a43195e4433bc370bf23fac9cc591f01a8898", 59707625),
    "tiny": ("818710568da3ca15689e31a743197b520007872ff9576237bda97bd1b469c3d7", 32152673),
    "small": ("ae85e4a935d7a567bd102fe55afc16bb595bdb618e11b2fc7591bc08120411bb", 190085487),
}
METHODS = ("compact", "base", "tiny", "small", "agreement")
WHISPER_FLAGS = ("-l", "es", "-nf", "-t", "2", "-nt", "-oj", "-np")
SANDBOX = (r'(version 1) (allow default) (deny network*) '
           r'(deny file-read* file-write* (regex #"^/dev/(cu|tty)[.].*"))')


def load_fixture(corpus, fixture):
    manifest = json.loads((corpus / "manifest.json").read_text("utf-8"))
    rows = cases(FIXTURE.parent.parent / "phrases_cases.json", fixture)
    if manifest != {"fixture_sha256": digest(fixture), "ald_sha256": ALD_SHA256, "samples": rows}:
        raise ValueError("El corpus Ald difiere del banco fijado")
    return [(row, corpus / "audio" / (row["id"] + ".wav")) for row in rows]


def replay_sources(corpus, older_wide):
    manifest = json.loads((corpus / "manifest.json").read_text("utf-8"))
    fixture = FIXTURE.parent.parent / "phrases_cases.json"
    if manifest["fixture_sha256"] != digest(fixture) or manifest["ald_sha256"] != ALD_SHA256:
        raise ValueError("El corpus acústico previo difiere del fijado")
    rows = [row for row in manifest["samples"] if row["split"] in {"test", "enrolled"}
            and row["category"] in {"legacy", "known_deny", "d6"}]
    wide = json.loads((older_wide / "manifest.json").read_text("utf-8"))
    if wide["fixture_sha256"] != digest(fixture.with_name("d6_cases.json")) or wide["ald_sha256"] != ALD_SHA256:
        raise ValueError("El corpus histórico difiere del fijado")
    known, = [row for row in wide["samples"] if row["id"] == "test-0379"]
    historical = dict(known, category="original_fact_veto")
    sets = [(row, corpus / "audio" / (row["id"] + ".wav")) for row in rows]
    return sets + [(historical, older_wide / "audio/test-0379.wav")]


def decode_batch(program, weights, wavs, output, flags=WHISPER_FLAGS):
    if not wavs:
        return [], 0.0
    output.mkdir(mode=0o700)
    files = []
    for index, source in enumerate(wavs):
        target = output / f"turn-{index:03d}.wav"
        shutil.copyfile(source, target)
        files.append(target)
    command = ["/usr/bin/sandbox-exec", "-p", SANDBOX, program, "-m", weights,
               *flags, *files]
    start = time.monotonic()
    result = subprocess.run(command, capture_output=True, timeout=240)
    elapsed = time.monotonic() - start
    if result.returncode:
        raise RuntimeError("Whisper local terminó con error")
    transcripts = []
    for wav in files:
        payload = json.loads(wav.with_suffix(".wav.json").read_text("utf-8"))
        transcripts.append(" ".join(part["text"].strip() for part in payload["transcription"]).strip())
    return transcripts, elapsed


def decide(grammar, original, compact, transcript):
    if len(transcript.encode("utf-8")) > 192:
        transcript = ""
    command = [grammar, str(original), str(compact)]
    process = subprocess.run(command, input=transcript.encode("utf-8"), capture_output=True,
                             check=True, timeout=3)
    return int(process.stdout)


def evaluate(inputs, args, temporary):
    rows = []
    for row, wav in inputs:
        overlong = too_long(wav)
        original = score(args.program, args.baseline, wav) if not overlong else {"label": 0}
        compact = (guarded_score(args.program, args.baseline, args.bank, wav, mode="phrases")
                   if not overlong else {"label": 0})
        audio_eligible = not overlong and not original.get("no_feature")
        # Both branches must be unknown. A known acoustic veto cannot be rescued by ASR.
        attempt = audio_eligible and original["label"] == compact["label"] == 0
        rows.append({
            "id": row["id"], "category": row["category"], "expected": row["label"],
            "original": original["label"], "overlong": overlong, "attempt": attempt,
            "decisions": dict.fromkeys(METHODS, compact["label"]),
        })
    attempts = [i for i, row in enumerate(rows) if row["attempt"]]
    elapsed = {}
    for key in MODELS:
        transcripts, elapsed[key] = decode_batch(
            args.whisper, getattr(args, key), [inputs[i][1] for i in attempts],
            temporary / key, args.decoder_flags,
        )
        for index, transcript in zip(attempts, transcripts, strict=True):
            row = rows[index]
            row["decisions"][key] = decide(args.grammar, row["original"],
                                           row["decisions"]["compact"], transcript)
    for i in attempts:
        values = rows[i]["decisions"]
        if values["base"] == values["tiny"] == 6:
            values["agreement"] = 6
    return rows, {"attempts": len(attempts), "decoder_wall_seconds": elapsed}


def totals(rows):
    out = {}
    for row in rows:
        group = out.setdefault(row["category"], {
            "trials": 0, "overlong": 0, "attempts": 0, "correct": dict.fromkeys(METHODS, 0),
            "false_d6": dict.fromkeys(METHODS, 0),
        })
        group["trials"] += 1
        group["overlong"] += row["overlong"]
        group["attempts"] += row["attempt"]
        for name, label in row["decisions"].items():
            group["correct"][name] += label == row["expected"]
            group["false_d6"][name] += label == 6 and row["expected"] == 0
    return out


def reference_text(grammar, inputs):
    counts = {}
    for sample, _ in inputs:
        group = counts.setdefault(sample["category"], {"trials": 0, "correct": 0})
        group["trials"] += 1
        group["correct"] += decide(grammar, 0, 0, sample["text"]) == sample["label"]
    return counts


def human_sources(archive, directory):
    dataset, size, published = DATASETS["female"]
    checksum = digest(archive)
    if archive.stat().st_size != size or checksum not in (published, LOCAL_CV_SHA256):
        raise ValueError("Common Voice difiere del origen fijado")
    prefix, rows, counts = read_tables(archive)
    exploratory, _, inventory = select(rows, fresh=False)
    fresh, _, _ = select(rows, fresh=True)
    selected = {prefix / "clips" / clip: f"{cohort}-{group}"
                for cohort, clips in (("exploratory", exploratory), ("fresh", fresh))
                for clip, group in clips.items()}
    sources = []
    compressed = directory / "clip.mp3"
    with tarfile.open(archive, "r|*") as members:
        for member in members:
            path = PurePosixPath(member.name)
            if path not in selected:
                continue
            if not member.isfile() or not 0 < member.size <= MAX_MEMBER_BYTES:
                raise ValueError("Clip capturado fuera de los límites")
            with members.extractfile(member) as stream:
                compressed.write_bytes(stream.read())
            identifier = f"public-{len(sources):03d}"
            wav = directory / (identifier + ".wav")
            subprocess.run(
                ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
                 "-i", compressed, "-t", "4.1", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav],
                check=True, capture_output=True, timeout=12,
            )
            row = {"id": identifier, "category": selected.pop(path), "label": 0}
            sources.append((row, wav))
            if not selected:
                break
    if selected:
        raise ValueError("Faltan clips humanos")
    metadata = {"archive_sha256": checksum, "catalog_sha256": published, "dataset_id": dataset,
                "provisional": checksum != published, "table_counts": counts,
                "lexical_inventory": inventory, "distinct_speakers": len(sources)}
    return sources, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=(*FIXTURES, "replay", "human", "predict"))
    for name in ("whisper", *MODELS, "program", "grammar", "baseline", "bank"):
        parser.add_argument(f"--{name}", type=lambda value: Path(value).resolve(), required=True)
    for name in ("corpus", "older-wide", "archive", "wav"):
        parser.add_argument(f"--{name}", type=lambda value: Path(value).resolve())
    parser.add_argument("--short-window", action="store_true", help="Repite el piloto con contexto de 256")
    args = parser.parse_args()
    args.decoder_flags = WHISPER_FLAGS + (("-ac", "256") if args.short_window else ())
    locations = {key: (args.corpus,) for key in FIXTURES}
    locations.update({"replay": (args.corpus, args.older_wide), "human": (args.archive,),
                      "predict": (args.wav,)})
    if not all(locations[args.operation]):
        parser.error("Falta la ruta de corpus para esta operación")
    if digest(args.baseline) != BASELINE_SHA256 or digest(args.bank) != BANK_SHA256:
        parser.error("La base o el banco de frases difiere de la referencia")
    for name, (checksum, size) in MODELS.items():
        weights = getattr(args, name)
        if weights.stat().st_size != size or digest(weights) != checksum:
            parser.error(f"El modelo {name} difiere de la referencia")
    version = subprocess.check_output([args.whisper, "--version"], stderr=subprocess.STDOUT,
                                      timeout=5, text=True).strip()
    if version != "whisper.cpp version: 1.9.6":
        parser.error("Versión de whisper.cpp fuera del experimento fijado")
    with TemporaryDirectory(prefix="bb-d6-transcript-") as scratch:
        work = Path(scratch)
        meta = {}
        if args.operation == "human":
            sources, meta = human_sources(args.archive, work)
        elif args.operation == "replay":
            sources = replay_sources(args.corpus, args.older_wide)
        elif args.operation == "predict":
            sources = [({"id": "input", "category": "prediction", "label": None}, args.wav)]
        else:
            fixture = FIXTURES[args.operation]
            sources = load_fixture(args.corpus, fixture)
            meta = {"fixture_sha256": digest(fixture), "ald_sha256": ALD_SHA256,
                    "grammar_with_reference_text": reference_text(args.grammar, sources)}
        rows, cost = evaluate(sources, args, work)
    if args.operation == "predict":
        row = rows[0]
        print(json.dumps({"decisions": row["decisions"], "overlong": row["overlong"],
                          "used_transcript": row["attempt"], "inference": cost}, indent=2))
        return
    result = {"operation": args.operation, "model_provenance": MODELS, "decoder": version,
              "whisper_flags": args.decoder_flags, "bank_sha256": BANK_SHA256,
              "categories": totals(rows), "inference": cost, **meta}
    if args.operation != "human":
        result["case_decisions"] = rows
    if args.operation == "panel":
        probed = {"asr-000", "asr-001", "asr-002", "asr-029", "asr-038"}
        result["excluding_decoder_probe"] = totals([row for row in rows if row["id"] not in probed])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
