"""Measure lexical near-miss controls from the public es-MX Common Voice archive."""

import argparse
from collections import Counter
import csv
from io import StringIO
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
from tempfile import TemporaryDirectory

from ald_round import BASELINE_SHA256, canonical, digest
from common_voice import DATASETS, MAX_MEMBER_BYTES
from d6_guard import LOCAL_CV_SHA256
from phrases import ANCHOR_SHA256, compare, totals

TABLES = ("train.tsv", "dev.tsv", "test.tsv")
MAX_TABLE_BYTES = 24 * 1024 * 1024
LIMITS = {
    "literal_confuser": 24,
    "negated_near": 24,
    "other_dice_terms": 24,
    "number": 24,
    "general_negation": 24,
    "general": 24,
}
NEGATORS = {"no", "nunca", "jamas", "tampoco"}
DICE = {"dado", "dados"}
FACT = {"dato", "datos", "curioso", "curiosa", "curiosidad", "curiosidades"}


def read_tables(archive):
    rows = []
    counts = {}
    prefix = None
    with tarfile.open(archive, "r|*") as tar:
        for member in tar:
            name = PurePosixPath(member.name)
            if name.name not in TABLES:
                continue
            if (not member.isfile() or not 0 < member.size <= MAX_TABLE_BYTES
                    or name.name in counts or (prefix is not None and name.parent != prefix)):
                raise ValueError("Tabla Common Voice inválida")
            prefix = name.parent
            with tar.extractfile(member) as source:
                split = list(csv.DictReader(StringIO(source.read().decode("utf-8-sig")), delimiter="\t"))
            counts[name.name] = len(split)
            rows.extend(split)
            if len(counts) == len(TABLES):
                break
    if len(counts) != len(TABLES):
        raise ValueError("Faltan tablas Common Voice")
    return prefix, rows, counts


def group(words):
    if {"dato", "curioso"} <= words or ("seis" in words and DICE & words):
        return "literal_confuser"
    if NEGATORS & words and (FACT & words or DICE & words or "seis" in words):
        return "negated_near"
    if DICE & words:
        return "other_dice_terms"
    if "seis" in words:
        return "number"
    if NEGATORS & words:
        return "general_negation"
    return "general"


def select(rows, fresh):
    pools = {group: [] for group in LIMITS}
    texts = Counter()
    lexical_exact = Counter()
    for row in sorted(rows, key=lambda item: item["path"]):
        path = row["path"]
        speaker = row["client_id"]
        sentence = canonical(row["sentence"])
        words = set(sentence.split())
        if (not speaker or len(words) < 3 or PurePosixPath(path).name != path
                or not path.endswith(".mp3")):
            continue
        category = group(words)
        texts[category] += 1
        # These are lexical inventory counters, independent of scoring.
        lexical_exact["dice_plus_seis"] += bool(DICE & words and "seis" in words)
        lexical_exact["dato_plus_curioso"] += {"dato", "curioso"} <= words
        lexical_exact["negation_plus_dato"] += "dato" in words and bool(NEGATORS & words)
        pools[category].append((path, speaker, sentence))

    def take(pools, speakers, sentences):
        selected = {}
        for category, candidates in pools.items():
            count = 0
            for path, speaker, sentence in candidates:
                if speaker in speakers or sentence in sentences or path in selected:
                    continue
                selected[path] = category
                speakers.add(speaker)
                sentences.add(sentence)
                count += 1
                if count == LIMITS[category]:
                    break
        return selected

    speakers = set()
    sentences = set()
    selected = take(pools, speakers, sentences)
    if fresh:
        # Hold out the speakers and texts selected for the exploratory run.
        shortest = {category: sorted(items, key=lambda row: (len(row[2].split()), row[0]))
                    for category, items in pools.items()}
        selected = take(shortest, speakers, sentences)
    if not selected:
        raise ValueError("Sin controles humanos elegibles")
    return selected, dict(texts), dict(lexical_exact)


def evaluate(archive, program, baseline, anchor, bank, fresh):
    dataset, size, published = DATASETS["female"]
    checksum = digest(archive)
    if archive.stat().st_size != size or checksum not in (published, LOCAL_CV_SHA256):
        raise ValueError("Common Voice difiere del archivo de catálogo y del diagnóstico fijado")
    prefix, rows, counts = read_tables(archive)
    clips, available, inventory = select(rows, fresh)
    needed = {prefix / "clips" / clip: category for clip, category in clips.items()}
    samples = []
    with TemporaryDirectory(prefix="bb-public-audio-") as temporary:
        compressed = Path(temporary) / "control.mp3"
        wav = Path(temporary) / "control.wav"
        with tarfile.open(archive, "r|*") as entries:
            for member in entries:
                path = PurePosixPath(member.name)
                if path not in needed:
                    continue
                if not member.isfile() or not 0 < member.size <= MAX_MEMBER_BYTES:
                    raise ValueError("Clip fuera de los límites de extracción")
                with entries.extractfile(member) as file:
                    compressed.write_bytes(file.read())
                subprocess.run(
                    ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
                     "-i", compressed, "-t", "4.1", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav],
                    check=True, capture_output=True, timeout=12)
                samples.append({"category": needed.pop(path), "expected": 0,
                                "decisions": compare(program, baseline, anchor, bank, wav)})
                wav.unlink()
                if not needed:
                    break
    if needed:
        raise ValueError("Faltan clips del conjunto fijado")
    return {"dataset_id": dataset, "archive_sha256": checksum, "catalog_sha256": published,
            "provisional": checksum != published, "table_counts": counts,
            "lexical_inventory": inventory, "available_by_group": available,
            "distinct_speakers": len(samples), "trial_type": "public_read_speech_rejection",
            "cohort": "new_read_short_first" if fresh else "exploratory_read",
            "categories": totals(samples)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "program", "baseline", "anchor", "bank"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--fresh", action="store_true", help="Excluye hablantes y textos del corte exploratorio")
    args = parser.parse_args()
    if digest(args.baseline) != BASELINE_SHA256 or digest(args.anchor) != ANCHOR_SHA256:
        parser.error("La base o el verificador anterior difieren de la referencia fijada")
    print(json.dumps({"bank_sha256": digest(args.bank),
                      **evaluate(args.archive, args.program, args.baseline, args.anchor, args.bank, args.fresh)}, indent=2))


if __name__ == "__main__":
    main()
