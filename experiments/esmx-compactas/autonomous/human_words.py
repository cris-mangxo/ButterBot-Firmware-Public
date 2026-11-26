"""Build local D6 training turns from licensed, speaker-partitioned MSWC words."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from hashlib import sha256
import io
import json
from pathlib import Path, PurePosixPath
import random
import re
import subprocess
import tarfile
import wave

import numpy as np

from ald_round import canonical, digest
from common_voice import DATASETS
from d6_guard import LOCAL_CV_SHA256
from phrases_human import read_tables

REVISION = "0bc9df68e92fd6bb54176bf7eb29e2b9e97cb218"
ARCHIVES = {
    "splits": "d78b2c6ce54edeb8fa8d2b87b70989cfb98b042925b10fec2879dfb2d27cc6c2",
    "alignments": "73155e9c1202ccc2218f5a9779659eba7b440d3875446bcb701f31723dff6215",
    "train-2": "959fb6b7e9dd592a92e1dd77510abeae5ded4eaa9a06ca62732196bb41071ffe",
    "train-3": "19b50433a88984bbbcf4f697934a4db8c3cf8a5a7b5b580b2701b9de094b67c4",
    "train-4": "9af7c6426dd3ef40266d69d28f73d9ddaaf6c4d674e6aa05b17e3de64e7d1617",
    "train-8": "a5a08683addbfe029852f15358ed8e58b77caf22d5e348d8db6ee10a2216a420",
    "train-9": "938a362060ee2ebe084d293d40959276de890c8b2086a838d33afa7b7e44ce8a",
    "train-10": "028c4488ceed146e710c73651a4399e8835e429bad3d1c26c9828667371db777",
    "train-13": "e12b2f7d27da2be39d108996b3f8d460b640cd2b062c731961d90fa0601ea4e4",
    "train-14": "fc044fab364c499d65bc04e547000b2624ccdb0d1791bb24efb064ca1fb88df9",
}
# Keep words within the pinned shards. Stop words such as «un» and «de»
# remain supplied by the continuous synthetic turns in the mixed training set.
VERBS = ("tira", "tirar", "lanza", "lanzar", "echa", "echar")
NUMBERS = ("dos", "tres", "cuatro", "cinco", "diez", "doce", "nueve")
NEGATIONS = ("nunca", "tampoco")
CONTEXT = ("dijo", "cuando", "luego", "dime")
FILLERS = ("mañana", "tarde", "solo", "claro", "todo")
WORDS = {*VERBS, *NUMBERS, *NEGATIONS, *CONTEXT, *FILLERS, "dado", "dados", "seis"}
SOURCE = re.compile(r"[^_]+_(common_voice_es_[0-9]+)\.opus")
WORD_INTERVAL = re.compile(r'xmin\s*=\s*([0-9.]+)\s+xmax\s*=\s*([0-9.]+)\s+text\s*=\s*"([^"]*)"')


def candidates(splits, reserved):
    pools = defaultdict(list)
    with tarfile.open(splits) as tar, tar.extractfile("train.csv") as data:
        for row in csv.DictReader(io.TextIOWrapper(data, encoding="utf-8")):
            word, speaker = row["WORD"], row["SPEAKER"]
            link = PurePosixPath(row["LINK"])
            if (word not in WORDS or speaker in reserved or row["VALID"] != "True"
                    or not re.fullmatch(r"[a-f0-9]{128}", speaker)
                    or link.parts != (word, link.name)
                    or not re.fullmatch(r"common_voice_es_[0-9]+\.opus", link.name)):
                continue
            fingerprint = sha256(speaker.encode("ascii")).hexdigest()
            split = "dev" if int(fingerprint[:2], 16) < 48 else "train"
            pools[split, word].append((f"{word}_{link.name}", link.stem, speaker))
    # A bounded, reproducible sample per token. Rare words keep all takes.
    return {(split, word): sorted(items, key=lambda r: sha256(r[0].encode()).digest())[:64 if split == "train" else 24]
            for (split, word), items in pools.items()}


def durations(archive, targets):
    lengths = {}
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            path = PurePosixPath(member.name)
            key = (path.stem, path.parent.name)
            if key not in targets or not member.isfile() or path.suffix != ".TextGrid":
                continue
            grid = tar.extractfile(member).read().decode("utf-8-sig")
            words = grid.split('name = "words"', 1)
            if len(words) < 2:
                continue
            tier = words[1].split("item [", 1)[0]
            matches = defaultdict(list)
            for start, end, token in WORD_INTERVAL.findall(tier):
                if token in targets[key]:
                    matches[token].append(round((float(end) - float(start)) * 16000))
            for token, sizes in matches.items():
                if len(sizes) == 1 and 1600 <= sizes[0] <= 12800:
                    lengths[f"{token}_{path.stem}.opus"] = sizes[0]
    return lengths


def read_shard(archive, needed):
    clips = {}
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if member.name in needed and member.isfile() and 0 < member.size < 65536:
                clips[member.name] = tar.extractfile(member).read()
    return clips


def decode_clip(blob, size):
    pcm = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", "pipe:0",
         "-ar", "16000", "-ac", "1", "-f", "s16le", "pipe:1"],
        input=blob, capture_output=True, check=True, timeout=10,
    ).stdout
    samples = np.frombuffer(pcm, dtype="<i2")
    if len(samples) != 16000:
        return None
    start = (len(samples) - size) // 2
    crop = samples[start:start + size].astype(np.float32)
    peak = np.max(np.abs(crop))
    # Check that a padded word corresponds to its alignment before cropping.
    edges = np.concatenate((samples[:max(0, start - 320)], samples[start + size + 320:]))
    if peak < 250 or np.max(np.abs(edges.astype(np.float32)), initial=0) > peak * 0.12:
        return None
    fade = np.linspace(0, 1, 80, dtype=np.float32)
    crop[:80] *= fade
    crop[-80:] *= fade[::-1]
    return crop / peak


def words_in_turn(rng, pool, label):
    verb = rng.choice([token for token in VERBS if pool.get(token)])
    number = rng.choice([token for token in NUMBERS if pool.get(token)])
    if label == 1:
        return [verb, "dado", "seis"]
    if label == 2:
        if rng.random() < 0.5:
            return [verb, "dado", number]
        return [verb, "dos", "dados", "seis"]
    if label == 3:
        return [rng.choice(NEGATIONS), verb, "dado", "seis"]
    patterns = [
        [rng.choice(CONTEXT), verb, "dado", "seis"],
        ["dado", "seis", rng.choice(FILLERS)],
        [rng.choice(FILLERS), verb, number],
    ]
    return rng.choice(patterns)


def render_turn(rng, pool, tokens, wav):
    samples = [np.zeros(rng.randint(640, 1440))]
    intervals = []
    sources = []
    previous_speaker = None
    cursor = len(samples[0])
    for token in tokens:
        candidates = pool[token]
        same = [item for item in candidates if item[0] == previous_speaker]
        speaker, clip_id, pcm = rng.choice(same or candidates)
        samples.append(pcm * rng.uniform(4500, 15000))
        intervals.append([token, cursor, cursor + len(pcm)])
        gap = np.zeros(rng.randint(220, 1100))
        samples.append(gap)
        cursor += len(pcm) + len(gap)
        sources.append([sha256(speaker.encode("ascii")).hexdigest(), clip_id])
        previous_speaker = speaker
    with wave.open(str(wav), "wb") as dst:
        dst.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        dst.writeframes(np.rint(np.concatenate(samples)).astype("<i2").tobytes())
    wav.with_suffix(".json").write_text(json.dumps({
        "sample_rate": 16000, "word_intervals": intervals,
    }) + "\n", encoding="utf-8")
    return sources


def build(root, reference, out):
    files = {key: root / f"mswc-es-{key}.tar.gz" for key in ARCHIVES}
    if any(digest(files[key]) != expected for key, expected in ARCHIVES.items()):
        raise ValueError("El corpus MSWC no coincide con la revisión fijada")
    _, _, catalog = DATASETS["female"]
    reference_sha = digest(reference)
    if reference_sha not in {catalog, LOCAL_CV_SHA256}:
        raise ValueError("El panel Common Voice no coincide con la referencia")
    _, reference_rows, _ = read_tables(reference)
    reserved = {row["client_id"] for row in reference_rows if row["client_id"]}
    pools = candidates(files["splits"], reserved)
    targets = defaultdict(set)
    for (_, token), items in pools.items():
        for _, stem, speaker in items:
            targets[stem, speaker].add(token)
    lengths = durations(files["alignments"], targets)
    blobs = {}
    with ThreadPoolExecutor(max_workers=4) as workers:
        for clips in workers.map(lambda name: read_shard(files[name], lengths), sorted(files.keys() - {"splits", "alignments"})):
            blobs.update(clips)
        pcm = dict(zip(sorted(blobs), workers.map(
            lambda name: decode_clip(blobs[name], lengths[name]), sorted(blobs)), strict=True))
    usable = defaultdict(dict)
    for (split, token), items in pools.items():
        selected = [(speaker, name, pcm[name]) for name, _, speaker in items
                    if name in pcm and pcm[name] is not None]
        if selected:
            usable[split][token] = selected
    required = {"dado", "dados", "dos", "seis", *NEGATIONS, *CONTEXT, *FILLERS}
    missing = {split: sorted(required - usable[split].keys()) for split in ("train", "dev")}
    if any(missing.values()):
        raise ValueError(f"Faltan palabras de contraste: {missing}")

    out.mkdir(mode=0o700)
    audio = out / "audio"
    audio.mkdir(mode=0o700)
    rng = random.Random(83731)
    rows = []
    for split, count in (("train", 160), ("dev", 40)):
        pool = usable[split]
        for i in range(count * 4):
            label = i % 4
            words = words_in_turn(rng, pool, label)
            name = f"{split}-{i:04d}"
            wav = audio / f"{name}.wav"
            sources = render_turn(rng, pool, words, wav)
            rows.append({
                "id": name, "split": split, "label": label, "text": " ".join(words),
                "sources": sources, "wav_sha256": digest(wav),
                "annotation_sha256": digest(wav.with_suffix(".json")),
            })
    (out / "manifest.json").write_text(json.dumps({
        "revision": REVISION, "reference_sha256": reference_sha,
        "reserved_common_voice_ids": len(reserved),
        "eligible_word_counts": {s: {w: len(clips) for w, clips in p.items()} for s, p in usable.items()},
        "samples": rows,
    }, indent=2) + "\n", encoding="utf-8")
    read_manifest(out)


def read_manifest(folder):
    report = json.loads((folder / "manifest.json").read_text("utf-8"))
    if report["revision"] != REVISION or report["reference_sha256"] not in {DATASETS["female"][2], LOCAL_CV_SHA256}:
        raise ValueError("Procedencia del corpus humano inválida")
    method = report.get("method", "mswc_word_compositions")
    if method not in {"mswc_word_compositions", "mswc_words_spliced_into_ald"}:
        raise ValueError("Preparación del corpus humano desconocida")
    speakers, clips, originals = (defaultdict(set) for _ in range(3))
    ids = set()
    for row in report["samples"]:
        split, name = row["split"], row["id"]
        if (split not in {"train", "dev"} or not re.fullmatch(fr"{split}-[0-9]{{4}}", name)
                or name in ids or row["label"] not in range(4) or not row["sources"]):
            raise ValueError("Muestra de entrenamiento humano inválida")
        ids.add(name)
        wav = folder / "audio" / f"{name}.wav"
        if digest(wav) != row["wav_sha256"] or digest(wav.with_suffix(".json")) != row["annotation_sha256"]:
            raise ValueError("Cambió la muestra humana o su alineación")
        for speaker, clip in row["sources"]:
            source = SOURCE.fullmatch(clip)
            if (source is None or not re.fullmatch(r"[a-f0-9]{64}", speaker)
                    or (int(speaker[:2], 16) < 48) != (split == "dev")):
                raise ValueError("Voz humana fuera de su partición")
            speakers[split].add(speaker)
            clips[split].add(clip)
            originals[split].add(source[1])
    if (not speakers["train"] or not speakers["dev"]
            or speakers["train"] & speakers["dev"] or originals["train"] & originals["dev"]):
        raise ValueError("Las particiones humanas deben usar voces y grabaciones distintas")
    return report["samples"], {
        "manifest_sha256": digest(folder / "manifest.json"), "revision": REVISION,
        "reference_sha256": report["reference_sha256"],
        "reserved_common_voice_ids": report["reserved_common_voice_ids"],
        "speakers": {split: len(items) for split, items in speakers.items()},
        "word_recordings": {split: len(items) for split, items in clips.items()},
        "original_recordings": {split: len(items) for split, items in originals.items()},
        "eligible_word_counts": report["eligible_word_counts"],
        "method": method,
        **{key: report[key] for key in ("human_word_manifest_sha256", "ald_carrier_manifest_sha256") if key in report},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("sources", "reference", "out"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    args = parser.parse_args()
    build(args.sources, args.reference, args.out)
