"""Prepare or evaluate speaker-disjoint CHM150 out-of-domain speech."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import unicodedata

import duckdb
import numpy as np
import torch

import keyword_lab as kw
from phoneme_reject_lab import BASE_SHA256, NEGATIVE_CACHE_SHA256, PROCEDURE
from phoneme_replay_lab import frozen_model, intent_from_frames, native_frames
from time_lab import file_digest, invoke


CORPUS_SHA256 = "f4a516b4bb69050897457428907f5df1c04ad89901aa83340ad68b68c05e7fe5"
SPEAKER_SPLIT_SHA256 = "161a15448fb26d0cf9b83147bead012e61732c709209ff72465419d0571421d8"
TARGET_WORDS = re.compile(r"\b(?:mantequilla|chistes?|horas?|dados?|seis)\b")
NEGATIVE_CUES = re.compile(r"\b(?:no|cancela|olvida|dejalo)\b")


def write_json(path, data):
    path.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")


def build_roles(connection, corpus):
    speakers = connection.execute(
        "SELECT speaker_id, gender, count(*) FROM read_parquet(?) GROUP BY ALL", [str(corpus)],
    ).fetchall()
    if len(speakers) != 150 or len({item[0] for item in speakers}) != 150:
        raise ValueError("Unexpected CHM150 speaker inventory")
    roles = {}
    stats = {}
    for gender in ("female", "male"):
        members = sorted(
            (item for item in speakers if item[1] == gender),
            key=lambda item: sha256(("bb-chm150-speaker-split-1:" + item[0]).encode()).digest(),
        )
        if len(members) != 75:
            raise ValueError("Unexpected CHM150 speaker strata")
        for phase, partition in zip(("train", "dev", "heldout"), (members[:45], members[45:60], members[60:]), strict=True):
            roles.update({item[0]: phase for item in partition})
            stats.setdefault(phase, Counter())["speakers"] += len(partition)
            stats[phase]["utterances"] += sum(item[2] for item in partition)
    return {"schema": "bb-chm150-speaker-split-1", "parquet_sha256": CORPUS_SHA256,
            "speaker_roles": roles, "stats": stats}


def eligible_rows(connection, corpus, speakers):
    rows = connection.execute(
        "SELECT audio_id, speaker_id, normalized_text, audio.bytes FROM read_parquet(?) "
        "WHERE speaker_id=ANY(?) AND duration>=0.35 AND duration<=4.0 ORDER BY audio_id",
        [str(corpus), sorted(speakers)],
    ).fetchall()
    for ident, speaker, text, wav in rows:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", ident):
            raise ValueError("Unexpected CHM150 utterance ID")
        plain = "".join(ch for ch in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(ch))
        if not TARGET_WORDS.search(plain):
            yield ident, speaker, bool(NEGATIVE_CUES.search(plain)), wav


def decode(ffmpeg, wav):
    return subprocess.run(
        [ffmpeg, "-nostdin", "-v", "error", "-i", "pipe:0", "-ac", "1", "-ar", "16000", "-f", "s16le", "pipe:1"],
        input=wav, capture_output=True, timeout=10, check=True,
    ).stdout


def three_per_speaker(rows):
    grouped = defaultdict(lambda: {True: [], False: []})
    for ident, speaker, cue, wav in rows:
        order = sha256(("bb-chm150-negative-v1:" + ident).encode()).hexdigest()
        grouped[speaker][cue].append((order, ident, wav))
    for speaker, groups in sorted(grouped.items()):
        cue = sorted(groups[True])
        chosen = cue[:1] + sorted(groups[False] + cue[1:])
        for _, ident, wav in chosen[:3]:
            yield ident, speaker, wav


def prepare(connection, args):
    args.output.mkdir(mode=0o700)
    assignment = build_roles(connection, args.corpus)
    split = args.output / "speaker-roles.json"
    write_json(split, assignment)
    if file_digest(split) != SPEAKER_SPLIT_SHA256:
        raise ValueError("Speaker assignment differs")
    roles = assignment["speaker_roles"]
    speakers = {item for item, role in roles.items() if role in ("train", "dev")}
    output = args.output / "native-negative-features"
    output.mkdir(mode=0o700)
    index = {}
    for ident, speaker, wav in three_per_speaker(eligible_rows(connection, args.corpus, speakers)):
        pcm = decode(args.ffmpeg, wav)
        features = native_frames(args.features, pcm)
        if features is None:
            continue
        artifact = output / f"{ident}.npz"
        np.savez_compressed(artifact, features=features)
        index[ident] = {"split": roles[speaker], "speaker": speaker, "label": "out_of_domain",
                        "feature_sha256": file_digest(artifact), "frames": len(features),
                        "effective_pcm_sha256": sha256(kw.trimmed(pcm)).hexdigest()}
    manifest = output / "index.json"
    write_json(manifest, {"schema": "bb-chm150-negative-frames-1", "source_sha256": CORPUS_SHA256,
                         "split_sha256": SPEAKER_SPLIT_SHA256, "extractor_sha256": file_digest(args.features),
                         "records": index})
    print(json.dumps({"manifest_sha256": file_digest(manifest),
                      "matches_initial_experiment": file_digest(manifest) == NEGATIVE_CACHE_SHA256,
                      "counts": Counter(item["split"] for item in index.values())}, indent=2))


def replay(connection, args):
    if file_digest(args.speaker_split) != SPEAKER_SPLIT_SHA256:
        raise ValueError("Unexpected CHM150 speaker split")
    negative_index = args.negatives / "index.json"
    initial, before = frozen_model(args.base)
    adapted, after = frozen_model(args.adapted)
    if (before["weights_sha256"] != BASE_SHA256 or after.get("procedure") != PROCEDURE
            or after.get("initial_weights_sha256") != BASE_SHA256
            or after.get("speaker_split_sha256") != SPEAKER_SPLIT_SHA256
            or file_digest(negative_index) != after["negative_cache_sha256"]):
        raise ValueError("Training provenance differs from the frozen adaptation")
    data = json.loads(negative_index.read_text())
    if file_digest(args.features) != data["extractor_sha256"]:
        raise ValueError("Feature extractor differs from the training cache")
    adjustment = {item["effective_pcm_sha256"] for item in data["records"].values()}
    roles = json.loads(args.speaker_split.read_text())["speaker_roles"]
    speakers = {item for item, role in roles.items() if role == args.split}
    seen = set()
    counters = Counter()
    results = {name: Counter() for name in ("compact", "initial", "adapted", "initial_adds", "adapted_adds")}
    for _, speaker, cue, wav in eligible_rows(connection, args.corpus, speakers):
        pcm = decode(args.ffmpeg, wav)
        if not 0 < len(pcm) <= 128_000 or len(pcm) % 2:
            counters["outside_pcm_window"] += 1
            continue
        if args.split == "heldout" and sha256(kw.trimmed(pcm)).hexdigest() in adjustment:
            raise ValueError("Reserved CHM150 waveform intersects training or development")
        seen.add(speaker)
        counters["evaluated"] += 1
        counters["includes_negative_word"] += cue
        baseline = int(invoke(args.reference, pcm))
        if baseline:
            results["compact"][baseline] += 1
        frames = native_frames(args.features, pcm)
        for name, model in (("initial", initial), ("adapted", adapted)):
            prediction = intent_from_frames(model, frames)
            if prediction:
                results[name][prediction] += 1
                if not baseline:
                    results[name + "_adds"][prediction] += 1
    return {"split": args.split, "speaker_groups": len(seen), "counts": counters, "activation_by_intent": results,
            "initial_sha256": before["weights_sha256"], "adapted_sha256": after["weights_sha256"],
            "reference_sha256": file_digest(args.reference), "negative_cache_sha256": after["negative_cache_sha256"]}


def existing(path):
    return Path(path).resolve(strict=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    children = parser.add_subparsers(dest="operation", required=True)
    preparation = children.add_parser("prepare")
    evaluation = children.add_parser("replay")
    for child in (preparation, evaluation):
        for name in ("corpus", "ffmpeg", "features"):
            child.add_argument("--" + name, required=True, type=existing)
    preparation.add_argument("--output", required=True, type=Path)
    for name in ("speaker-split", "negatives", "base", "adapted", "reference"):
        evaluation.add_argument("--" + name, required=True, type=existing)
    evaluation.add_argument("--split", choices=("dev", "heldout"), required=True)
    args = parser.parse_args()
    if file_digest(args.corpus) != CORPUS_SHA256:
        raise ValueError("CHM150 corpus differs from source pin")
    with duckdb.connect(config={"autoload_known_extensions": False, "autoinstall_known_extensions": False}) as connection:
        if args.operation == "prepare":
            prepare(connection, args)
        else:
            torch.set_num_threads(2)
            print(json.dumps(replay(connection, args), indent=2))


if __name__ == "__main__":
    main()
