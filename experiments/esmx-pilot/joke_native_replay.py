"""Verify the opt-in C++ Chiste route on sealed local recordings."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from capture_replay import evaluate, pcm_from, recordings
from time_lab import file_digest, invoke, supplemental


def label(program, pcm):
    return int(invoke(program, pcm))


def historical(args):
    rows = list(recordings(args.root))
    rows.append(supplemental(args.root, args.coerll.resolve(strict=True)))
    matrix = {mode: Counter() for mode in ("raw", "retained")}
    for name, _, _, target, path, seal in rows:
        pcm = pcm_from(path, seal)
        retained = invoke(args.capture, pcm, "--pcm-with-hangover")
        for mode, audio in (("raw", pcm), ("retained", retained)):
            before = label(args.baseline, audio)
            after = label(args.pilot, audio)
            if after != before and (before != 0 or after != 3 or target != 3):
                raise ValueError(f"Incorrect activation or regression for {name}: {before} → {after}")
            matrix[mode][f"{target}:{before}:{after}"] += 1
    return matrix


def reserved(args):
    root = args.reserve.resolve(strict=True).parent
    spec = json.loads(args.reserve.read_text())
    if spec["role"] != "heldout" or len({row["speaker_group"] for row in spec["rows"]}) != 2:
        raise ValueError("Expected the two designated reserved voices")
    result = []
    for row in spec["rows"]:
        wav = (root / row["sealed_wav"]).resolve(strict=True)
        if wav.parent != root or row["intent"] != 3:
            raise ValueError("Reserved row differs from its declared scope")
        pcm = pcm_from(wav, row["sealed_wav_sha256"])
        if sha256(pcm).hexdigest() != row["sealed_pcm_sha256"]:
            raise ValueError("Reserved PCM differs from its seal")
        retained = invoke(args.capture, pcm, "--pcm-with-hangover")
        before = label(args.baseline, pcm)
        raw = label(args.pilot, pcm)
        captured = label(args.pilot, retained)
        route = evaluate(args.capture, pcm)
        if raw not in (0, 3) or raw != captured or route["observed"] != captured:
            raise ValueError("Pilot and capture disagree on reserved Chiste")
        result.append({"id": row["id"], "before": before, "raw": raw, "captured": captured,
                       "capture_route": route["route"], "retained_pcm_sha256": sha256(retained).hexdigest()})
    expected = next((row for row in result if row["id"] == "5183"), None)
    if not expected or (expected["before"], expected["raw"], expected["captured"]) != (0, 3, 3):
        raise ValueError("The opt-in candidate did not complete the reserved Chiste replay")
    return result


def conversation(args):
    import duckdb
    from phoneme_chm150_lab import CORPUS_SHA256, SPEAKER_SPLIT_SHA256, decode, eligible_rows

    if file_digest(args.corpus) != CORPUS_SHA256 or file_digest(args.speaker_split) != SPEAKER_SPLIT_SHA256:
        raise ValueError("Unexpected conversation corpus or speaker partition")
    roles = json.loads(args.speaker_split.read_text())["speaker_roles"]
    heldout = {speaker for speaker, role in roles.items() if role == "heldout"}
    counts = Counter()
    tested = set()
    config = {"autoload_known_extensions": False, "autoinstall_known_extensions": False}
    with duckdb.connect(config=config) as connection:
        for _, speaker, _, wav in eligible_rows(connection, args.corpus, heldout):
            pcm = decode(args.ffmpeg, wav)
            if not 0 < len(pcm) <= 128_000 or len(pcm) % 2:
                continue
            tested.add(speaker)
            before, after = label(args.baseline, pcm), label(args.pilot, pcm)
            if before != after:
                raise ValueError("Unexpected new decision on reserved conversation")
            counts[str(after)] += 1
    return {"evaluated": sum(counts.values()), "speakers": len(tested), "labels": counts, "added": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("baseline", "pilot", "capture", "root", "coerll", "reserve"):
        parser.add_argument("--" + key, type=Path, required=True)
    for key in ("corpus", "speaker-split", "ffmpeg"):
        parser.add_argument("--" + key, type=Path)
    args = parser.parse_args()
    if any((args.corpus, args.speaker_split, args.ffmpeg)) and not all((args.corpus, args.speaker_split, args.ffmpeg)):
        parser.error("conversation replay requires --corpus, --speaker-split and --ffmpeg together")
    for key in ("baseline", "pilot", "capture"):
        setattr(args, key, getattr(args, key).resolve(strict=True))
    if file_digest(args.baseline) == file_digest(args.pilot):
        raise ValueError("Expected distinct pilot binaries for comparison")
    report = {"baseline_sha256": file_digest(args.baseline), "candidate_sha256": file_digest(args.pilot),
              "capture_sha256": file_digest(args.capture), "reserve_sha256": file_digest(args.reserve),
              "historical": historical(args), "reserve": reserved(args)}
    if args.corpus:
        report["conversation"] = conversation(args)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
