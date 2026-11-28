"""Replay the compact CHM150 adaptation against the current firmware decisions."""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json

import duckdb
import torch

import keyword_lab as kw
from capture_replay import pcm_from, recordings
from chm_transcript_ctc_lab import ARMS
from phoneme_chm150_lab import CORPUS_SHA256, SPEAKER_SPLIT_SHA256, decode, eligible_rows
from phoneme_replay_lab import frozen_model, intent_from_frames, native_frames, overlaps_adjustment
from time_lab import file_digest, invoke


BUTTER_DIAGNOSTIC_SHA256 = "ca4df37ec70e5b6c32e7debeb276b3d6542a5131a6202160ae817bc95ff3d8f7"


def observe(audio, models, args):
    base = int(invoke(args.host, audio))
    if base or not audio:
        return base, "not_probed", {arm: 0 for arm in models}
    reason = json.loads(invoke(args.host, audio, "--decision"))["disposition"]
    frames = native_frames(args.frontend, audio)
    return base, reason, {arm: intent_from_frames(model, frames) for arm, model in models.items()}


def historical_cases(root, butter_diagnostic):
    yield from recordings(root)
    local = root / "primer-candidato"
    massive = local / "fallback-trace-r19" / "massive"
    clock = json.loads((massive / "2425.reserve.json").read_text())
    yield (clock["id"], "public_reserved_time", "time", 5,
           massive / clock["sealed_wav"], clock["wav_sha256"])
    for joke in json.loads((massive / "reserve.json").read_text())["rows"]:
        yield (joke["id"], "public_reserved_joke", "joke", joke["intent"],
               massive / joke["sealed_wav"], joke["sealed_wav_sha256"])
    yield ("angelina-61", "diagnostic_narration", "butter", 1, butter_diagnostic, BUTTER_DIAGNOSTIC_SHA256)


def main():
    parser = ArgumentParser(description=__doc__)
    for key in ("corpus", "speakers", "speech", "base", "adaptations", "root",
                "frontend", "ffmpeg", "host", "capture", "butter-diagnostic"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--wide-bank", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    tools = {key: file_digest(getattr(args, key)) for key in ("frontend", "host", "capture", "ffmpeg")}
    if file_digest(args.corpus) != CORPUS_SHA256 or file_digest(args.speakers) != SPEAKER_SPLIT_SHA256:
        raise ValueError("Corpus or held-out speakers have changed")
    base, before = frozen_model(args.base)
    index = args.adaptations / "local-input.json"
    training = json.loads(index.read_text())
    roles = json.loads(args.speakers.read_text())["speaker_roles"]
    heldout = {speaker for speaker, role in roles.items() if role == "heldout"}
    occupied = set()
    if set(training["entries"]) != {"train", "dev"}:
        raise ValueError("Expected only training and development in the local manifest")
    for part, rows in training["entries"].items():
        for row in rows:
            if roles.get(row["speaker_partition_id"]) != part:
                raise ValueError("Unexpected speaker partition in preparation")
            occupied.add(row["pcm_sha256"])
    if (training["corpus"] != CORPUS_SHA256 or training["speaker_assignment"] != SPEAKER_SPLIT_SHA256
            or training["tools"]["frontend"] != tools["frontend"]):
        raise ValueError("Unexpected preparation")
    speech = args.speech / "index.json"
    synthetic = {row["pcm_sha256"] for row in json.loads(speech.read_text())["records"].values()}
    models = {"initial": base}
    hashes = {"initial": before["weights_sha256"]}
    speech_hash = file_digest(speech)
    if before["teacher_cache_sha256"] != speech_hash:
        raise ValueError("The original model requires its original cache")
    for arm, phone_weight in ARMS.items():
        model, report = frozen_model(args.adaptations / arm)
        if (report["initial_weights_sha256"] != hashes["initial"]
                or report["teacher_cache_sha256"] != speech_hash
                or report["human_manifest_sha256"] != file_digest(index)
                or report["human_phone_loss_weight"] != phone_weight):
            raise ValueError("Adaptation weights are not tied to the declared inputs")
        models[arm] = model
        hashes[arm] = report["weights_sha256"]
    if args.wide_bank:
        model, report = frozen_model(args.wide_bank)
        if (report.get("width") != 96 or report["initial_weights_sha256"] != hashes["initial"]
                or report["teacher_cache_sha256"] != speech_hash
                or report["human_manifest_sha256"] != file_digest(index)):
            raise ValueError("Wide model is outside the pinned experiment")
        models["wide_96"] = model
        hashes["wide_96"] = report["weights_sha256"]

    seen = set()
    historical = {}
    excluded = []
    for identifier, split, _, expected, wav, seal in historical_cases(args.root, args.butter_diagnostic):
        if identifier in seen:
            raise ValueError("Duplicate fixture identifier")
        seen.add(identifier)
        pcm = pcm_from(wav, seal)
        signal = {"raw": pcm, "retained": invoke(args.capture, pcm, "--pcm-with-hangover")}
        if any(overlaps_adjustment(audio, synthetic, occupied) for audio in signal.values()):
            if not split.startswith("synthetic"):
                raise ValueError("Public recording overlaps training or development waveforms")
            excluded.append(identifier)
            continue
        historical[identifier] = {"expected": expected, "split": split}
        for mode, audio in signal.items():
            prior, reason, predictions = observe(audio, models, args)
            historical[identifier][mode] = {"prior": prior, "reason": reason, "proposals": predictions,
                                              "strict": {arm: value if reason == "no_match" else 0
                                                         for arm, value in predictions.items()}}
    if len(seen) != 116:
        raise ValueError("Historical panel size has changed")

    options = {"autoload_known_extensions": False, "autoinstall_known_extensions": False}
    observed_speakers = set()
    evaluated = 0
    conversation = {arm: Counter() for arm in models}
    baseline = Counter()
    with duckdb.connect(config=options) as conn:
        for _, speaker, _, wav in eligible_rows(conn, args.corpus, heldout):
            pcm = decode(args.ffmpeg, wav)
            if not 0 < len(pcm) <= 128_000 or len(pcm) % 2:
                continue
            if sha256(kw.trimmed(pcm)).hexdigest() in occupied:
                raise ValueError("Held-out audio duplicates preparation")
            observed_speakers.add(speaker)
            evaluated += 1
            result, reason, predictions = observe(pcm, models, args)
            baseline[str(result)] += 1
            for arm, value in predictions.items():
                if value:
                    conversation[arm]["all_existing_abstentions"] += 1
                    if reason == "no_match":
                        conversation[arm]["strict"] += 1
    if evaluated != 536 or len(observed_speakers) != 30:
        raise ValueError("Held-out conversation panel has changed")
    print(json.dumps({
        "screen": "stand-alone intent proposals and strict NoMatch hypothesis",
        "models": hashes,
        "tools": tools,
        "manifest_sha256": file_digest(index),
        "historical": historical,
        "excluded_synthetic": excluded,
        "conversation": {"turns": evaluated, "speakers": len(observed_speakers), "baseline": baseline,
                         "additional_activations": conversation},
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
