"""Compare original-bank clock rules with the compiled pilot on frozen input."""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json

import torch

from capture_replay import pcm_from, recordings
from clock_minimal_pairs_lab import CLOCK, features
from joke_contrast_lab import passed, score
from phoneme_chm150_lab import CORPUS_SHA256, SPEAKER_SPLIT_SHA256, decode, eligible_rows
from phoneme_confirm_lab import natural_adjustment_hashes
from phoneme_replay_lab import frozen_model, native_frames, overlaps_adjustment, strip_digital_edges
from time_lab import file_digest, invoke


def examples(root):
    yield from recordings(root)
    workspace = root / "primer-candidato"
    podcast = workspace / "dice-recorrido-r22"
    for row in json.loads((podcast / "clock-public-sealed.json").read_text())["items"]:
        yield (row["id"], "diagnostic_podcast", "time", row["expected"],
               podcast / (row["id"] + ".wav"), row["wave_sha256"])
    massive = workspace / "fallback-trace-r19" / "massive"
    item = json.loads((massive / "2425.reserve.json").read_text())
    yield (item["id"], "public_reserved_time", "time", CLOCK,
           massive / item["sealed_wav"], item["wav_sha256"])
    for row in json.loads((massive / "reserve.json").read_text())["rows"]:
        yield (row["id"], "public_reserved_joke", "joke", row["intent"],
               massive / row["sealed_wav"], row["sealed_wav_sha256"])


def weak_dice_conflict(stages):
    banks = stages.get("hypothetical_acoustic_checks", {})
    active = [banks.get(name) for name in ("reference", "candidate")]
    return (all(active) and stages.get("ordered_die") is False
            and any(bank["would_compete_as_negation"] for bank in active)
            and all(not bank["would_prefer_negation"] and not bank["would_meet_negation_floor"]
                    for bank in active))


def observe(pcm, args, model, frozen):
    decision = json.loads(invoke(args.host, pcm, "--decision"))
    baseline = int(invoke(args.host, pcm))
    ft = features(model, native_frames(args.frontend, strip_digital_edges(pcm)), frozen["paths"])
    accepted = {arm for arm, value in frozen["rules"].items() if passed(ft, value["rule"])}
    clean = baseline == 0 and decision["disposition"] == "no_match"
    blocked = baseline == 0 and decision["disposition"] == "dice_negation_conflict"
    weak = bool(blocked and accepted and weak_dice_conflict(json.loads(invoke(args.stages, pcm))))
    return {"baseline": baseline, "disposition": decision["disposition"],
            "hypotheses": {arm: {"score": score(ft, value["rule"]),
                                  "strict": arm in accepted and clean,
                                  "weak_conflict": arm in accepted and (clean or weak),
                                  "all_conflicts": arm in accepted and (clean or blocked)}
                           for arm, value in frozen["rules"].items()}}


def conversation(args, visit, excluded):
    import duckdb

    speaker_split = args.chm / "speaker-roles.json"
    corpus = args.chm / "corpus.parquet"
    if file_digest(corpus) != CORPUS_SHA256 or file_digest(speaker_split) != SPEAKER_SPLIT_SHA256:
        raise ValueError("Unexpected conversation corpus or speaker partition")
    roles = json.loads(speaker_split.read_text())["speaker_roles"]
    heldout = {speaker for speaker, role in roles.items() if role == "heldout"}
    counts = Counter()
    observed_speakers = set()
    options = {"autoload_known_extensions": False, "autoinstall_known_extensions": False}
    with duckdb.connect(config=options) as connection:
        for _, speaker, _, wav in eligible_rows(connection, corpus, heldout):
            pcm = decode(args.ffmpeg, wav)
            if not 0 < len(pcm) <= 128_000 or len(pcm) % 2:
                continue
            if excluded(pcm):
                raise ValueError("Held-out conversation intersects adjustment data")
            observed_speakers.add(speaker)
            result = visit(pcm)
            counts["evaluated"] += 1
            for arm, policies in result["hypotheses"].items():
                for policy in ("strict", "weak_conflict", "all_conflicts"):
                    if policies[policy]:
                        counts[arm + ":" + policy] += 1
    if counts["evaluated"] != 536 or len(observed_speakers) != 30:
        raise ValueError("Incomplete sealed conversation panel")
    return {"speakers": len(observed_speakers), "counts": counts}


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("root", "bank", "recipe", "speech", "chm", "frontend", "host", "capture", "stages", "ffmpeg"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model, weights = frozen_model(args.bank)
    frozen = json.loads(args.recipe.read_text())
    speech = args.speech / "index.json"
    pins = {
        "initial_bank": weights["weights_sha256"],
        "frontend": file_digest(args.frontend),
        "speech": file_digest(speech),
        "chm_negative_features": file_digest(args.chm / "native-negative-features" / "index.json"),
        "speaker_assignment": file_digest(args.chm / "speaker-roles.json"),
        "source": file_digest(Path(__file__).with_name("clock_initial_guard_lab.py")),
    }
    if any(frozen["pins"].get(key) != value for key, value in pins.items()):
        raise ValueError("Replay inputs differ from original freeze")
    occupied = set(frozen["occupied_pcm"]) | natural_adjustment_hashes(args.chm / "native-negative-features")
    adjusted = {row["pcm_sha256"] for row in json.loads(speech.read_text())["records"].values()}

    def excluded(pcm):
        return (overlaps_adjustment(pcm, adjusted, occupied)
                or sha256(strip_digital_edges(pcm)).hexdigest() in occupied)

    def visit(pcm):
        return observe(pcm, args, model, frozen)

    rows = {}
    excluded_synthetic = []
    seen = set()
    for name, split, category, target, wav, seal in examples(args.root):
        pcm = pcm_from(wav, seal)
        captured = invoke(args.capture, pcm, "--pcm-with-hangover")
        if name in seen:
            raise ValueError("Repeated historical test")
        seen.add(name)
        if excluded(pcm) or excluded(captured):
            if not split.startswith("synthetic"):
                raise ValueError(f"Test intersects adjustment data: {name}")
            excluded_synthetic.append(name)
            continue
        rows[name] = {"split": split, "category": category, "target": target,
                      "raw": visit(pcm), "retained": visit(captured)}
    if len(seen) != 120:
        raise ValueError("Incomplete historical panel")
    report = {"recipe_sha256": file_digest(args.recipe),
              "tools": {key: file_digest(getattr(args, key))
                        for key in ("frontend", "host", "capture", "stages", "ffmpeg")},
              "history": rows, "excluded_synthetic": excluded_synthetic,
              "conversation": conversation(args, visit, excluded)}
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
