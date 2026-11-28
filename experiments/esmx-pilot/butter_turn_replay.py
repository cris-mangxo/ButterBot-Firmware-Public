"""Replay a frozen Mantequilla contrast with the current integrated firmware route."""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json

import duckdb
import torch

from butter_turn_lab import BUTTER, features
from capture_replay import pcm_from, recordings, RESERVED_INTENTS
from joke_contrast_lab import passed, score
from keyword_lab import trimmed
from phoneme_chm150_lab import CORPUS_SHA256, SPEAKER_SPLIT_SHA256, decode, eligible_rows
from phoneme_reject_lab import NEGATIVE_CACHE_SHA256
from phoneme_replay_lab import frozen_model, native_frames, strip_digital_edges
from time_lab import file_digest, invoke


ANGELINA_WAV_SHA256 = "ca4df37ec70e5b6c32e7debeb276b3d6542a5131a6202160ae817bc95ff3d8f7"


RESERVED_SPLITS = {"public_reserved_time", *(f"public_reserved_{key}" for key in RESERVED_INTENTS)}


def main():
    parser = ArgumentParser(description=__doc__)
    for key in ("recipe", "bank", "frontend", "pilot", "capture", "root",
                "angelina", "chm", "ffmpeg"):
        parser.add_argument("--" + key, type=Path, required=True)
    for executable in ("pilot", "capture", "ffmpeg"):
        parser.add_argument(f"--{executable}-sha256", required=True)
    args = parser.parse_args()
    hashes = {name: file_digest(getattr(args, mode)) for name, mode in
              (("host", "pilot"), ("capture", "capture"), ("ffmpeg", "ffmpeg"))}
    if (hashes["host"] != args.pilot_sha256 or hashes["capture"] != args.capture_sha256
            or hashes["ffmpeg"] != args.ffmpeg_sha256):
        raise ValueError("Replay executable fingerprint differs")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    rule = json.loads(args.recipe.read_text())
    model, meta = frozen_model(args.bank)
    if (rule["weights"] != meta["weights_sha256"] or rule["frontend"] != file_digest(args.frontend)
            or rule["recipe_source_sha256"] != file_digest(Path(__file__).with_name("butter_turn_lab.py"))):
        raise ValueError("Replay does not match frozen calibration")
    occupied = set(rule["occupied_pcm"])
    calibrated_manifest = args.chm / "native-negative-features/index.json"
    if file_digest(calibrated_manifest) != NEGATIVE_CACHE_SHA256:
        raise ValueError("Calibration inputs are not pinned")
    fitted_conversation = {row["effective_pcm_sha256"]
                           for row in json.loads(calibrated_manifest.read_text())["records"].values()}

    def overlaps_calibration(pcm):
        return (sha256(strip_digital_edges(pcm)).hexdigest() in occupied or
                sha256(trimmed(pcm)).hexdigest() in fitted_conversation)

    def observe(pcm):
        decision = json.loads(invoke(args.pilot, pcm, "--decision"))
        label = int(invoke(args.pilot, pcm))
        eligible = decision["allows_fallback"] and decision["intent"] == 0 and decision["disposition"] == "no_match"
        stripped = strip_digital_edges(pcm)
        value = features(model, native_frames(args.frontend, stripped), rule["paths"])
        guarded = {arm: {"score": score(value, entry["rule"]),
                         "dispatched": label or (BUTTER if eligible and passed(value, entry["rule"]) else 0)}
                   for arm, entry in rule["rules"].items()}
        return label, decision["disposition"], guarded

    raw_changes, captured_changes = Counter(), Counter()
    panels = []
    for identifier, split, category, expected, wav, seal in recordings(args.root):
        pcm = pcm_from(wav, seal)
        retained = invoke(args.capture, pcm, "--pcm-with-hangover")
        panels.append((identifier, split, expected, pcm, retained))
    natural = json.loads((args.angelina / "sealed.json").read_text())
    if natural["wav_sha256"] != ANGELINA_WAV_SHA256:
        raise ValueError("Angelina manifest differs from the previously published recording")
    sample = pcm_from(args.angelina / "mantequilla.wav", ANGELINA_WAV_SHA256)
    if sha256(sample).hexdigest() != natural["pcm_sha256"]:
        raise ValueError("Angelina PCM does not match its diagnostic seal")
    panels.append(("angelina-61", "diagnostic", BUTTER, sample, invoke(args.capture, sample, "--pcm-with-hangover")))
    selected_cases, coverage = {}, Counter()
    matched_calibration = Counter()
    keys = {"angelina-61", "butter", "public-03", "public-37", "public-81", "public-88", "public-89", "veto-fact"}
    for key, split, expected, pcm, retained in panels:
        for mode, view, changes in (("raw", pcm, raw_changes), ("retained", retained, captured_changes)):
            if not view:
                coverage["empty_" + mode] += 1
                continue
            if overlaps_calibration(view):
                if split == "diagnostic" or split in RESERVED_SPLITS:
                    raise ValueError(f"Case intersects adjustment: {key}")
                matched_calibration[f"{split}:{mode}"] += 1
            coverage["scored_" + mode] += 1
            base, disposition, arms = observe(view)
            for arm, row in arms.items():
                if row["dispatched"] != base:
                    changes[f"{arm}:{expected}:{base}:{row['dispatched']}"] += 1
            if key in keys:
                selected_cases.setdefault(key, {})[mode] = {"baseline": base, "primary_disposition": disposition, "arms": arms}

    corpus, split = args.chm / "corpus.parquet", args.chm / "speaker-roles.json"
    if file_digest(corpus) != CORPUS_SHA256 or file_digest(split) != SPEAKER_SPLIT_SHA256:
        raise ValueError("Conversation inputs are not pinned")
    holdout = {key for key, role in json.loads(split.read_text())["speaker_roles"].items() if role == "heldout"}
    additions, before, speakers = Counter(), Counter(), set()
    conversation_pcm = sha256()
    with duckdb.connect(config={"autoload_known_extensions": False, "autoinstall_known_extensions": False}) as connection:
        for _, speaker, _, wav in eligible_rows(connection, corpus, holdout):
            pcm = decode(args.ffmpeg, wav)
            if not 0 < len(pcm) <= 128000 or len(pcm) % 2:
                continue
            conversation_pcm.update(len(pcm).to_bytes(4, "little"))
            conversation_pcm.update(pcm)
            base, _, arms = observe(pcm)
            if overlaps_calibration(pcm):
                raise ValueError("Conversation PCM intersects adjustment")
            speakers.add(speaker)
            before[base] += 1
            additions.update([arm for arm, row in arms.items() if row["dispatched"] != base])
    print(json.dumps({"recipe_sha256": file_digest(args.recipe), "selected_before_replay": rule["selected"],
                      "recordings": len(panels), "changes": {"raw": raw_changes, "retained": captured_changes},
                      "conversation": {"count": sum(before.values()), "speakers": len(speakers), "prior": before,
                                       "length_framed_pcm_sha256": conversation_pcm.hexdigest(),
                                       "new_activations": {arm: additions[arm] for arm in rule["rules"]}},
                      "controls": selected_cases, "executable_sha256": hashes, "scored_or_empty": coverage,
                      "exact_calibration_matches_outside_reserve": matched_calibration}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
