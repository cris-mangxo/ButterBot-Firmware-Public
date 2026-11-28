"""Measure full-turn Hora using the original small encoder before recent rejection tuning."""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json

import torch

from capture_replay import pcm_from
from clock_minimal_pairs_lab import CLOCK, forms, features
from joke_contrast_lab import fit, passed, score
from phoneme_confirm_lab import human_anchors
from phoneme_reject_lab import load_negatives
from phoneme_replay_lab import frozen_model, native_frames, strip_digital_edges
from phoneme_student_lab import load_cache
from time_lab import file_digest


ORIGINAL_WEIGHTS = "8129251b438936609c613a5baea3a1f4cda8014c93013a3485f56400c96307aa"
ARMS = {
    "intent": (1, 3),
    "whole_turn": tuple(range(9)),
    "grammar": (0, 1, 2, 4, 5, 6, 7, 8),
    "short_form": (0, 1, 4, 5, 6, 8),
}


def statistics(rows, rule):
    return dict(Counter(f"{int(expected)}:{int(passed(value, rule))}" for expected, value in rows))


def freeze(args):
    model, info = frozen_model(args.bank)
    frontend_hash = file_digest(args.frontend)
    previous = args.speech / "index.json"
    speakers = args.chm / "speaker-roles.json"
    negatives = args.chm / "native-negative-features"
    synthetic = args.pairs / "synthesis.json"
    pins = {
        "initial_bank": info["weights_sha256"],
        "frontend": frontend_hash,
        "pairs": file_digest(synthetic),
        "chm_negative_features": file_digest(negatives / "index.json"),
        "speaker_assignment": file_digest(speakers),
        "speech": file_digest(previous),
        "espeak": file_digest(args.espeak),
        "source": file_digest(Path(__file__)),
        "agenda": file_digest(args.agenda),
    }
    if (pins["initial_bank"] != ORIGINAL_WEIGHTS or pins["speech"] != info["teacher_cache_sha256"]
            or json.loads(previous.read_text())["frontend_sha256"] != frontend_hash):
        raise ValueError("Inputs differ from the original frozen bank")
    vocabulary, cached = load_cache(args.speech)
    candidates = forms(args.espeak, vocabulary)
    conversation = load_negatives(negatives, speakers, frontend_hash)
    pairs = {split: [] for split in ("train", "dev", "test")}
    pcm_checksums = set()

    def extract(audio):
        pcm = strip_digital_edges(audio)
        pcm_checksums.add(sha256(pcm).hexdigest())
        return features(model, native_frames(args.frontend, pcm), candidates)

    for row in json.loads(synthetic.read_text()):
        file = args.pairs / (row["id"] + ".wav")
        pairs[row["split"]].append((row["time_now"], extract(pcm_from(file, row["wav_sha256"]))))
    negative_anchors = json.loads(args.agenda.read_text())
    if negative_anchors["role"] != "local_calibration_only":
        raise ValueError("Agenda controls must be declared for calibration")
    anchors = [(False, extract(pcm_from(args.agenda.parent / row["wav"], row["sha256"])))
               for row in negative_anchors["rows"]]
    anchor, = [(True, extract(pcm_from(args.human / name, seal)))
                for label, _, name, seal in human_anchors(args.human) if label == CLOCK]
    base = {split: pairs[split] + [(row["intent"] == CLOCK and not row["denied"], features(model, frames, candidates))
                                  for row, frames, *_ in cached[split]] +
            [(False, features(model, frames, candidates)) for frames in conversation[split]]
            for split in ("train", "dev")}
    rules = {}
    for method, selected in ARMS.items():
        for transfer in (False, True):
            training = base["train"] + (8 * [anchor] if transfer else [])
            rule = fit(training, selected)
            rule["threshold"] = max(score(value, rule) for expected, value in training + base["dev"] + anchors
                                    if not expected and value is not None)
            name = method + ("_anchor" if transfer else "_synthetic")
            rules[name] = {"rule": rule, "counts": {split: statistics(rows, rule) for split, rows in pairs.items()}}
    with args.out.open("x") as file:
        json.dump({"pins": pins, "paths": candidates, "rules": rules, "occupied_pcm": sorted(pcm_checksums)},
                  file, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({name: body["counts"] for name, body in rules.items()}, indent=2, allow_nan=False))


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("out", "bank", "pairs", "speech", "chm", "frontend", "espeak", "human", "agenda"):
        parser.add_argument("--" + name, type=Path, required=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    freeze(parser.parse_args())


if __name__ == "__main__":
    main()
