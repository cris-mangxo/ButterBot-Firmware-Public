"""Freeze a compact phonetic contrast for complete requests to pass the butter."""
from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
import json

import numpy as np
import torch

from capture_replay import pcm_from
from joke_contrast_lab import cheapest, fit, passed, score
from keyword_lab import LABELS, VOICE_SPLITS, render
from phoneme_confirm_lab import model_output
from phoneme_reject_lab import load_negatives
from phoneme_replay_lab import frozen_model, native_frames, strip_digital_edges
from phoneme_student_lab import load_cache
from phoneme_teacher_lab import phonemes
from time_lab import file_digest


BUTTER = 1
PAIRS = {
    "train": (
        ("Pásame la mantequilla", "No me pases la mantequilla"),
        ("Dame la mantequilla", "Dame la sal"),
        ("Tráeme la mantequilla", "Tráeme la mermelada"),
        ("Me puedes pasar la mantequilla", "Me puedes pasar la servilleta"),
        ("Pásame la mantequilla por favor", "Pásame la mantequilla de cacahuate"),
        ("Puedes traerme la mantequilla", "Puedes agregar la mantequilla"),
        ("Dame tantita mantequilla", "Dale tantita mantequilla"),
        ("Me pasas la mantequilla", "Ya me pasaste la mantequilla"),
    ),
    "dev": (
        ("Oye pásame la mantequilla", "Oye no me pases la mantequilla"),
        ("Puedes darme mantequilla", "Puedes ponerle mantequilla"),
        ("Me podrías traer la mantequilla", "Me podrías decir la hora"),
        ("Me das la mantequilla por favor", "Me das la mermelada por favor"),
        ("Alcánzame la mantequilla", "Alcánzame la mantequilla de almendra"),
    ),
    "test": (
        ("Porfa me pasas la mantequilla", "Porfa no me pases la mantequilla"),
        ("Me regalas tantita mantequilla", "Me regalas tantita miel"),
        ("Puedes alcanzarme la mantequilla", "Puedes derretir la mantequilla"),
        ("Me da la mantequilla por favor", "Me da la crema por favor"),
        ("Oye me das la mantequilla", "Oye me das la mantequilla, mejor no"),
        ("Hey tráeme un poco de mantequilla", "Hay que traer la mantequilla"),
    ),
}
FORMS = {
    "noun": ("mantequilla",),
    "request": ("pásame", "dame", "tráeme", "alcánzame", "páseme", "traerme", "traiga",
                "me pasa", "me das", "me da", "me pasas", "pasarme", "darme", "alcanzarme"),
    "deny": ("no me pases", "no me traigas", "no me des", "no quiero", "mejor no",
             "cancela", "olvídalo"),
    "positive": tuple(pair[0] for pair in PAIRS["train"]),
    "control": tuple(pair[1] for pair in PAIRS["train"]) + (
        "es mantequilla", "hay mantequilla", "dónde está la mantequilla", "mezcla con mantequilla",
        "en mantequilla", "agrega la mantequilla", "mantequilla", "tira un dado de seis", "échame un chiste"),
}
ARMS = {"local": (0, 1, 2, 3, 4, 5, 6), "complete": tuple(range(9)),
        "local_no_intent": (0, 2, 3, 4, 5, 6), "complete_no_intent": (0, 2, 3, 4, 5, 6, 7, 8)}


def templates(espeak, vocabulary):
    mapping = {phone: index for index, phone in enumerate(vocabulary)}
    return {key: [tuple(mapping.get(phone, len(mapping) - 1) for phone in phonemes(str(espeak), text + "."))
                  for text in phrases] for key, phrases in FORMS.items()}


@torch.inference_mode()
def features(model, output, paths):
    if output is None or len(output) < 16:
        return None
    phones, intents, denied = model_output(model, output)
    if denied.item() >= 0:
        return None
    frames = phones[0].numpy()
    count = len(frames)
    noun, (start, end) = cheapest(frames, paths["noun"])
    verb, (_, verb_end) = cheapest(frames[:start], paths["request"])
    denial = min(cheapest(view, paths["deny"])[0] for view in (frames[:start], frames[end:]))
    ranks = intents[0].numpy()
    label = LABELS.index(BUTTER)
    index = len(paths["positive"])
    turns = paths["positive"] + paths["control"]
    lengths = torch.tensor([len(path) for path in turns])
    tokens = torch.tensor([token for path in turns for token in path])
    log_probs = phones.log_softmax(-1).transpose(0, 1).expand(-1, len(turns), -1)
    costs = torch.nn.functional.ctc_loss(log_probs, tokens, torch.full((len(turns),), count),
                                        lengths, reduction="none").numpy() / count
    costs = np.where(np.isfinite(costs), costs, 12)
    affirmative = min(costs[:index])
    control = min(costs[index:])
    return np.asarray([noun, ranks[label] - max(np.delete(ranks, label)), -denied.item(), verb,
                       denial - verb, (start - verb_end) / count, (count - end) / count,
                       affirmative, control - affirmative])


def generate(folder):
    folder.mkdir(mode=0o700)
    jobs = [(split, sequence, kind, number, voice, rate, pair[kind])
            for split, pairs in PAIRS.items() for sequence, pair in enumerate(pairs)
            for kind in (0, 1) for number, (voice, rate) in enumerate(VOICE_SPLITS[split])]
    def synthesize(row):
        split, sequence, kind, number, voice, rate, phrase = row
        key = f"{split}-{sequence:02}-{kind}-{number}"
        wav = render(folder, key, voice, phrase + ".", rate)
        return {"id": key, "split": split, "expected": kind == 0, "wav_sha256": file_digest(wav)}
    with ThreadPoolExecutor(max_workers=4) as workers:
        rows = list(workers.map(synthesize, jobs))
    (folder / "synthesis.json").write_text(json.dumps(rows, indent=2) + "\n")


def measure(rows, rule):
    return dict(Counter(f"{int(target)}:{int(passed(value, rule))}" for target, value in rows))


def freeze(args, model, paths, meta):
    phoneme_bank = args.synthetic / "index.json"
    negative_bank = args.chm / "native-negative-features"
    frontend = file_digest(args.frontend)
    if (file_digest(phoneme_bank) != meta["teacher_cache_sha256"] or
            json.loads(phoneme_bank.read_text())["frontend_sha256"] != frontend or
            file_digest(negative_bank / "index.json") != meta["negative_cache_sha256"]):
        raise ValueError("Unexpected calibration inputs")
    _, cached = load_cache(args.synthetic)
    negative = load_negatives(negative_bank, args.chm / "speaker-roles.json", frontend)
    challenge = {split: [] for split in VOICE_SPLITS}
    occupied = set()
    rows = json.loads((args.output / "synthesis.json").read_text())
    expected_rows = {
        f"{split}-{index:02}-{kind}-{voice}": (split, kind == 0)
        for split, pairs in PAIRS.items() for index in range(len(pairs))
        for kind in (0, 1) for voice in range(len(VOICE_SPLITS[split]))
    }
    if len(rows) != len(expected_rows) or {
        row["id"]: (row["split"], row["expected"]) for row in rows
    } != expected_rows:
        raise ValueError("Synthetic manifest does not match the fixed challenge")
    for row in rows:
        pcm = strip_digital_edges(pcm_from(args.output / (row["id"] + ".wav"), row["wav_sha256"]))
        checksum = sha256(pcm).hexdigest()
        if checksum in occupied:
            raise ValueError("Synthetic waveform appears twice")
        occupied.add(checksum)
        value = features(model, native_frames(args.frontend, pcm), paths)
        challenge[row["split"]].append((row["expected"], value))
    base = {split: challenge[split] + [(row["intent"] == BUTTER and not row["denied"], features(model, matrix, paths))
                                      for row, matrix, *_ in cached[split]] +
            [(False, features(model, matrix, paths)) for matrix in negative[split]] for split in ("train", "dev")}
    rules = {}
    for arm, columns in ARMS.items():
        rule = fit(base["train"], columns)
        rule["threshold"] = max(score(value, rule) for target, value in base["train"] + base["dev"]
                                if not target and value is not None)
        rules[arm] = {"rule": rule, "matrix": {split: measure(rows, rule) for split, rows in challenge.items()}}
    best_count = max(row["matrix"]["dev"].get("1:1", 0) for row in rules.values())
    selected = next(arm for arm, row in rules.items() if row["matrix"]["dev"].get("1:1", 0) == best_count) if best_count else None
    result = {"paths": paths, "weights": meta["weights_sha256"], "frontend": frontend,
              "recipe_source_sha256": file_digest(Path(__file__)), "synthesis_sha256": file_digest(args.output / "synthesis.json"),
              "selected": selected, "rules": rules, "occupied_pcm": sorted(occupied),
              "base_rows": {split: len(rows) for split, rows in base.items()}}
    with (args.output / "freeze.json").open("x") as file:
        json.dump(result, file, indent=2, allow_nan=False)
    print(json.dumps({"selected": selected, "arm_matrices": {key: row["matrix"] for key, row in rules.items()},
                      "examples": len(rows), "base_rows": result["base_rows"]}, indent=2))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("generate", "freeze"))
    parser.add_argument("output", type=Path)
    inputs = ("bank", "espeak", "frontend", "synthetic", "chm")
    for key in inputs:
        parser.add_argument("--" + key, type=Path)
    args = parser.parse_args()
    if args.stage == "generate":
        generate(args.output)
        return
    if any(getattr(args, key) is None for key in inputs):
        parser.error("freeze requires " + ", ".join("--" + key for key in inputs))
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model, meta = frozen_model(args.bank)
    vocabulary = json.loads((args.synthetic / "index.json").read_text())["vocabulary"]
    freeze(args, model, templates(args.espeak, vocabulary), meta)


if __name__ == "__main__":
    main()
