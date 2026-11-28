"""Compare paired phonetic evidence for a complete D6 request on a frozen model."""
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
import itertools
import json

import numpy as np
import torch

from capture_replay import pcm_from, recordings
from compact_d6_training_lab import load_examples
from d6_teacher_guard_lab import pcm_seals
from joke_contrast_lab import cheapest, fit, passed, score
from keyword_lab import LABELS
from phoneme_confirm_lab import model_output
from phoneme_reject_lab import load_negatives
from phoneme_replay_lab import frozen_model, native_frames, overlaps_adjustment, strip_digital_edges
from phoneme_student_lab import load_cache
from phoneme_teacher_lab import phonemes
from time_lab import file_digest, invoke


FORMS = {
    "object": ("dado de seis", "dado de seis caras", "dado de seis lados", "d seis"),
    "verb": ("tira", "tire", "tírame", "tíreme", "lance", "lánceme", "aviente", "aviénteme",
             "tienes que tirar", "puedes tirar", "échame", "écheme", "avienta", "lanza"),
    "request": ("tira un dado de seis", "tírame un dado de seis", "tire un dado de seis",
                "tíreme un dado de seis", "lánceme un dado de seis", "aviénteme un dado de seis",
                "avienta un dado de seis", "lanza un dado de seis", "tienes que tirar un dado de seis",
                "puedes tirar un dado de seis", "echa un dado de seis", "tira tu d seis"),
    "denial": ("no tires", "no tire", "no lo tires", "no lances", "no lo lances", "no quiero tirar",
               "mejor no", "no cancela", "cancela la tirada", "olvida"),
    "other": ("dado de ocho", "dado de diez", "dado de doce", "dado de veinte",
              "dos dados", "tres dados", "dados de seis", "el dado de seis es", "se tira un dado",
              "yo voy a tirar", "no tires un dado de seis", "mejor no tires un dado de seis",
              "cancela el dado de seis", "a qué hora es", "qué hora es"),
}
ARMS = {"intent": (0, 1), "words": tuple(range(7)), "contrast": tuple(range(9))}


@torch.inference_mode()
def features(model, frames, paths):
    if frames is None or len(frames) < 16:
        return None
    phones, intent, denial = model_output(model, frames)
    if denial.item() >= 0:
        return None
    logits = phones[0].numpy()
    ranks = intent[0].numpy()
    index = LABELS.index(6)
    noun, (noun_start, noun_end) = cheapest(logits, paths["object"])
    verb, (_, verb_end) = cheapest(logits[:noun_start], paths["verb"])
    cancel = min(cheapest(logits[:noun_start], paths["denial"])[0],
                 cheapest(logits[noun_end:], paths["denial"])[0])
    request, _ = cheapest(logits, paths["request"])
    other, _ = cheapest(logits, paths["other"])
    return np.array([ranks[index] - max(np.delete(ranks, index)), -denial.item(),
                     noun, verb, cancel - verb, (noun_start - verb_end) / len(logits),
                     (len(logits) - noun_end) / len(logits), request, other - request])


def main():
    parser = ArgumentParser(description=__doc__)
    for key in ("root", "output", "bank", "speech", "labelled", "chm", "frontend", "pilot", "capture", "espeak"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model, report = frozen_model(args.bank)
    vocab, examples = load_cache(args.speech)
    chm_features = args.chm / "native-negative-features"
    chm_roles = args.chm / "speaker-roles.json"
    pins = {
        "source": file_digest(Path(__file__)),
        "bank_weights": report["weights_sha256"],
        "speech": file_digest(args.speech / "index.json"),
        "labelled_turns": file_digest(args.labelled / "index.json"),
        "chm_features": file_digest(chm_features / "index.json"),
        "chm_speakers": file_digest(chm_roles),
        "frontend": file_digest(args.frontend),
        "pilot": file_digest(args.pilot),
        "capture": file_digest(args.capture),
        "espeak": file_digest(args.espeak),
    }
    synthetic_manifest = json.loads((args.speech / "index.json").read_text())
    if (synthetic_manifest["frontend_sha256"] != pins["frontend"] or
            report["teacher_cache_sha256"] != pins["speech"] or
            report["negative_cache_sha256"] != pins["chm_features"]):
        raise ValueError("Frozen bank and frontend must share the same feature sources")
    named = load_examples(args.labelled, pins["frontend"])
    negatives = load_negatives(chm_features, chm_roles, pins["frontend"])
    mapping = {symbol: index for index, symbol in enumerate(vocab)}
    paths = {name: [tuple(mapping.get(phone, len(vocab) - 1) for phone in phonemes(args.espeak, text + "."))
                    for text in row] for name, row in FORMS.items()}
    fit_groups = {}
    for split in ("train", "dev"):
        rows = [("synthetic", row["intent"] == 6 and not row["denied"], features(model, frames, paths))
                for row, frames, *_ in examples[split]]
        rows += [(row["group"], row["order"], features(model, frames, paths))
                 for row, frames, _ in named[split]]
        rows += [("conversation", False, features(model, frames, paths)) for frames in negatives[split]]
        fit_groups[split] = rows
    summaries = {}
    specs = {}
    for arm, cols in ARMS.items():
        rule = fit([(truth, vector) for _, truth, vector in fit_groups["train"]], cols)
        rule["threshold"] = max(score(vector, rule) for _, truth, vector in
                                itertools.chain(*fit_groups.values()) if not truth and vector is not None)
        summaries[arm] = {split: dict(Counter(f"{group}:{int(truth)}:{int(passed(vector, rule))}"
                                             for group, truth, vector in values))
                          for split, values in fit_groups.items()}
        specs[arm] = rule
    selected = max(ARMS, key=lambda arm: sum(count for key, count in summaries[arm]["dev"].items()
                                             if key.endswith(":1:1")))
    frozen = {"pins": pins, "frame_trim": "strip_digital_edges", "paths": paths, "selected": selected,
              "arms": specs, "scores": summaries}
    args.output.mkdir(mode=0o700)
    recipe = args.output / "frozen.json"
    recipe.write_text(json.dumps(frozen, indent=2, allow_nan=False) + "\n")
    known_synthetic = {row["pcm_sha256"] for row in synthetic_manifest["records"].values()}
    known_human = {digest for rows in named.values() for row, *_ in rows for digest in row["pcm_sha256"]}
    known_human.update(row["effective_pcm_sha256"] for row in
                       json.loads((chm_features / "index.json").read_text())["records"].values())

    def observe(pcm):
        disposition = json.loads(invoke(args.pilot, pcm, "--decision"))["disposition"]
        before = int(invoke(args.pilot, pcm))
        observed = {"before": before, "disposition": disposition}
        vector = features(model, native_frames(args.frontend, strip_digital_edges(pcm)), paths)
        observed["score"] = score(vector, specs[selected])
        for name, rule in specs.items():
            proposed = passed(vector, rule)
            observed[name] = {
                "strict": before or (6 if proposed and disposition == "no_match" else 0),
                "conflict_lab": before or (6 if proposed and disposition in (
                    "no_match", "dice_negation_conflict") else 0),
            }
        return observed

    cases, excluded = {}, []
    for identifier, split, category, target, file, checksum in recordings(args.root):
        pcm = pcm_from(file, checksum)
        views = {"raw": pcm, "retained": invoke(args.capture, pcm, "--pcm-with-hangover")}
        if any(overlaps_adjustment(payload, known_synthetic, known_human) or
               set(pcm_seals(payload)) & known_human for payload in views.values()):
            excluded.append(identifier)
            continue
        cases[identifier] = {"split": split, "category": category, "target": target,
                             **{view: observe(payload) for view, payload in views.items()}}
    print(json.dumps({"recipe_sha256": file_digest(recipe), "historical": cases, "excluded": excluded},
                     indent=2, allow_nan=False))

if __name__ == "__main__":
    main()
