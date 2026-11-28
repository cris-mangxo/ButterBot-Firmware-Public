"""Measure exact clock questions next to their event-time counterparts."""
from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
import json

import numpy as np
import torch

from capture_replay import pcm_from, recordings
from joke_contrast_lab import cheapest, fit, passed, score
from keyword_lab import LABELS, VOICE_SPLITS, render
from phoneme_confirm_lab import human_anchors, model_output, natural_adjustment_hashes
from phoneme_reject_lab import load_negatives
from phoneme_replay_lab import frozen_model, native_frames, overlaps_adjustment, strip_digital_edges
from phoneme_student_lab import load_cache
from phoneme_teacher_lab import phonemes
from time_lab import file_digest, invoke


CLOCK = 5
PAIRS = {
    "train": (
        ("Qué hora es", "A qué hora es"), ("Dime qué hora es", "Dime a qué hora es"),
        ("Me puedes decir qué hora es", "Me puedes decir a qué hora es"),
        ("Dime la hora", "Dime la hora de la cita"),
        ("Qué horas son", "Qué horas son las de atención"),
        ("Me da la hora por favor", "Me da la hora de la salida por favor"),
        ("Qué hora tienes", "Qué hora tienes para la comida"),
    ),
    "dev": (
        ("Disculpa qué hora es", "Disculpa a qué hora es"),
        ("Me dices qué hora es ahora", "Me dices a qué hora empieza"),
        ("Qué hora es ahorita", "A qué hora es la reunión"),
        ("Puedes decirme la hora actual", "Puedes decirme la hora de la entrada"),
    ),
    "test": (
        ("Oye dime qué hora es", "Oye dime a qué hora es"),
        ("Por favor qué horas son", "Por favor a qué hora es"),
        ("Me puede decir qué hora es", "Me puede decir a qué hora es la clase"),
        ("Me dices la hora ahorita", "Me dices la hora de la cita"),
    ),
}
FULL = {
    "now": tuple(pair[0] for pair in PAIRS["train"]),
    "event": tuple(pair[1] for pair in PAIRS["train"]) + (
        "a qué hora empieza", "a qué hora llega", "qué hora era", "qué hora es en otra ciudad"),
    "other": ("no me digas qué hora es", "no quiero saber la hora", "mejor no me digas la hora",
              "es la hora de comer", "hora", "horas", "tira un dado de seis", "no tires el dado",
              "dime un chiste", "pásame la mantequilla"),
}
ARMS = {"summary": (0, 1, 2, 3), "grammar": (0, 1, 2, 3, 4, 5, 6, 7, 8),
        "grammar_without_intent": (0, 2, 4, 5, 6, 7, 8)}


def forms(espeak, vocabulary):
    vocab = {phoneme: i for i, phoneme in enumerate(vocabulary)}
    text = dict(FULL, prefix=("a qué hora", "qué hora"), noun=("hora", "horas"),
                local_now=("qué hora es", "qué horas son", "dime la hora"))
    return {key: [tuple(vocab.get(phone, len(vocab) - 1) for phone in phonemes(espeak, s + "."))
                  for s in values] for key, values in text.items()}


@torch.inference_mode()
def features(model, frames, paths):
    if frames is None or len(frames) < 12:
        return None
    phones, intent, denied = model_output(model, frames)
    if denied.item() >= 0:
        return None
    logits = phones[0].numpy()
    count = len(logits)
    groups = [paths[key] for key in ("now", "event", "other")]
    templates = [sequence for group in groups for sequence in group]
    lengths = torch.tensor([len(seq) for seq in templates])
    tokens = torch.tensor([token for seq in templates for token in seq])
    log_probs = phones.log_softmax(-1).transpose(0, 1).expand(-1, len(templates), -1)
    full_costs = torch.nn.functional.ctc_loss(log_probs, tokens, torch.full((len(templates),), count),
                                             lengths, reduction="none").numpy() / count
    boundaries = np.cumsum([0, *(len(group) for group in groups)])
    costs = [min(full_costs[start:end]) for start, end in zip(boundaries, boundaries[1:])]
    if not np.isfinite(costs).all():
        return None
    now, event, other = costs
    noun, (start, end) = cheapest(logits, paths["noun"])
    direct, (direct_start, direct_end) = cheapest(logits, paths["local_now"])
    a_que = cheapest(logits[:end], paths["prefix"][:1])[0]
    que = cheapest(logits[:end], paths["prefix"][1:])[0]
    ranking = intent[0].numpy()
    index = LABELS.index(CLOCK)
    return np.array([noun, ranking[index] - max(np.delete(ranking, index)), now, -denied.item(),
                     event - now, other - now, a_que - que, (direct_end-direct_start)/count, direct])


def generate(args):
    args.output.mkdir(mode=0o700)
    parameters = [(split, n, direction, speaker, speed, pair[direction])
                  for split, pairs in PAIRS.items() for n, pair in enumerate(pairs)
                  for direction in (0, 1) for speaker, speed in VOICE_SPLITS[split]]

    def one(values):
        split, n, direction, speaker, speed, text = values
        voice_number = [v for v, _ in VOICE_SPLITS[split]].index(speaker)
        entry = f"{split}-{n:02}-{direction}-{voice_number}"
        wav = render(args.output, entry, speaker, text + "?", speed)
        return {"id": entry, "split": split, "time_now": direction == 0,
                "voice": speaker, "wav_sha256": file_digest(wav), "text": text}

    with ThreadPoolExecutor(max_workers=4) as executor:
        rows = list(executor.map(one, parameters))
    (args.output / "synthesis.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("generate", "fit", "replay"))
    for key in ("output", "root", "bank", "speech", "chm", "espeak", "frontend", "pilot", "capture",
                "podcast", "massive-extra", "agenda-calibration"):
        parser.add_argument("--" + key, type=Path, required=key == "output")
    args = parser.parse_args()
    if args.stage == "generate":
        generate(args)
        return
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model, report = frozen_model(args.bank)
    source = args.speech / "index.json"
    negatives_path = args.chm / "native-negative-features"
    speakers = args.chm / "speaker-roles.json"
    synthetic = args.output / "synthesis.json"
    recipe = args.output / "frozen.json"
    pins = {key: file_digest(path) for key, path in
            {"source": Path(__file__), "feature_cache": source, "chm": negatives_path / "index.json",
             "speakers": speakers, "frontend": args.frontend, "espeak": args.espeak, "synthetic": synthetic}.items()}
    pins["weights"] = report["weights_sha256"]
    if (report["teacher_cache_sha256"] != pins["feature_cache"] or
            report["negative_cache_sha256"] != pins["chm"] or
            json.loads(source.read_text())["frontend_sha256"] != pins["frontend"]):
        raise ValueError("Inputs differ from the frozen bank")
    rows = json.loads(synthetic.read_text())
    voices = args.root / "primer-candidato-voces"
    if args.stage == "fit":
        vocab, previous = load_cache(args.speech)
        contrasts = forms(args.espeak, vocab)
        negatives = load_negatives(negatives_path, speakers, pins["frontend"])
        labelled = {split: [] for split in VOICE_SPLITS}
        occupied = set()
        for row in rows:
            wave = args.output / (row["id"] + ".wav")
            payload = strip_digital_edges(pcm_from(wave, row["wav_sha256"]))
            occupied.add(sha256(payload).hexdigest())
            frames = native_frames(args.frontend, payload)
            labelled[row["split"]].append((row["time_now"], features(model, frames, contrasts)))
        anchor, = [pcm_from(voices / name, digest) for label, _, name, digest in human_anchors(voices)
                   if label == CLOCK]
        adjustment = [(True, features(model, native_frames(args.frontend, anchor), contrasts))]
        known = json.loads(args.agenda_calibration.read_text())
        if known["role"] != "local_calibration_only":
            raise ValueError("Agenda voices not declared for calibration")
        for donor in known["rows"]:
            file = args.agenda_calibration.parent / donor["wav"]
            pcm = pcm_from(file, donor["sha256"])
            adjustment.append((False, features(model, native_frames(args.frontend, pcm), contrasts)))
            occupied.add(sha256(strip_digital_edges(pcm)).hexdigest())
        occupied.add(sha256(strip_digital_edges(anchor)).hexdigest())
        fit_base = {part: [(row["intent"] == CLOCK and not row["denied"], features(model, matrix, contrasts))
                           for row, matrix, *_ in previous[part]] + [(False, features(model, matrix, contrasts))
                           for matrix in negatives[part]] + labelled[part] for part in ("train", "dev")}
        evaluations = {}
        for name, columns in ARMS.items():
            for transfer in (False, True):
                train = fit_base["train"] + (8 * adjustment[:1] if transfer else [])
                rule = fit(train, columns)
                rule["threshold"] = max(score(ft, rule) for target, ft in
                                        train + fit_base["dev"] + adjustment if not target and ft is not None)
                counts = {split: dict(Counter(f"{int(goal)}:{int(passed(ft, rule))}"
                                              for goal, ft in examples))
                          for split, examples in labelled.items()}
                key = name + ("_anchor" if transfer else "_synthetic")
                evaluations[key] = {"rule": rule, "pairs": counts, "anchor": passed(adjustment[0][1], rule)}
        selected = max(evaluations, key=lambda k: evaluations[k]["pairs"]["dev"].get("1:1", 0))
        with recipe.open("x") as handle:
            json.dump({"pins":pins, "forms":contrasts, "rules":evaluations, "selected":selected,
                       "occupied_pcm":sorted(occupied)},handle,ensure_ascii=False,indent=2,allow_nan=False)
        print(json.dumps({"selected":selected, "counts":{k:v["pairs"] for k,v in evaluations.items()}},
                         ensure_ascii=False, indent=2))
        return

    bank = json.loads(recipe.read_text())
    if bank["pins"] != pins:
        raise ValueError("Replay input differs from the freeze")
    paths = bank["forms"]
    rules = {name: row["rule"] for name, row in bank["rules"].items()}
    occupied = set(bank["occupied_pcm"]) | natural_adjustment_hashes(negatives_path)
    adjusted = {r["pcm_sha256"] for r in json.loads(source.read_text())["records"].values()}

    def observe(pcm):
        disposition = json.loads(invoke(args.pilot, pcm, "--decision"))["disposition"]
        baseline = int(invoke(args.pilot, pcm))
        ft = features(model, native_frames(args.frontend, strip_digital_edges(pcm)), paths)
        variants = {}
        for name, rule in rules.items():
            passed_now = passed(ft, rule)
            variants[name] = {"score":score(ft,rule),
                              "strict":baseline or (CLOCK if passed_now and disposition=="no_match" else 0),
                              "conflict_lab":baseline or (CLOCK if passed_now and disposition in
                              ("no_match","dice_negation_conflict") else 0)}
        return {"base":baseline, "disposition":disposition, "variants":variants}

    cases = list(recordings(args.root))
    if args.massive_extra:
        extra = json.loads(args.massive_extra.read_text())
        cases.append((extra["id"], "public_reserved_time", "time", CLOCK,
                      args.massive_extra.parent / extra["sealed_wav"], extra["wav_sha256"]))
    if args.podcast:
        for item in json.loads(args.podcast.read_text())["items"]:
            cases.append((item["id"], "diagnostic_podcast", "time", item["expected"],
                          args.podcast.parent/(item["id"]+".wav"), item["wave_sha256"]))
    result, excluded = {}, []
    for name, split, _, target, wav, checksum in cases:
        pcm = pcm_from(wav, checksum)
        views = {"raw":pcm,"retained":invoke(args.capture,pcm,"--pcm-with-hangover")}
        if any(overlaps_adjustment(v, adjusted, occupied) or
               sha256(strip_digital_edges(v)).hexdigest() in occupied for v in views.values()):
            if split.startswith("public"):
                raise ValueError("Human test PCM intersects the fitting data")
            excluded.append(name)
            continue
        result[name] = {"target":target,"split":split, **{k:observe(v) for k,v in views.items()}}
    print(json.dumps({"recipe_sha256":file_digest(recipe),"selected":bank["selected"],
                      "results":result,"excluded":excluded}, indent=2, allow_nan=False))

if __name__ == "__main__":
    main()
