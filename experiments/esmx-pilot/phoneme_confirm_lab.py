"""Verify a frozen intent using local CTC phoneme evidence."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from math import ceil
from pathlib import Path

import numpy as np
import torch

import keyword_lab as kw
from capture_replay import pcm_from
from phoneme_reject_lab import BASE_SHA256, load_negatives
from phoneme_replay_lab import evaluate, frozen_model, native_frames, summarize
from phoneme_student_lab import gate, load_cache
from phoneme_teacher_lab import phonemes
from time_lab import file_digest, invoke


PATTERNS = {
    1: ("mantequilla",),
    3: ("chiste", "chistes"),
    5: ("hora", "horas"),
    6: ("dado de seis", "dado que tiene seis", "dado que sea de seis"),
}
CLOCK_CONTEXT = ("qué hora es", "qué horas son", "dime la hora", "la hora actual")
METHOD = "bb-frozen-ctc-word-1"
COVERAGE = 0.80
HUMAN_MANIFESTS = {
    "manifest-ajuste.json": "af68f9a947215f5a0a52d4c65090135a3d04ebab055957ca48837379db06c8d0",
    "manifest-lexico-esmx.json": "4804e4a4cfc24ae54436f477e5c13cf820c8f1b360b22acfe131fff4331970a2",
}


def local_ctc_cost(logits, tokens):
    """Minimum deficit from framewise maxima for a completion anywhere in the turn."""
    labels = np.zeros(2 * len(tokens) - 1, dtype=np.int64)
    labels[::2] = tokens
    skips = np.zeros(len(labels), dtype=bool)
    skips[2::2] = labels[2::2] != labels[:-2:2]
    deficits = logits.max(axis=1, keepdims=True) - logits[:, labels]
    state = np.full(len(labels), np.inf)
    best = np.inf
    for costs in deficits:
        # Starting at this frame is free. Internal skips must respect CTC repeats.
        advance = np.concatenate(([0.0], state[:-1]))
        skip = np.full_like(state, np.inf)
        skip[2:] = np.where(skips[2:], state[:-2], np.inf)
        state = np.minimum(np.minimum(state, advance), skip) + costs
        best = min(best, state[-1])
    return float(best / len(tokens))


def templates(espeak, vocabulary, clock_context=False):
    symbols = {key: index for index, key in enumerate(vocabulary)}
    paths = {}
    for intent, choices in PATTERNS.items():
        if intent == 5 and clock_context:
            choices = CLOCK_CONTEXT
        # End punctuation makes eSpeak flush the last phoneme in stdin mode.
        alternatives = [phonemes(str(espeak), phrase + ".") for phrase in choices]
        if intent == 1:
            # This word has a palatal glide sequence in eSpeak. Also allow
            # the single palatal consonant available in the Spanish inventory.
            alternative = alternatives[0]
            if alternative[-3:] == ("j", "j", "a"):
                alternatives.append((*alternative[:-3], "ʝ", "a"))
        paths[intent] = [tuple(symbols[phone] for phone in sequence) for sequence in alternatives]
    return paths


def model_output(model, frames):
    tensor = torch.from_numpy(frames.astype(np.float32).T[None] / 32)
    return model(tensor, torch.tensor([len(frames)]))


@torch.inference_mode()
def observe(model, frames, patterns):
    if frames is None:
        return 0, None
    phones, logits, denied = model_output(model, frames)
    intent = kw.LABELS[int(gate(logits, denied).item())]
    if not intent:
        return 0, None
    output = phones[0].numpy()
    return intent, min(local_ctc_cost(output, path) for path in patterns[intent])


def human_anchors(directory):
    sources = {}
    for name, expected in HUMAN_MANIFESTS.items():
        path = directory / name
        if file_digest(path) != expected:
            raise ValueError("Human anchor manifest is outside the approved development inputs")
        sources[name] = json.loads(path.read_text())
    clock, = sources["manifest-ajuste.json"]["clips"]
    if (clock["split"] != "train_candidate" or clock["intent"] != "hora"
            or clock["file"] != kw.PUB_CONTEXT[0] or clock["wav_sha256"] != kw.PUB_CONTEXT[1]):
        raise ValueError("Unexpected complete clock anchor")
    yield 5, "spanishpod-time", clock["file"], clock["wav_sha256"]
    for row in sources["manifest-lexico-esmx.json"]["rows"]:
        keyword = row["keyword"]
        if row["isolated_asr"] in ("mantequilla", "un chiste"):
            clip = row["lexical_audio"]
            yield {"mantequilla": 1, "chiste": 3}[keyword], keyword, clip["file"], clip["sha256"]


@torch.inference_mode()
def anchor_thresholds(args, model, patterns, thresholds):
    evidence = {}
    for intent, label, name, checksum in human_anchors(args.human_anchors):
        path = (args.human_anchors / name).resolve(strict=True)
        if path.parent != args.human_anchors:
            raise ValueError("Anchor audio must belong to the public voice development directory")
        pcm = pcm_from(path, checksum)
        frames = native_frames(args.features, pcm)
        if frames is None:
            raise ValueError("Human anchor has no admissible frames")
        scores = model_output(model, frames)[0][0].numpy()
        cost = min(local_ctc_cost(scores, tokens) for tokens in patterns[intent])
        if not np.isfinite(cost):
            raise ValueError("Human anchor has no complete CTC path")
        thresholds[intent] = max(thresholds[intent], cost)
        evidence[label] = {"intent": intent, "ctc_cost": cost, "wav_sha256": checksum,
                           "native_pcm_sha256": sha256(kw.trimmed(pcm)).hexdigest()}
    if {item["intent"] for item in evidence.values()} != {1, 3, 5}:
        raise ValueError("Expected the three designated human development anchors")
    return evidence


def thresholds_for(model, samples, patterns):
    costs = {intent: [] for intent in patterns}
    for row, frames, *_ in samples:
        intent, cost = observe(model, frames, patterns)
        if intent and intent == row["intent"] and not row["denied"]:
            if not np.isfinite(cost):
                raise ValueError("Expected a realizable training phoneme template")
            costs[intent].append(cost)
    if not all(costs.values()):
        raise ValueError("Every intention needs an affirmative training example")
    thresholds = {intent: sorted(group)[ceil(COVERAGE * len(group)) - 1] for intent, group in costs.items()}
    return thresholds, {intent: len(group) for intent, group in costs.items()}


def confirmed(intent, cost, thresholds):
    return intent if intent and cost <= thresholds[intent] else 0


def compare_features(model, rows, patterns, thresholds):
    before, after, rejected = Counter(), Counter(), Counter()
    for expected, denied, frames in rows:
        intent, cost = observe(model, frames, patterns)
        filtered = confirmed(intent, cost, thresholds)
        before[f"{expected}:{intent}"] += 1
        after[f"{expected}:{filtered}"] += 1
        if denied:
            rejected["turns"] += 1
            rejected["initial_activations"] += bool(intent)
            rejected["verified_activations"] += bool(filtered)
    return {"initial": before, "verified": after, "synthetic_cancellations": rejected}


def freeze(args, model, source):
    vocabulary, speech = load_cache(args.cache)
    patterns = templates(args.espeak, vocabulary, args.clock_context)
    # Thresholds use affirmative synthetic training turns. Human and synthetic
    # development are measured only after the thresholds have been fixed.
    thresholds, fitted_counts = thresholds_for(model, speech["train"], patterns)
    anchors = {}
    if args.human_anchors:
        if args.features is None or file_digest(args.features) != source["frontend_sha256"]:
            raise ValueError("Human anchors need the same feature extractor as the synthetic cache")
        anchors = anchor_thresholds(args, model, patterns, thresholds)
    negative = load_negatives(args.negatives, args.speaker_split, source["frontend_sha256"])
    report = {"schema": METHOD, "bank_sha256": BASE_SHA256, "cache_sha256": file_digest(args.cache / "index.json"),
              "negative_cache_sha256": file_digest(args.negatives / "index.json"),
              "speaker_split_sha256": file_digest(args.speaker_split), "coverage": COVERAGE, "templates": patterns,
              "thresholds": thresholds, "training_affirmatives": fitted_counts,
              "synthetic_development": compare_features(
                  model, ((row["intent"], row["denied"], frames) for row, frames, *_ in speech["dev"]),
                  patterns, thresholds,
              ),
              "chm150_development": compare_features(
                  model, ((0, False, frames) for frames in negative["dev"]), patterns, thresholds,
              )}
    if anchors:
        report["human_development_anchors"] = anchors
    if args.clock_context:
        report["clock_context"] = list(CLOCK_CONTEXT)
    with args.guard.open("x") as out:
        out.write(json.dumps(report, indent=2) + "\n")
    return report


def parameters(args, vocabulary):
    data = json.loads(args.guard.read_text())
    if (data["schema"] != METHOD or data["bank_sha256"] != BASE_SHA256
            or data["cache_sha256"] != file_digest(args.cache / "index.json")
            or data["negative_cache_sha256"] != file_digest(args.negatives / "index.json")
            or data["speaker_split_sha256"] != file_digest(args.speaker_split)):
        raise ValueError("Guard provenance does not match the requested replay")
    patterns = {int(label): value for label, value in data["templates"].items()}
    thresholds = {int(label): value for label, value in data["thresholds"].items()}
    if (patterns.keys() != PATTERNS.keys() or thresholds.keys() != PATTERNS.keys()
            or any(not group or any(not path or any(type(symbol) is not int or not 0 < symbol < len(vocabulary) - 1
                                                   for symbol in path) for path in group) for group in patterns.values())
            or any(not np.isfinite(limit) or limit < 0 for limit in thresholds.values())):
        raise ValueError("Invalid CTC template or deficit limit")
    anchor_pcm = {item["native_pcm_sha256"] for item in data.get("human_development_anchors", {}).values()}
    return patterns, thresholds, anchor_pcm


def replay(args, model, source):
    patterns, thresholds, anchors = parameters(args, source["vocabulary"])
    if source["frontend_sha256"] != file_digest(args.features):
        raise ValueError("Unexpected feature extractor")

    def prediction(selected, features, pcm):
        return confirmed(*observe(selected, native_frames(features, pcm), patterns), thresholds)

    adjustment = {row["pcm_sha256"] for row in source["records"].values()}
    rows = list(evaluate(args, model, adjustment, anchors, predict=prediction))
    original = list(evaluate(args, model, adjustment, anchors))
    pure = {row["id"] for row in rows[:112]
            if not any(row[mode]["matches_adjustment_pcm"] for mode in ("raw", "retained"))}
    panels = {"historical_without_exact_adjustment_pcm": pure}
    for key in ("public_reserved_time", "public_reserved_d6", "supplemental_human_time"):
        panels[key] = {row["id"] for row in rows if row["split"] == key}
    summaries = {}
    for name, ids in panels.items():
        summaries[name] = {"items": len(ids)}
        for label, episode in (("initial", original), ("verified", rows)):
            panel = [row for row in episode if row["id"] in ids]
            summaries[name][label] = {mode: summarize(panel, mode) for mode in ("raw", "retained")}
    diagnostics = []
    if args.diagnostic:
        item = json.loads((args.diagnostic / "sealed.json").read_text())
        pcm = pcm_from(args.diagnostic / "mantequilla.wav", item["wav_sha256"])
        if sha256(kw.trimmed(pcm)).hexdigest() in anchors:
            raise ValueError("Human diagnostic overlaps an anchor")
        diag = {"classification": item["status"], "wav_sha256": item["wav_sha256"], "expected": 1}
        for mode, sample in (("raw", pcm), ("retained", invoke(args.capture, pcm, "--pcm-with-hangover"))):
            intent, cost = observe(model, native_frames(args.features, sample), patterns)
            diag[mode] = {"initial": intent, "ctc_cost": cost, "verified": confirmed(intent, cost, thresholds)}
        diagnostics.append(diag)
    return {"guard_sha256": file_digest(args.guard), "summary": summaries, "diagnostic": diagnostics,
            "episodes": [row for row in rows if row["id"] in {
                "mty-042-04-15", "coerll-int02c-hora-esmx", "public-03", "public-37", "public-81",
                "public-88", "public-89", "veto-fact",
            }]}


def chm_replay(args, model, source):
    import duckdb
    from phoneme_chm150_lab import CORPUS_SHA256, SPEAKER_SPLIT_SHA256, decode, eligible_rows

    patterns, thresholds, anchors = parameters(args, source["vocabulary"])
    if (file_digest(args.corpus) != CORPUS_SHA256 or file_digest(args.speaker_split) != SPEAKER_SPLIT_SHA256
            or source["frontend_sha256"] != file_digest(args.features)):
        raise ValueError("Unexpected CHM150 corpus, speaker split or feature extractor")
    role = json.loads(args.speaker_split.read_text())["speaker_roles"]
    speakers = {who for who, assignment in role.items() if assignment == "heldout"}
    adjustment = {row["effective_pcm_sha256"]
                  for row in json.loads((args.negatives / "index.json").read_text())["records"].values()}
    adjustment.update(anchors)
    compact, initial, verified = Counter(), Counter(), Counter()
    tested = set()
    count = 0
    with duckdb.connect(config={"autoload_known_extensions": False, "autoinstall_known_extensions": False}) as connection:
        for _, speaker, _, blob in eligible_rows(connection, args.corpus, speakers):
            pcm = decode(args.ffmpeg, blob)
            if not 0 < len(pcm) <= 128_000 or len(pcm) % 2:
                continue
            if sha256(kw.trimmed(pcm)).hexdigest() in adjustment:
                raise ValueError("Reserved CHM150 waveform intersects adjustment or development")
            tested.add(speaker)
            count += 1
            old = int(invoke(args.reference, pcm))
            if old:
                compact[old] += 1
                continue
            intent, cost = observe(model, native_frames(args.features, pcm), patterns)
            if intent:
                initial[intent] += 1
            accepted = confirmed(intent, cost, thresholds)
            if accepted:
                verified[accepted] += 1
    return {"guard_sha256": file_digest(args.guard), "utterances": count, "speakers": len(tested),
            "compact": compact, "initial_added": initial, "verified_added": verified}


def existing(path):
    return Path(path).resolve(strict=True)


def main():
    common = argparse.ArgumentParser(add_help=False)
    for key in ("base", "cache", "negatives", "speaker-split"):
        common.add_argument("--" + key, type=existing, required=True)
    common.add_argument("--guard", type=Path, required=True)
    parser = argparse.ArgumentParser(description=__doc__)
    children = parser.add_subparsers(dest="stage", required=True)
    preparation = children.add_parser("freeze", parents=[common])
    preparation.add_argument("--espeak", type=existing, required=True)
    preparation.add_argument("--human-anchors", type=existing)
    preparation.add_argument("--features", type=existing)
    preparation.add_argument("--clock-context", action="store_true")
    evaluation = children.add_parser("replay", parents=[common])
    for key in ("root", "features", "reference", "capture", "coerll-supplemental", "diagnostic"):
        evaluation.add_argument("--" + key, type=existing, required=key not in ("coerll-supplemental", "diagnostic"))
    chm = children.add_parser("replay-chm", parents=[common])
    for key in ("corpus", "features", "reference", "ffmpeg"):
        chm.add_argument("--" + key, type=existing, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    model, report = frozen_model(args.base)
    index = args.cache / "index.json"
    if report["weights_sha256"] != BASE_SHA256 or report["teacher_cache_sha256"] != file_digest(index):
        raise ValueError("Unexpected frozen bank or synthetic cache")
    source = json.loads(index.read_text())
    result = {"freeze": freeze, "replay": replay, "replay-chm": chm_replay}[args.stage](args, model, source)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
