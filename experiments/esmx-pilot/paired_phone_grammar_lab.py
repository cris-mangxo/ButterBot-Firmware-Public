"""Score complete command grammars on compact phones without using intention logits."""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path

import duckdb
import torch
from torch.nn.functional import ctc_loss

import keyword_lab as kw
from capture_replay import pcm_from
from chm_transcript_ctc_replay import historical_cases
from clock_minimal_pairs_lab import FULL as CLOCK_PHRASES
from phoneme_chm150_lab import CORPUS_SHA256, SPEAKER_SPLIT_SHA256, decode, eligible_rows
from phoneme_confirm_lab import model_output
from phoneme_replay_lab import frozen_model, native_frames, overlaps_adjustment, strip_digital_edges
from phoneme_teacher_lab import known_codes
from time_lab import file_digest, invoke


INTENTS = {1: "butter", 5: "clock", 6: "d6"}
FORMS = {
    1: (
        ("pásame la mantequilla", "dame la mantequilla", "tráeme la mantequilla",
         "me pasas la mantequilla", "me puedes pasar la mantequilla", "puedes traerme la mantequilla",
         "pásame la mantequilla por favor", "dame tantita mantequilla"),
        ("no me pases la mantequilla", "no me traigas la mantequilla", "pásame la mantequilla mejor no",
         "pásame la mantequilla de cacahuate", "puedes agregar la mantequilla", "me pasaste la mantequilla",
         "dónde está la mantequilla", "dale tantita mantequilla", "dame la sal", "mantequilla"),
    ),
    5: (CLOCK_PHRASES["now"], CLOCK_PHRASES["event"] + CLOCK_PHRASES["other"]),
    6: (
        ("tira un dado de seis", "tírame un dado de seis", "lanza un dado de seis", "avienta un dado de seis",
         "tira un dado de seis caras", "tíreme un dado de seis", "por favor tire un dado de seis",
         "puedes tirar un dado de seis"),
        ("no tires un dado de seis", "no me tire un dado de seis", "tire un dado de seis mejor no",
         "tira un dado de seis no cancélalo", "no lo tires", "tírame un dado de ocho",
         "lanza dos dados de seis", "si atacas avienta un dado de seis", "el manual dice tira un dado de seis caras",
         "tira un dado de veinte", "dado de seis", "el dado de seis es", "tira un dado", "dime un chiste"),
    ),
}
PAIR_FIELDS = {1: ("expected", "wav_sha256"), 5: ("time_now", "wav_sha256"), 6: ("d6", "sha256")}
PAIR_COUNTS = {1: {"train": 48, "dev": 20, "test": 24}, 5: {"train": 42, "dev": 16, "test": 16},
               6: {"train": 48, "dev": 16, "test": 24}}
BANKS = ("initial", "reject_only", "transcript_035", "transcript_100")
FORMS_OF_SCORE = {"margin": 1, "evidence": 2}


class Grammar:
    def __init__(self, paths):
        indexes = sorted(INTENTS)
        self.accepted = {}
        self.competitors = {}
        alternatives = []
        own_controls = {}
        for intent in indexes:
            orders, controls = paths[str(intent)]
            for group, collection in ((orders, self.accepted), (controls, own_controls)):
                collection[intent] = list(range(len(alternatives), len(alternatives) + len(group)))
                alternatives.extend(group)
        for intent in indexes:
            self.competitors[intent] = own_controls[intent] + [
                index for rival in indexes if rival != intent for index in self.accepted[rival]
            ]
        if any(not row or min(row) <= 0 for row in alternatives):
            raise ValueError("A command path must be nonempty and exclude CTC blank")
        self.lengths = torch.tensor([len(row) for row in alternatives])
        self.tokens = torch.tensor([token for row in alternatives for token in row])
        self.minimum_frames = torch.tensor([
            len(row) + sum(a == b for a, b in zip(row, row[1:])) for row in alternatives
        ])

    def scores(self, logits):
        n = len(logits)
        if n < 12 or not torch.isfinite(logits).all():
            return {}
        log_probs = logits.log_softmax(-1)[:, None, :].expand(-1, len(self.lengths), -1)
        costs = ctc_loss(
            log_probs, self.tokens, torch.full_like(self.lengths, n), self.lengths,
            reduction="none",
        ).masked_fill(self.minimum_frames > n, float("inf")) / n
        values = {}
        for intent, group in self.accepted.items():
            positive = costs[group].amin().item()
            opponent = costs[self.competitors[intent]].amin().item()
            if isfinite(positive) and isfinite(opponent):
                values[str(intent)] = {key: opponent - multiplier * positive
                                       for key, multiplier in FORMS_OF_SCORE.items()}
        return values


@torch.inference_mode()
def measure(pcm, args, grammar, models):
    frames = native_frames(args.frontend, strip_digital_edges(pcm))
    if frames is None:
        return {}
    logits, _, denial = model_output(models["initial"], frames)
    # Keep a single negation decision. Adapted intentions and negation never vote.
    if denial.item() >= 0 or not torch.isfinite(denial).all():
        return {}
    return {name: grammar.scores(logits[0] if name == "initial" else model_output(model, frames)[0][0])
            for name, model in models.items()}


def prepare_models(args):
    models = {}
    hashes = {}
    synthetic_sha = file_digest(args.speech)
    manifest = args.adaptations / "local-input.json"
    manifest_sha = file_digest(manifest)
    for name in BANKS:
        folder = args.base if name == "initial" else args.adaptations / name
        model, report = frozen_model(folder)
        if report["teacher_cache_sha256"] != synthetic_sha:
            raise ValueError("Model has another synthetic cache")
        if name != "initial" and (report["initial_weights_sha256"] != hashes["initial"]
                                  or report["human_manifest_sha256"] != manifest_sha):
            raise ValueError("Adapted model has another initialization or voice partition")
        hashes[name] = report["weights_sha256"]
        models[name] = model
    return models, hashes, manifest


def load_development(args, manifest):
    data = json.loads(manifest.read_text())
    roles = json.loads(args.speakers.read_text())["speaker_roles"]
    selected = {row["id"]: row for row in data["entries"]["dev"]}
    if len(selected) != 494 or any(roles.get(row["speaker_partition_id"]) != "dev" for row in selected.values()):
        raise ValueError("Expected the 494 pinned development turns")
    options = {"autoload_known_extensions": False, "autoinstall_known_extensions": False}
    with duckdb.connect(config=options) as connection:
        for ident, speaker, wav in connection.execute(
            "SELECT audio_id, speaker_id, audio.bytes FROM read_parquet(?) WHERE audio_id=ANY(?) ORDER BY audio_id",
            [str(args.corpus), sorted(selected)],
        ).fetchall():
            pcm = decode(args.ffmpeg, wav)
            saved = selected.pop(ident)
            if speaker != saved["speaker_partition_id"] or sha256(kw.trimmed(pcm)).hexdigest() != saved["pcm_sha256"]:
                raise ValueError("Human calibration PCM differs from the frozen manifest")
            yield pcm
    if selected:
        raise ValueError("Incomplete human development data")


def proposals(scores, thresholds, name, criterion):
    passed = [int(intent) for intent, values in scores.get(name, {}).items()
              if values[criterion] > thresholds[name][intent][criterion]]
    return passed[0] if len(passed) == 1 else 0


def fit(args, grammar, models, manifest, pins, paths):
    folders = {1: args.butter_pairs, 5: args.clock_pairs, 6: args.d6_pairs}
    examples = []
    prepared = set()
    for intent, folder in folders.items():
        truth_key, hash_key = PAIR_FIELDS[intent]
        seen = set()
        counts = Counter()
        for row in json.loads((folder / "synthesis.json").read_text()):
            name, split = row["id"], row["split"]
            if (split not in ("train", "dev", "test") or not name.startswith(split + "-")
                    or not all(c.isalnum() or c == "-" for c in name) or name in seen):
                raise ValueError("Unexpected paired fixture")
            seen.add(name)
            counts[split] += 1
            pcm = pcm_from(folder / (name + ".wav"), row[hash_key])
            normalized = strip_digital_edges(pcm)
            prepared.add(sha256(normalized).hexdigest())
            intent_expected = intent if row[truth_key] else 0
            examples.append((split, intent, intent_expected, measure(pcm, args, grammar, models)))
        if counts != PAIR_COUNTS[intent]:
            raise ValueError("Paired fixture count differs from its declared split")
    for pcm in load_development(args, manifest):
        examples.append(("dev_human", 0, 0, measure(pcm, args, grammar, models)))
    thresholds = {}
    counters = {}
    for bank in models:
        thresholds[bank] = {}
        for intent in map(str, INTENTS):
            thresholds[bank][intent] = {}
            for criterion in FORMS_OF_SCORE:
                controls = [scores[bank][intent][criterion] for split, _, expected, scores in examples
                            if split != "test" and expected != int(intent) and intent in scores.get(bank, {})]
                if not controls:
                    raise ValueError("Each intent needs calibration negatives")
                thresholds[bank][intent][criterion] = max(controls)
        counters[bank] = {}
        for criterion in FORMS_OF_SCORE:
            counts = Counter()
            for split, focal, expected, scores in examples:
                actual = proposals(scores, thresholds, bank, criterion)
                counts[f"{split}:{focal}:{expected}:{actual}"] += 1
            counters[bank][criterion] = counts
    return {"pins": pins, "paths": paths, "thresholds": thresholds, "synthesis_and_development": counters,
            "occupied_paired_pcm": sorted(prepared)}


def replay(args, grammar, models, recipe, manifest):
    occupied = {item["pcm_sha256"] for rows in json.loads(manifest.read_text())["entries"].values() for item in rows}
    pairs = set(recipe["occupied_paired_pcm"])
    synthetic = {row["pcm_sha256"] for row in json.loads(args.speech.read_text())["records"].values()}
    policies = {"strict": {"no_match"}, "conflict_diagnostic": {"no_match", "dice_negation_conflict"}}

    def observe(pcm):
        prior = int(invoke(args.host, pcm))
        disposition = json.loads(invoke(args.host, pcm, "--decision"))["disposition"]
        measured = measure(pcm, args, grammar, models) if not prior else {}
        variants = {}
        for bank in models:
            for criterion in FORMS_OF_SCORE:
                proposal = proposals(measured, recipe["thresholds"], bank, criterion)
                variants[bank + ":" + criterion] = {key: proposal if disposition in allowed else 0
                                                     for key, allowed in policies.items()}
        return {"baseline": prior, "disposition": disposition, "hypotheses": variants}

    rows = {}
    excluded = []
    for name, split, kind, expected, wav, seal in historical_cases(args.root, args.butter_diagnostic):
        pcm = pcm_from(wav, seal)
        captured = invoke(args.capture, pcm, "--pcm-with-hangover")
        if any(overlaps_adjustment(audio, synthetic, occupied)
               or sha256(strip_digital_edges(audio)).hexdigest() in pairs for audio in (pcm, captured)):
            if not split.startswith("synthetic"):
                raise ValueError("Human replay overlaps prior fitting or calibration")
            excluded.append(name)
            continue
        rows[name] = {"expected": expected, "split": split, "kind": kind,
                      "raw": observe(pcm), "retained": observe(captured)}
    if len(rows) + len(excluded) != 116:
        raise ValueError("Incomplete historic replay")

    counts = Counter()
    seen_speakers = set()
    options = {"autoload_known_extensions": False, "autoinstall_known_extensions": False}
    held = {speaker for speaker, role in json.loads(args.speakers.read_text())["speaker_roles"].items() if role == "heldout"}
    with duckdb.connect(config=options) as connection:
        for _, speaker, _, wav in eligible_rows(connection, args.corpus, held):
            pcm = decode(args.ffmpeg, wav)
            if not 0 < len(pcm) <= 128_000 or len(pcm) % 2:
                continue
            if overlaps_adjustment(pcm, synthetic, occupied):
                raise ValueError("Held-out conversation duplicates model input")
            seen_speakers.add(speaker)
            item = observe(pcm)
            counts["total"] += 1
            counts[f"base:{item['baseline']}"] += 1
            for name, result in item["hypotheses"].items():
                for key, value in result.items():
                    if value:
                        counts[name + ":" + key] += 1
    if counts["total"] != 536 or len(seen_speakers) != 30:
        raise ValueError("Incomplete conversation replay")
    return {"recipes_sha256": file_digest(args.recipe), "historical": rows, "excluded_synthetic": excluded,
            "conversation": counts, "conversation_speakers": len(seen_speakers)}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fit", "replay"))
    for key in ("recipe", "base", "adaptations", "speech", "frontend", "corpus", "speakers", "ffmpeg", "espeak",
                "butter-pairs", "clock-pairs", "d6-pairs", "root", "host", "capture", "butter-diagnostic"):
        parser.add_argument("--" + key, required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if file_digest(args.corpus) != CORPUS_SHA256 or file_digest(args.speakers) != SPEAKER_SPLIT_SHA256:
        raise ValueError("Unexpected conversation corpus")
    models, hashes, manifest = prepare_models(args)
    speech = json.loads(args.speech.read_text())
    pins = {"models": hashes, "manifest": file_digest(manifest), "script": file_digest(Path(__file__)),
            **{key: file_digest(getattr(args, key)) for key in ("speech", "frontend", "corpus", "speakers",
                                                               "ffmpeg", "espeak", "host", "capture")},
            **{key: file_digest(getattr(args, key) / "synthesis.json")
               for key in ("butter_pairs", "clock_pairs", "d6_pairs")}}
    if speech["frontend_sha256"] != pins["frontend"]:
        raise ValueError("Unexpected native frontend")
    codes = {phone: n for n, phone in enumerate(speech["vocabulary"])}
    if codes.get("<pad>") != 0:
        raise ValueError("CTC blank is not index zero")
    if args.stage == "fit":
        paths = {str(key): [[known_codes(args.espeak, item + ".", codes) for item in group] for group in groups]
                 for key, groups in FORMS.items()}
        recipe = fit(args, Grammar(paths), models, manifest, pins, paths)
        with args.recipe.open("x") as file:
            json.dump(recipe, file, indent=2, allow_nan=False)
        result = recipe["synthesis_and_development"]
    else:
        recipe = json.loads(args.recipe.read_text())
        if recipe["pins"] != pins:
            raise ValueError("Inputs changed since grammar freeze")
        result = replay(args, Grammar(recipe["paths"]), models, recipe, manifest)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
