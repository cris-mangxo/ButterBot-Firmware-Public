"""Adapt the compact encoder on labelled D6 speech and whole-turn contrasts."""

import argparse
from collections import Counter
from hashlib import sha256
from itertools import cycle
import json
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch.nn import functional as F

from capture_replay import pcm_from
import keyword_lab as kw
from d6_teacher_guard_lab import (
    HUMAN_SHA256, PHRASES, STUDENT_SHA256, SYNTHETIC_SHA256, eligible_pcm, exterior_phone_counts, pcm_seals, scope_cases,
)
from phoneme_confirm_lab import local_ctc_alignment, model_output
from phoneme_reject_lab import load_negatives, pack_negative
from phoneme_replay_lab import evaluate, frozen_model, load_int8_weights, native_frames, strip_digital_edges, summarize
from phoneme_student_lab import RECIPE, BATCH, load_cache, measure, quantize, supervised_loss
from phoneme_teacher_lab import WEIGHTS_SHA256, align_frontend, load_teacher, phonemes, teacher_outputs
from time_lab import file_digest, invoke


PROCEDURE = "bb-compact-d6-adaptation-1"
CONTEXT_SCHEMA = "bb-compact-d6-context-1"
FIXTURE = Path(__file__).with_name("compact_d6_train_cases.json")
VOICES = (("Paulina", 185), ("Eddy (Español (México))", 210), ("Sandy (Español (México))", 200))
SEED = 20261201
EPOCHS = 28
D6 = kw.LABELS.index(6)


def synthetic_source(path):
    if file_digest(path) != SYNTHETIC_SHA256:
        raise ValueError("Unexpected sealed synthetic bank")
    return json.loads(path.read_text())


def inputs(args):
    manifest = args.human / "manifest.json"
    if file_digest(manifest) != HUMAN_SHA256:
        raise ValueError("Unexpected human speaker allocation")
    for row in json.loads(manifest.read_text())["samples"]:
        path = args.human / "audio" / (row["id"] + ".wav")
        yield row["id"], row["split"], "human", row["label"] == 1, (
            1 if row["label"] == 3 else 0 if row["label"] == 1 else None
        ), pcm_from(path, row["wav_sha256"])
    with TemporaryDirectory(prefix="bb-compact-d6-") as directory:
        for group, phrases in json.loads(FIXTURE.read_text()).items():
            for number, text in enumerate(phrases):
                for variant, (voice, rate) in enumerate(VOICES):
                    identifier = f"context-{group}-{number:02}-{variant}"
                    path = kw.render(Path(directory), identifier, voice, text, rate)
                    yield identifier, "train", "synthetic_context", group == "orders", int(group == "cancelled"), pcm_from(path)


def prepare(args):
    source = synthetic_source(args.synthetic_index)
    frontend_sha = file_digest(args.features)
    if source["frontend_sha256"] != frontend_sha:
        raise ValueError("Unexpected native feature extractor")
    teacher, extractor, vocab = load_teacher(args.teacher)
    symbols = {symbol: index for index, symbol in enumerate(source["vocabulary"])}
    lookup = torch.empty(len(vocab), dtype=torch.long)
    for symbol, index in vocab.items():
        lookup[index] = symbols.get(symbol, symbols["<other>"])
    rows = []
    for identifier, split, group, order, denied, pcm in inputs(args):
        # Keep the same effective signal for the feature extractor and teacher.
        if not eligible_pcm(pcm):
            raise ValueError(f"Out-of-window training input: {identifier}")
        aligned = strip_digital_edges(pcm)
        frames = native_frames(args.features, aligned)
        if frames is None:
            raise ValueError(f"No usable native frames for {identifier}")
        samples = np.frombuffer(kw.trimmed(aligned), dtype="<i2").astype(np.float32) / 32_768
        row = {"id": identifier, "split": split, "group": group, "order": order, "denied": denied,
               "pcm_sha256": pcm_seals(pcm), "frames": len(frames)}
        rows.append((row, samples, frames))
    args.output.mkdir(mode=0o700)
    records = {}
    for offset in range(0, len(rows), 4):
        batch = rows[offset:offset + 4]
        predictions = teacher_outputs(teacher, extractor, batch, lookup, len(symbols))
        for (row, _, frames), posterior in zip(batch, predictions, strict=True):
            path = args.output / (row["id"] + ".npz")
            aligned = align_frontend(posterior, len(frames)).astype(np.float16)
            np.savez_compressed(path, frames=frames, teacher=aligned)
            records[row["id"]] = {**row, "artifact_sha256": file_digest(path)}
    manifest = {"schema": PROCEDURE, "teacher_sha256": WEIGHTS_SHA256, "frontend_sha256": frontend_sha,
                "synthetic_sha256": SYNTHETIC_SHA256, "human_sha256": HUMAN_SHA256,
                "fixture_sha256": file_digest(FIXTURE), "records": records, "labels": len(symbols)}
    target = args.output / "index.json"
    target.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return {"feature_cache_sha256": file_digest(target), "turns": len(records)}


def load_examples(directory, frontend_sha):
    manifest = json.loads((directory / "index.json").read_text())
    if (manifest["schema"] != PROCEDURE or manifest["frontend_sha256"] != frontend_sha
            or manifest["teacher_sha256"] != WEIGHTS_SHA256 or manifest["synthetic_sha256"] != SYNTHETIC_SHA256
            or manifest["human_sha256"] != HUMAN_SHA256 or manifest["fixture_sha256"] != file_digest(FIXTURE)):
        raise ValueError("Unexpected labelled turn cache")
    partitions = {"train": [], "dev": []}
    pcms = {split: set() for split in partitions}
    for key, row in manifest["records"].items():
        if (key != row["id"] or not key.replace("-", "").replace("_", "").isalnum()
                or row["split"] not in partitions or (row["split"] == "dev" and row["group"] != "human")):
            raise ValueError("Unexpected label or split")
        path = directory / (key + ".npz")
        if file_digest(path) != row["artifact_sha256"]:
            raise ValueError("Labelled feature archive differs from its seal")
        with np.load(path, allow_pickle=False) as archive:
            frames, teacher = archive["frames"], archive["teacher"]
        if (frames.shape != (row["frames"], kw.BANDS) or frames.dtype != np.int8
                or not 8 <= len(frames) <= 199 or teacher.shape != (len(frames), manifest["labels"])
                or not np.isfinite(teacher).all() or not np.allclose(teacher.sum(-1), 1, atol=2e-3)):
            raise ValueError("Invalid labelled feature geometry")
        partitions[row["split"]].append((row, frames, teacher))
        pcms[row["split"]].update(row["pcm_sha256"])
    if not all(partitions.values()) or pcms["train"] & pcms["dev"]:
        raise ValueError("Empty partition or role overlap in labelled turns")
    return partitions


def labelled_loss(model, batch):
    packed, lengths = pack_negative([frames for _, frames, _ in batch])
    phones, intent, denial = model(packed, lengths)
    target = torch.tensor([D6 if row["order"] else 0 for row, *_ in batch])
    weights = torch.tensor([(2.0 if row["group"] == "human" else 1.0) * (2.0 if row["order"] else 1.0)
                            for row, *_ in batch])
    logphones = torch.cat([phones[index, :length].log_softmax(-1) for index, length in enumerate(lengths)])
    teacher = torch.from_numpy(np.concatenate([posterior for _, _, posterior in batch]).astype(np.float32))
    teacher = teacher / teacher.sum(-1, keepdim=True)
    loss = F.kl_div(logphones, teacher, reduction="batchmean") * 0.45
    loss += (F.cross_entropy(intent, target, reduction="none") * weights).mean()
    known = [i for i, (row, *_) in enumerate(batch) if row["denied"] is not None]
    if known:
        labels = torch.tensor([batch[i][0]["denied"] for i in known], dtype=torch.float32)
        loss += 0.35 * F.binary_cross_entropy_with_logits(denial[known], labels)
    return loss


def adapt(model, commands, labelled, negatives):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    labels = torch.tensor([kw.LABELS.index(row["intent"]) for row, *_ in commands])
    weights = torch.bincount(labels, minlength=len(kw.LABELS)).float().rsqrt()
    for _ in range(EPOCHS):
        command_order = torch.randperm(len(commands)).tolist()
        labelled_order = cycle(torch.randperm(len(labelled)).tolist())
        negative_order = cycle(torch.randperm(len(negatives)).tolist())
        for offset in range(0, len(command_order), BATCH):
            batch = [commands[i] for i in command_order[offset:offset + BATCH]]
            mixed = [labelled[next(labelled_order)] for _ in range(12)]
            outside, lengths = pack_negative([negatives[next(negative_order)] for _ in range(12)])
            _, other_intent, _ = model(outside, lengths)
            loss = (supervised_loss(model, batch, weights) + 0.60 * labelled_loss(model, mixed)
                    + 0.50 * F.cross_entropy(other_intent, torch.zeros(len(outside), dtype=torch.long)))
            if not loss.isfinite():
                raise ValueError("Non-finite compact adaptation loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0, error_if_nonfinite=True)
            optimizer.step()
    return model.eval()


@torch.inference_mode()
def scores(model, frames, patterns=()):
    if frames is None:
        return None
    phones, intent, negation = model_output(model, frames)
    result = {"d6_probability": float(intent.softmax(dim=-1)[0, D6]), "veto": bool(negation.item() >= 0)}
    if patterns:
        logits = phones[0].numpy()
        cost, interval = min((local_ctc_alignment(logits, path, track_span=True)
                              for path in patterns), key=lambda pair: pair[0])
        if interval is not None:
            before, after = exterior_phone_counts(logits.argmax(axis=1), *interval)
            result["ctc_cost"] = cost
            result["outside"] = before + after
    return result


def accepted(row, ceiling):
    return bool(row and ceiling is not None and not row["veto"] and row["d6_probability"] > ceiling)


def calibrate(rows):
    controls = [data["d6_probability"] for expected, _, data in rows if not expected and data and not data["veto"]]
    if not controls:
        raise ValueError("Expected a development control without the negation veto")
    return max(controls)


def development_rows(model, speech, labelled, negatives, patterns=()):
    rows = [(row["intent"] == 6 and not row["denied"], "synthetic", scores(model, frames, patterns))
            for row, frames, *_ in speech]
    rows.extend((row["order"], "human", scores(model, frames, patterns)) for row, frames, _ in labelled)
    rows.extend((False, "CHM150", scores(model, frames, patterns)) for frames in negatives)
    return rows


def development(model, speech, labelled, negatives):
    rows = development_rows(model, speech, labelled, negatives)
    ceiling = calibrate(rows)
    counts = Counter((group, expected, accepted(data, ceiling)) for expected, group, data in rows)
    return {"ceiling": ceiling, "counts": [[*key, value] for key, value in sorted(counts.items())]}


def preservation_regressions(before, after):
    regressions = []
    for metric in ("raw_confusion", "guarded_confusion"):
        for label in kw.LABELS[1:]:
            key = f"{label}:{label}"
            prior, current = before[metric].get(key, 0), after[metric].get(key, 0)
            if current < prior:
                regressions.append({"intent": label, "metric": metric, "before": prior, "after": current})
    return regressions


def train(args):
    source = synthetic_source(args.cache / "index.json")
    labels, speech = load_cache(args.cache)
    labelled = load_examples(args.labelled, source["frontend_sha256"])
    negatives = load_negatives(args.negatives, args.speaker_split, source["frontend_sha256"])
    torch.manual_seed(SEED)
    model, initial = frozen_model(args.base)
    if initial["weights_sha256"] != STUDENT_SHA256 or initial["teacher_cache_sha256"] != SYNTHETIC_SHA256:
        raise ValueError("Unexpected compact starting point")
    before = development(model, speech["dev"], labelled["dev"], negatives["dev"])
    before_all = measure(model, speech["dev"])
    updated = adapt(model, speech["train"], labelled["train"], negatives["train"])
    arrays, _ = quantize(updated)
    args.output.mkdir(mode=0o700)
    bank = args.output / "student-int8-weights.npz"
    np.savez_compressed(bank, **arrays)
    measured = load_int8_weights(bank, len(labels))
    after_all = measure(measured, speech["dev"])
    regressions = preservation_regressions(before_all, after_all)
    report = {"schema": RECIPE, "procedure": PROCEDURE, "teacher_cache_sha256": SYNTHETIC_SHA256,
              "labelled_cache_sha256": file_digest(args.labelled / "index.json"),
              "negative_cache_sha256": file_digest(args.negatives / "index.json"),
              "initial_weights_sha256": STUDENT_SHA256,
              "weights_sha256": file_digest(bank), "labels": len(labels), "bytes": sum(a.nbytes for a in arrays.values()),
              "seed": SEED, "epochs": EPOCHS, "train_counts": [len(speech["train"]), len(labelled["train"]), len(negatives["train"])],
              "initial_calibration": before,
              "calibration": development(measured, speech["dev"], labelled["dev"], negatives["dev"]),
              "initial_all_intents_synthetic_development": before_all,
              "all_intents_synthetic_development": after_all,
              "application_decision": {"eligible": False, "preservation_regressions": regressions,
                                       "human_end_to_end_validation_pending": True}}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def context_accepted(row, limits):
    return bool(limits and accepted(row, limits["probability"]) and "ctc_cost" in row
                and row["ctc_cost"] <= limits["ctc_cost"] and row["outside"] <= limits["outside"])


def select_context_limits(rows):
    orders = [(group, row) for expected, group, row in rows
              if expected and accepted(row, 0.5) and "ctc_cost" in row]
    controls = [row for expected, _, row in rows
                if not expected and row and not row["veto"] and "ctc_cost" in row]
    proposals = []
    for cost in sorted({row["ctc_cost"] for _, row in orders}):
        for outside in sorted({row["outside"] for _, row in orders}):
            blockers = [row["d6_probability"] for row in controls
                        if row["ctc_cost"] <= cost and row["outside"] <= outside]
            ceiling = max([0.5, *blockers])
            limits = {"probability": ceiling, "ctc_cost": cost, "outside": outside}
            activated = [group for group, row in orders if context_accepted(row, limits)]
            if activated:
                rank = (activated.count("human"), len(activated), -outside, -cost, ceiling)
                proposals.append((rank, limits))
    return max(proposals, key=lambda pair: pair[0])[1] if proposals else None


def calibrate_context(args, scorer):
    source = synthetic_source(args.cache / "index.json")
    frontend = source["frontend_sha256"]
    labels, speech = load_cache(args.cache)
    labelled = load_examples(args.labelled, frontend)
    negatives = load_negatives(args.negatives, args.speaker_split, frontend)
    symbols = {symbol: index for index, symbol in enumerate(labels)}
    patterns = [[symbols[phone] for phone in phonemes(str(args.espeak), phrase + ".")]
                for phrase in PHRASES["order"]]
    rows = development_rows(scorer.model, speech["dev"], labelled["dev"], negatives["dev"], patterns)
    limits = select_context_limits(rows)
    groups = Counter((group, expected, context_accepted(row, limits)) for expected, group, row in rows)
    report = {"schema": CONTEXT_SCHEMA, "bank_sha256": scorer.bank_sha256,
              "espeak_sha256": file_digest(args.espeak), "patterns": patterns, "limits": limits,
              "calibration_counts": [[*key, value] for key, value in sorted(groups.items())]}
    with args.output.open("x") as output:
        output.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return {**report, "context_sha256": file_digest(args.output)}


class ReplayScorer:
    def __init__(self, args):
        self.model, report = frozen_model(args.bank)
        source = synthetic_source(args.cache / "index.json")
        frontend = source["frontend_sha256"]
        if (report.get("procedure") != PROCEDURE or report["teacher_cache_sha256"] != SYNTHETIC_SHA256
                or report["labelled_cache_sha256"] != file_digest(args.labelled / "index.json")
                or report["negative_cache_sha256"] != file_digest(args.negatives / "index.json")
                or frontend != file_digest(args.features)):
            raise ValueError("Unexpected replay bank, sources or frontend")
        self.features = args.features
        self.bank_sha256 = report["weights_sha256"]
        self.ceiling = report["calibration"]["ceiling"]
        if not 0 <= self.ceiling <= 1:
            raise ValueError("Invalid frozen D6 probability limit")
        labelled = json.loads((args.labelled / "index.json").read_text())["records"].values()
        negatives = json.loads((args.negatives / "index.json").read_text())["records"].values()
        self.synthetic_pcm = {row["pcm_sha256"] for row in source["records"].values()}
        self.native_pcm = {pcm for row in labelled for pcm in row["pcm_sha256"]}
        self.native_pcm.update(row["effective_pcm_sha256"] for row in negatives)
        self.patterns = ()
        self.context = None
        self.context_sha256 = None
        if getattr(args, "context", None):
            self.context = json.loads(args.context.read_text())
            if self.context["schema"] != CONTEXT_SCHEMA or self.context["bank_sha256"] != self.bank_sha256:
                raise ValueError("Context guard differs from compact bank")
            self.patterns = self.context["patterns"]
            if (not self.patterns or any(not path or any(type(token) is not int or not 0 < token < report["labels"] - 1
                                                       for token in path) for path in self.patterns)):
                raise ValueError("Invalid compact CTC pattern")
            limits = self.context["limits"]
            if limits is not None and (not 0.5 <= limits["probability"] <= 1
                    or not isfinite(limits["ctc_cost"]) or limits["ctc_cost"] < 0
                    or type(limits["outside"]) is not int or limits["outside"] < 0):
                raise ValueError("Invalid compact context limits")
            self.context_sha256 = file_digest(args.context)

    def run(self, pcm):
        return self.frame_scores(native_frames(self.features, strip_digital_edges(pcm))) if eligible_pcm(pcm) else None

    def frame_scores(self, frames):
        return scores(self.model, frames, self.patterns)

    def permit(self, row):
        return context_accepted(row, self.context["limits"]) if self.context else accepted(row, self.ceiling)

    def seal(self):
        limits = self.context["limits"] if self.context else {"probability": self.ceiling}
        return {"bank_sha256": self.bank_sha256, "context_sha256": self.context_sha256, "limits": limits}


def replay(args, scorer):
    def predict(model, features, aligned):
        row = scores(model, native_frames(features, aligned), scorer.patterns)
        return 6 if scorer.permit(row) else 0

    rows = list(evaluate(args, scorer.model, scorer.synthetic_pcm, scorer.native_pcm,
                         predict=predict, align_zero_edges=True))
    for row in rows:
        if any(row[mode]["matches_adjustment_pcm"] for mode in ("raw", "retained")):
            row["overlap"] = True
    clean = [row for row in rows if not row.get("overlap")]
    groups = {"historical_without_exact_overlap": clean,
              "public_historical_excluding_reserved_d6": [
                  row for row in clean if row["split"].startswith("public") and row["split"] != "public_reserved_d6"
              ],
              "public_reserved_d6_diagnostic": [row for row in clean if row["split"] == "public_reserved_d6"]}
    return {**scorer.seal(), "items": len(rows),
            "excluded_exact_overlap": [row["id"] for row in rows if row.get("overlap")],
            "summary": {key: {mode: summarize(group, mode) for mode in ("raw", "retained")}
                        for key, group in groups.items()},
            "episodes": [row for row in rows if row["id"] in {
                "public-03", "public-37", "public-45", "public-52", "public-81",
                "public-82", "public-88", "public-89", "veto-fact", "mty-042-04-15",
            }]}


def replay_scope_test(args, scorer):
    rows = list(scope_cases("test", scorer))
    if any(not scorer.native_pcm.isdisjoint(row["pcm_sha256"]) for row in rows):
        raise ValueError("Scope test intersects labelled or negative PCM")
    counts = Counter((row["positive"], scorer.permit(row["score"])) for row in rows)
    return {**scorer.seal(), "counts": [[*key, value] for key, value in sorted(counts.items())]}


def replay_chm(args, scorer):
    import duckdb
    from phoneme_chm150_lab import CORPUS_SHA256, SPEAKER_SPLIT_SHA256, decode, eligible_rows

    if file_digest(args.corpus) != CORPUS_SHA256 or file_digest(args.speaker_split) != SPEAKER_SPLIT_SHA256:
        raise ValueError("Unexpected corpus or heldout role assignment")
    assignment = json.loads(args.speaker_split.read_text())["speaker_roles"]
    heldout = {speaker for speaker, role in assignment.items() if role == "heldout"}
    speakers = set()
    count = compact = omitted = 0
    added = []
    with duckdb.connect(config={"autoload_known_extensions": False, "autoinstall_known_extensions": False}) as db:
        for _, speaker, _, blob in eligible_rows(db, args.corpus, heldout):
            pcm = decode(args.ffmpeg, blob)
            if not eligible_pcm(pcm):
                omitted += 1
                continue
            if not scorer.native_pcm.isdisjoint(pcm_seals(pcm)):
                raise ValueError("Reserved CHM150 PCM intersects calibration or training")
            speakers.add(speaker)
            count += 1
            score = scorer.run(pcm)
            if int(invoke(args.reference, pcm)):
                compact += 1
            elif scorer.permit(score):
                added.append({"pcm_sha256": sha256(pcm).hexdigest(), "score": score})
    return {**scorer.seal(), "utterances": count, "speakers": len(speakers),
            "compact_activations": compact, "added_d6": len(added), "added": added, "omitted": omitted,
            "ffmpeg_sha256": file_digest(args.ffmpeg)}


def existing(path):
    return Path(path).resolve(strict=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    prep = stages.add_parser("prepare")
    fit = stages.add_parser("train")
    common = argparse.ArgumentParser(add_help=False)
    for name in ("bank", "cache", "labelled", "negatives", "features"):
        common.add_argument("--" + name, type=existing, required=True)
    historical = stages.add_parser("replay", parents=[common])
    historical.set_defaults(coerll_supplemental=None)
    natural = stages.add_parser("replay-chm", parents=[common])
    scope_test = stages.add_parser("replay-scope-test", parents=[common])
    context = stages.add_parser("calibrate-context", parents=[common])
    for command in (historical, natural, scope_test):
        command.add_argument("--context", type=existing)
    for command, names in ((prep, ("human", "teacher", "features", "synthetic-index")),
                           (fit, ("base", "cache", "labelled", "negatives", "speaker-split")),
                           (historical, ("root", "capture", "reference")),
                           (natural, ("reference", "corpus", "speaker-split", "ffmpeg")),
                           (context, ("speaker-split", "espeak"))):
        for name in names:
            command.add_argument("--" + name, type=existing, required=True)
    for command in (prep, fit, context):
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    stages_without_replay = {"prepare": prepare, "train": train}
    if args.stage in stages_without_replay:
        result = stages_without_replay[args.stage](args)
    else:
        replays = {"replay": replay, "replay-chm": replay_chm, "replay-scope-test": replay_scope_test,
                   "calibrate-context": calibrate_context}
        result = replays[args.stage](args, ReplayScorer(args))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
