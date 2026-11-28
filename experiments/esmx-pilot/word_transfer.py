"""Pretrain an acoustic keyword encoder on aligned human words, then verify whole turns locally."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

import keyword_lab as kw
from capture_replay import pcm_from, recordings
from time_lab import digest, file_digest, invoke, supplemental

WORD_SOURCE = Path(__file__).resolve().parents[1] / "esmx-compactas" / "autonomous"
sys.path.insert(0, str(WORD_SOURCE))
import human_words
import word_train

BANK_KIND = "BB-WORD-TRANSFER-LAB-1"
WEIGHT_NAME = "word-transfer-host.npz"
WORD_LABELS = tuple(range(len(word_train.WORDS)))
UNUSED = -100
MASK_FLOOR = -100.0
ANAHUAC_ALIGNMENT_SHAS = {
    "79851e2ea743c1312d6a0c58eb4e0751bab618b1de02dfde2240371d0a9529bc",
    "034c733f848559b34b996babf069762019665d6232a57487ca6efbfbf338ce1e",
}
CIEMPIESS_ALIGNMENT_SHA = "4804e4a4cfc24ae54436f477e5c13cf820c8f1b360b22acfe131fff4331970a2"


class Transfer(kw.Keywords):
    def __init__(self):
        super().__init__()
        self.words = nn.Conv1d(kw.CHANNELS, len(WORD_LABELS), 1)

    def word_logits(self, pcm_features, lengths):
        index = torch.arange(kw.MODEL_FRAMES // 2)
        valid = index[None, :] < ((lengths + 1) // 2)[:, None]
        first = torch.relu(self.stem(pcm_features)) * valid[:, None]
        hidden = self.local(first) * valid[:, None]
        return self.words(hidden).masked_fill(~valid[:, None], MASK_FLOOR), valid


def lexical_tensors(samples):
    data = torch.zeros((len(samples), kw.BANDS, kw.MODEL_FRAMES))
    length = torch.zeros(len(samples), dtype=torch.long)
    targets = torch.full((len(samples), kw.MODEL_FRAMES // 2), UNUSED, dtype=torch.long)
    for index, (features, labels, _, _) in enumerate(samples):
        size = min(len(features), kw.MODEL_FRAMES)
        # Use matching indices for long recordings to keep frame supervision aligned.
        positions = np.arange(size) * len(features) // size
        data[index, :, :size] = torch.from_numpy(features[positions].T.copy()).float() / 32
        target = torch.from_numpy(labels[positions[::2]].copy())
        targets[index, :len(target)] = target
        length[index] = size
    return data, length, targets


def masked_noise(features, lengths):
    use = torch.arange(kw.MODEL_FRAMES)[None, None] < lengths[:, None, None]
    return torch.randn_like(features) * 0.038 * use


def word_criterion(targets):
    weights = torch.bincount(targets[targets >= 0], minlength=len(WORD_LABELS)).float()
    if not torch.all(weights):
        raise ValueError("Incomplete set of aligned human word labels")
    return nn.CrossEntropyLoss(weight=weights.pow(-0.45), ignore_index=UNUSED)


def word_accuracy(model, data):
    x, lengths, targets = data
    with torch.inference_mode():
        logits, _ = model.word_logits(x, lengths)
        guessed = logits.argmax(dim=1)
    return {label: [int(((targets == label) & (guessed == label)).sum()),
                    int((targets == label).sum())] for label in WORD_LABELS}


def step(opt, loss):
    opt.zero_grad()
    loss.backward()
    opt.step()


def fit(human, turns):
    torch.manual_seed(kw.SEED)
    model = Transfer()
    xw, nw, yw = human["train"]
    word_ce = word_criterion(yw)
    stem = [*model.stem.parameters(), *model.local.parameters(), *model.words.parameters()]
    pretrain = torch.optim.AdamW(stem, lr=0.003, weight_decay=0.014)
    for _ in range(40):
        for index in torch.randperm(len(xw)).split(64):
            logits, _ = model.word_logits(xw[index] + masked_noise(xw[index], nw[index]), nw[index])
            step(pretrain, word_ce(logits, yw[index]))
    word_dev_after_pretrain = word_accuracy(model, human["dev"])

    _, x, lengths, labels, keywords, denied = kw.batch(turns)
    heads = [*model.context.parameters(), *model.intent.parameters(), *model.keyword.parameters(),
             *model.negation.parameters()]
    joint = torch.optim.AdamW([{"params": stem, "lr": 0.0008},
                               {"params": heads, "lr": 0.003}], weight_decay=0.012)
    counts = torch.bincount(labels, minlength=len(kw.LABELS)).float()
    turn_ce = nn.CrossEntropyLoss(weight=counts.pow(-0.5), label_smoothing=0.04)
    bce = nn.BCEWithLogitsLoss()
    for _ in range(62):
        for index in torch.randperm(len(x)).split(40):
            logits, present, veto = model(x[index] + masked_noise(x[index], lengths[index]), lengths[index])
            turn_loss = (turn_ce(logits, labels[index]) + 0.6 * bce(present, keywords[index])
                         + 0.35 * bce(veto, denied[index]))
            replay = torch.randint(len(xw), (32,))
            word_logits, _ = model.word_logits(xw[replay] + masked_noise(xw[replay], nw[replay]), nw[replay])
            step(joint, turn_loss + 0.5 * word_ce(word_logits, yw[replay]))
    return model.eval(), word_dev_after_pretrain


def ordered_d6(logits, valid):
    # Each lexical margin competes against the other acoustic classes before
    # finding the strongest ordered Verb → Die → Six occurrence.
    def margin(label):
        other = torch.cat([logits[:, :label], logits[:, label + 1:]], dim=1).amax(dim=1)
        return (logits[:, label] - other).masked_fill(~valid, MASK_FLOOR)

    def prior(strength):
        prefix = strength.cummax(dim=1).values
        return torch.cat([torch.full_like(prefix[:, :1], MASK_FLOOR), prefix[:, :-1]], dim=1)

    verb, die, six = [margin(label) for label in (1, 2, 3)]
    with_die = torch.minimum(prior(verb), die)
    ordered = torch.minimum(prior(with_die), six).amax(dim=1)
    return ordered


def scores(model, features, lengths):
    turn = kw.scores(model, features, lengths).copy()
    with torch.inference_mode():
        temporal, valid = model.word_logits(features, lengths)
        ordered = ordered_d6(temporal, valid).numpy()
    d6_index = kw.LABELS.index(kw.INTENTS["d6"]) - 1
    turn[:, d6_index] = np.minimum(turn[:, d6_index], ordered)
    return turn


def load_weight(path):
    model = Transfer()
    with np.load(path, allow_pickle=False) as bank:
        model.load_state_dict({name: torch.from_numpy(bank[name].astype(np.float32) * bank[name + ":scale"])
                               for name in model.state_dict()})
    return model.eval()


def recipe():
    return {**kw.recipe(), **{path.name: file_digest(path) for path in (
        Path(__file__), WORD_SOURCE / "word_train.py", WORD_SOURCE / "human_words.py",
    )}}


def added_words(args):
    # Isolated lexical evidence can support the presence head but must reject
    # a routine: none of these crops is a whole turn.
    selected = []
    if args.anahuac_words:
        if file_digest(args.anahuac_words) not in ANAHUAC_ALIGNMENT_SHAS:
            raise ValueError("Anáhuac development manifest digest mismatch")
        data = json.loads(args.anahuac_words.read_text())
        if data["license"] != "CC0-1.0" or any(
                row["split"] != "development_only" or row["speaker_verified"] for row in data["clips"]):
            raise ValueError("unrecognized Anáhuac partition or provenance")
        selected.extend((args.anahuac_words, row["id"], row["word"], row["file"], row["wav_sha256"])
                        for row in data["clips"])
    if args.ciempiess_words:
        if file_digest(args.ciempiess_words) != CIEMPIESS_ALIGNMENT_SHA:
            raise ValueError("CIEMPIESS development manifest digest mismatch")
        data = json.loads(args.ciempiess_words.read_text())
        # The third crop lacked isolated confirmation and is not used to fit.
        for index in (0, 1):
            row = data["rows"][index]
            if row["locale"] != "es-MX" or row["use"] != "development_lexical_and_context_negative":
                raise ValueError("unrecognized CIEMPIESS partition or provenance")
            isolated = row["lexical_audio"]
            selected.append((args.ciempiess_words, f"ciempiess-lexical-{index}", row["keyword"],
                             isolated["file"], isolated["sha256"]))
    for manifest, identifier, word, name, seal in selected:
        path = (manifest.parent / name).resolve(strict=True)
        if path.parent != manifest.parent:
            raise ValueError("invalid development audio path")
        yield kw.Row(identifier, "train", kw.INTENTS["reject"], word, path, seal)


def freeze(args):
    # read_manifest verifies pinned dataset revision, per-file audio and
    # annotation digests and partitions by original recording and speaker.
    human_rows, provenance = human_words.read_manifest(args.human)
    human, exclusions = word_train.load_rows(args.human, args.features, human_rows)
    lexical = {split: lexical_tensors(samples) for split, samples in human.items()}
    auxiliary = list(added_words(args))
    turn_audio = kw.make_features(args, [*kw.rows(args, ("train", "dev")), *auxiliary])
    turns = {split: [item for item in turn_audio if item[0].split == split] for split in ("train", "dev")}
    human_pcm = {digest(pcm_from(sample[3])): f"mswc-{split}-{sample[3].stem}"
                 for split, samples in human.items() for sample in samples}
    # Verify synthetic and human development used in calibration have no
    # exact same PCM available to either fitting stage.
    training = {digest(pcm_from(sample[3])) for sample in human["train"]}
    development = {digest(pcm_from(sample[3])) for sample in human["dev"]}
    training.update(pcm for _, pcm, _ in turns["train"])
    development.update(pcm for _, pcm, _ in turns["dev"])
    if training & development:
        raise ValueError("Exact PCM overlap between training and development")
    model, lexical_pretrain = fit(lexical, turns["train"])
    artifact = args.output / WEIGHT_NAME
    weight_bytes = kw.quantize(model, artifact)
    deploy_reference = load_weight(artifact)
    refs, features, lengths, labels, _, _ = kw.batch(turns["dev"])
    strengths = scores(deploy_reference, features, lengths)
    limits = {label: max(0.1, float(strengths[labels.numpy() != i, i - 1].max()) + 0.04)
              for i, label in enumerate(kw.LABELS[1:], start=1)}
    measured = [(row.intent, kw.decode(score, limits)) for row, score in zip(refs, strengths)]
    hashes = {**human_pcm, **{pcm: row.key for row, pcm, _ in turn_audio}}
    config = {
        "kind": BANK_KIND, "recipe": recipe(), "environment": kw.environment(),
        "word_source": provenance, "human_train_dev_omissions": exclusions, "seed": kw.SEED,
        "features_sha256": file_digest(args.features), "weight_sha256": file_digest(artifact),
        "weight_bytes_int8_plus_scales": weight_bytes, "host_container_bytes": artifact.stat().st_size,
        "human_lexical_labels": [sorted(tokens) for tokens in word_train.WORDS],
        "external_word_manifest_digests": {
            name: file_digest(manifest)
            for name, manifest in (("anahuac", args.anahuac_words), ("ciempiess", args.ciempiess_words))
            if manifest
        },
        "external_word_inputs_in_train": {row.key: row.phrase for row in auxiliary},
        "human_word_frame_dev_after_pretrain": lexical_pretrain,
        "human_word_frame_dev_after_joint_and_weight_quantization": word_accuracy(deploy_reference, lexical["dev"]),
        "synthetic_turn_development": {
            label: [sum(expected == label == observed for expected, observed in measured),
                    sum(expected == label for expected, _ in measured)] for label in kw.LABELS},
        "turn_train_dev_omissions": [row.key for row, _, vector in turn_audio if vector is None],
        "calibration": limits, "adjustment_pcm_hashes": hashes,
    }
    with (args.output / "frozen.json").open("x") as stream:
        json.dump(config, stream, indent=2)
    print(json.dumps({key: val for key, val in config.items() if key != "adjustment_pcm_hashes"}, indent=2))


def score_one(model, vector):
    if vector is None:
        return None
    x, length = vector
    return scores(model, torch.from_numpy(x[None]), torch.tensor([length]))[0]


def replay(args):
    artifact = args.output / WEIGHT_NAME
    config = json.loads((args.output / "frozen.json").read_text())
    if (config["kind"] != BANK_KIND or config["recipe"] != recipe()
            or config["environment"] != kw.environment()
            or config["features_sha256"] != file_digest(args.features)
            or config["weight_sha256"] != file_digest(artifact)):
        raise ValueError("Frozen weight bank, frontend or recipe changed")
    limits = {int(label): threshold for label, threshold in config["calibration"].items()}
    model = load_weight(artifact)
    panel = list(recordings(args.root, args.reserved_manifest))
    if len(panel) < 112:
        raise ValueError("Missing historic baseline recordings")
    if args.coerll_supplemental:
        panel.append(supplemental(args.root, args.coerll_supplemental))
    overlap = config["adjustment_pcm_hashes"]

    def observe(pcm):
        baseline = int(invoke(args.reference, pcm)) if pcm and len(pcm) <= 128_000 else 0
        trial = kw.decode(score_one(model, kw.encode(args.features, pcm)), limits)
        return {"baseline": baseline, "candidate": trial, "combined": baseline or trial}

    results = []
    for key, split, category, expected, wave, seal in panel:
        pcm = pcm_from(wave, seal)
        hash_pcm = digest(pcm)
        seen = hash_pcm in overlap
        if seen and split not in ("synthetic_panel", "synthetic_aliases"):
            raise ValueError(f"Frozen heldout intersects training or development: {key}")
        results.append({"id": key, "split": split, "category": category, "expected": expected,
                        "seen_pcm": seen, "pcm_sha256": hash_pcm, "raw": observe(pcm),
                        "retained_capture": observe(invoke(args.capture, pcm, "--pcm-with-hangover"))})
    for row, hash_pcm, vector in kw.make_features(args, kw.rows(args, ("test",))):
        if hash_pcm in overlap:
            raise ValueError(f"Frozen synthetic test overlaps fitting: {row.key}")
        results.append({"id": row.key, "split": "synthetic_test", "expected": row.intent,
                        "pcm_sha256": hash_pcm, "negation": bool(row.denied()),
                        "candidate": kw.decode(score_one(model, vector), limits)})

    panels = {
        "historical_112": results[:112],
        "public_time_heldout": [row for row in results if row["split"] == "public_reserved_time"],
        "public_d6_heldout": [row for row in results if row["split"] == "public_reserved_d6"],
        "supplemental_time": [row for row in results if row["split"] == "supplemental_human_time"],
    }
    summary = {}
    for name, cohort in panels.items():
        modes = {}
        for mode in ("raw", "retained_capture"):
            modes[mode] = {
                "new_wrong_dispatch": [row["id"] for row in cohort if row[mode]["baseline"] == 0
                                       and row[mode]["combined"] not in (0, row["expected"])],
                "correct_by_label": {
                    label: [sum(row["expected"] == label == row[mode][arm] for row in cohort)
                            for arm in ("baseline", "candidate", "combined")]
                    for label in kw.LABELS[1:]
                },
            }
        summary[name] = {"counts": dict(Counter(row["expected"] for row in cohort)), **modes}
    synthetic = [row for row in results if row["split"] == "synthetic_test"]
    print(json.dumps({
        "weight_sha256": config["weight_sha256"], "baseline_sha256": file_digest(args.reference),
        "capture_sha256": file_digest(args.capture), "summary": summary,
        "synthetic_test": {label: [sum(row["expected"] == label == row["candidate"] for row in synthetic),
                                   sum(row["expected"] == label for row in synthetic)] for label in kw.LABELS},
        "negated_synthetic_activated": [row["id"] for row in synthetic if row["negation"] and row["candidate"]],
        "rows": results,
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "replay"))
    existing = lambda path: Path(path).resolve(strict=True)
    for key in ("root", "wav", "features", "output"):
        parser.add_argument("--" + key, type=existing, required=True)
    for key in ("human", "reference", "capture", "coerll-supplemental"):
        parser.add_argument("--" + key, type=existing)
    parser.add_argument("--reserved-manifest", type=existing, action="append", default=[])
    parser.add_argument("--anahuac-words", type=existing)
    parser.add_argument("--ciempiess-words", type=existing)
    parser.add_argument("--skip-adjustment", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(3)
    if args.stage == "freeze" and args.human:
        freeze(args)
    elif args.stage == "replay" and args.reference and args.capture:
        replay(args)
    else:
        parser.error("freeze requires --human and replay requires --reference and --capture")


if __name__ == "__main__":
    main()
