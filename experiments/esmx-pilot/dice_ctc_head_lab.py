"""Adapt one phoneme projection with a full-turn D6 contrast on a frozen encoder."""
from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from capture_replay import pcm_from
from compact_d6_training_lab import load_examples
from keyword_lab import VOICE_SPLITS, render
from phoneme_reject_lab import load_negatives
from phoneme_replay_lab import frozen_model, native_frames, strip_digital_edges
from phoneme_student_lab import load_cache
from phoneme_teacher_lab import phonemes
from time_lab import file_digest


PAIRS = {
    "train": (
        ("Tira un dado de seis", "No tires un dado de seis"),
        ("Tírame un dado de seis", "Tírame un dado de ocho"),
        ("Lanza un dado de seis", "Lanza dos dados de seis"),
        ("Avienta un dado de seis", "Si atacas, avienta un dado de seis"),
        ("Tira un dado de seis caras", "El manual dice tira un dado de seis caras"),
        ("Tíreme un dado de seis", "No me tire un dado de seis"),
        ("Por favor tire un dado de seis", "Tire un dado de seis, mejor no"),
        ("Puedes tirar un dado de seis", "Dijeron que puedes tirar un dado de seis"),
    ),
    "dev": (
        ("Oye lanza un dado de seis", "Oye no lances un dado de seis"),
        ("Porfa tírame un dado de seis", "Porfa tírame un dado de veinte"),
        ("Me puedes aventar un dado de seis", "Me puedes aventar dos dados de seis"),
        ("Echa un dado de seis por favor", "En este juego se echa un dado de seis"),
    ),
    "test": (
        ("Tire usted un dado de seis", "No tire usted un dado de seis"),
        ("Aviéntame un dado de seis porfa", "Aviéntame otro dado de ocho porfa"),
        ("Lánceme un dado de seis", "En su lugar lánceme tres dados de seis"),
        ("Podrías tirar un dado de seis", "El mensaje decía podrías tirar un dado de seis"),
        ("Pruébalo y tira un dado de seis", "Tira un dado de seis, no, cancélalo"),
        ("Haz una tirada de un dado de seis", "Así es como se tira un dado de seis"),
    ),
}
EXTRA_DENIALS = (
    "no lances un dado de seis", "no avientes un dado de seis", "no lo tires",
    "mejor no tires un dado de seis", "cancela esa tirada de dado de seis",
    "dado de seis", "dado de seis y otro dado", "tira un dado de diez", "tira un dado de veinte",
    "tira un dado de doce", "dos dados de seis", "el dado de seis es", "tira un dado",
    "vamos a tirar un dado de seis", "a qué hora es", "dime un chiste", "pásame la mantequilla",
)
SEED = 20261128
D6 = 6


def paths(espeak, vocab):
    ids = dict(zip(vocab, range(len(vocab)), strict=True))
    text = ([pair[0] for pair in PAIRS["train"]], [pair[1] for pair in PAIRS["train"]] + list(EXTRA_DENIALS))
    return [[tuple(ids.get(token, len(ids) - 1) for token in phonemes(espeak, phrase + "."))
             for phrase in group] for group in text]


def generate(output):
    output.mkdir(mode=0o700)
    jobs = [(split, index, kind, voice_num, voice, rate, pair[kind])
            for split, pairs in PAIRS.items() for index, pair in enumerate(pairs)
            for kind in range(2) for voice_num, (voice, rate) in enumerate(VOICE_SPLITS[split])]

    def one(job):
        split, index, kind, voice_num, voice, rate, phrase = job
        key = f"{split}-{index:02}-{kind}-{voice_num}"
        wav = render(output, key, voice, phrase + ".", rate)
        return {"id": key, "split": split, "d6": kind == 0, "sha256": file_digest(wav)}
    with ThreadPoolExecutor(max_workers=4) as workers:
        manifest = list(workers.map(one, jobs))
    (output / "synthesis.json").write_text(json.dumps(manifest, indent=2) + "\n")


class Grammar:
    def __init__(self, order, control):
        alternatives = order + control
        if not order or not control or any(not seq or any(type(token) is not int or not 0 < token < 32
                                                         for token in seq) for seq in alternatives):
            raise ValueError("Invalid CTC grammar")
        self.lengths = torch.tensor([len(path) for path in alternatives])
        self.targets = torch.tensor([token for path in alternatives for token in path])
        self.order_count = len(order)
        self.minimum_frames = torch.tensor([
            len(path) + sum(left == right for left, right in zip(path, path[1:]))
            for path in alternatives
        ])

    def costs(self, logits, frames):
        size = logits.shape[0]
        losses = F.ctc_loss(
            logits.log_softmax(-1).transpose(0, 1).repeat_interleave(len(self.lengths), dim=1),
            self.targets.repeat(size),
            frames.repeat_interleave(len(self.lengths)),
            self.lengths.repeat(size),
            reduction="none",
            zero_infinity=True,
        ).reshape(size, -1) / frames[:, None]
        viable = self.minimum_frames[None, :] <= frames[:, None]
        order = losses[:, :self.order_count].masked_fill(~viable[:, :self.order_count], float("inf")).amin(-1)
        opposing = losses[:, self.order_count:].masked_fill(~viable[:, self.order_count:], float("inf")).amin(-1)
        return order, opposing


@torch.no_grad()
def encode(model, frames):
    if frames is None or not 32 <= len(frames) <= 199:
        return None
    signal = torch.from_numpy(np.copy(frames.T)).float()[None] / 32
    summary = []
    handle = model.phones.register_forward_pre_hook(lambda _, inputs: summary.append(inputs[0]))
    try:
        output, _, negation = model(signal, torch.tensor([len(frames)]))
    finally:
        handle.remove()
    return summary[0][0].T, output[0], float(negation[0])


def pack(rows):
    data = [(yes, features, original) for yes, _, (features, original, _) in rows]
    lengths = torch.tensor([len(value) for _, value, _ in data])
    target = torch.tensor([yes for yes, _, _ in data], dtype=torch.float)
    signal = nn.utils.rnn.pad_sequence([value for _, value, _ in data], batch_first=True)
    teacher = nn.utils.rnn.pad_sequence([original for _, _, original in data], batch_first=True)
    return signal, lengths, target, teacher


def head_like(model):
    head = nn.Linear(model.phones.in_channels, model.phones.out_channels)
    with torch.no_grad():
        head.weight.copy_(model.phones.weight[..., 0])
        head.bias.copy_(model.phones.bias)
    return head


def adapt(head, grammar, examples, epochs, strength, human_weight):
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.0015)
    for epoch in range(epochs):
        order = torch.randperm(len(examples)).tolist()
        for start in range(0, len(examples), 12):
            batch = [examples[i] for i in order[start:start + 12]]
            signal, lengths, target, teacher = pack(batch)
            example_weight = torch.tensor([human_weight if group == "human" else 1
                                           for _, group, _ in batch], dtype=torch.float32)
            logits = head(signal)
            order_cost, control_cost = grammar.costs(logits, lengths)
            score = 6 * (control_cost - order_cost) - 2 * order_cost
            compare = F.binary_cross_entropy_with_logits(score, target, reduction="none")
            weights = (1 + 2 * target) * example_weight
            compare = (compare * weights).sum() / weights.sum()
            active = torch.arange(logits.shape[1])[None] < lengths[:, None]
            fidelity = F.kl_div(logits[active].log_softmax(-1), teacher[active].softmax(-1),
                                reduction="batchmean")
            loss = compare + strength * fidelity
            if not loss.isfinite():
                raise ValueError(f"Invalid contrast loss at epoch {epoch}")
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.5, error_if_nonfinite=True)
            optimizer.step()
    return head.eval()


@torch.no_grad()
def metrics(head, grammar, encoded):
    if encoded is None:
        return None
    features, _, veto = encoded
    order, other = grammar.costs(head(features[None]), torch.tensor([len(features)]))
    cost, margin = float(order[0]), float(other[0] - order[0])
    if not np.isfinite(cost) or not np.isfinite(margin):
        return None
    return {"order_cost": cost, "margin": margin, "negative_logit": veto,
            "score": 6 * margin - 2 * cost}


def calibrate(head, grammar, rows):
    observations = [(target, group, metrics(head, grammar, encoded)) for target, group, encoded in rows]
    negatives = [item["score"] for target, _, item in observations
                 if not target and item is not None and item["negative_logit"] < 0]
    if not negatives:
        raise ValueError("Missing calibration controls")
    return max(negatives)


def accepted(result, cutoff):
    return bool(result is not None and result["negative_logit"] < 0 and result["score"] > cutoff)


def quantize(head, artifact):
    with torch.no_grad():
        weights = head.weight.detach()
        scales = weights.abs().amax(dim=1).clamp_min(1e-9) / 127
        integers = (weights / scales[:, None]).round().clamp(-127, 127).to(torch.int8)
        np.savez(artifact, weights=integers.numpy(), scales=scales.numpy(), bias=head.bias.detach().numpy())
        head.weight.copy_(integers.float() * scales[:, None])
    return head


def matrix(head, grammar, rows, cutoff):
    return dict(Counter(f"{section}:{int(target)}:{int(accepted(metrics(head, grammar, inp), cutoff))}"
                        for target, section, inp in rows))


def human_rejections(head, grammar, rows, cutoff):
    reasons = Counter()
    for affirmative, group, data in rows:
        if not affirmative or group != "human":
            continue
        item = metrics(head, grammar, data)
        if item is None:
            reasons["invalid"] += 1
        elif item["negative_logit"] >= 0:
            reasons["existing_negation_head"] += 1
        elif item["score"] <= cutoff:
            reasons["below_calibration_threshold"] += 1
        else:
            reasons["accepted"] += 1
    return reasons


def load_data(args, model, expected):
    frontend = file_digest(args.frontend)
    vocab, speech = load_cache(args.speech)
    references = {key: file_digest(path) for key, path in {
        "speech": args.speech / "index.json", "labelled": args.labelled / "index.json",
        "chm": args.chm / "native-negative-features/index.json", "speaker_split": args.chm / "speaker-roles.json",
        "engine": args.frontend, "espeak": args.espeak, "synthesis": args.output / "synthesis.json",
    }.items()}
    if (references["speech"] != expected["teacher_cache_sha256"] or references["chm"] != expected["negative_cache_sha256"]
            or json.loads((args.speech / "index.json").read_text())["frontend_sha256"] != frontend):
        raise ValueError("Calibration inputs do not match the encoder")
    labelled = load_examples(args.labelled, frontend)
    negative = load_negatives(args.chm / "native-negative-features", args.chm / "speaker-roles.json", frontend)
    combined = {split: [(row["intent"] == D6 and not row["denied"], "prior_synthetic", encode(model, frames))
                        for row, frames, *_ in speech[split]] +
                       [(row["order"], row["group"], encode(model, frames)) for row, frames, _ in labelled[split]] +
                       [(False, "conversation", encode(model, frames)) for frames in negative[split]]
                for split in ("train", "dev")}
    combined["test"] = []
    manifest = json.loads((args.output / "synthesis.json").read_text())
    occupied = set()
    expected_rows = {
        f"{split}-{index:02}-{kind}-{voice_num}": (split, kind == 0)
        for split, pairs in PAIRS.items() for index in range(len(pairs))
        for kind in range(2) for voice_num in range(len(VOICE_SPLITS[split]))
    }
    if len(manifest) != len(expected_rows) or {
        row["id"]: (row["split"], row["d6"]) for row in manifest
    } != expected_rows:
        raise ValueError("Synthetic inputs differ from the challenge")
    for row in manifest:
        pcm = strip_digital_edges(pcm_from(args.output / f"{row['id']}.wav", row["sha256"]))
        digest = sha256(pcm).hexdigest()
        if digest in occupied:
            raise ValueError("Synthetic waveform repeated")
        occupied.add(digest)
        combined[row["split"]].append((row["d6"], "new_synthetic", encode(model, native_frames(args.frontend, pcm))))
    return paths(args.espeak, vocab), combined, references


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("generate", "fit"))
    parser.add_argument("output", type=Path)
    fit_inputs = ("bank", "speech", "labelled", "chm", "frontend", "espeak")
    for key in fit_inputs:
        parser.add_argument("--" + key, type=Path)
    args = parser.parse_args()
    if args.stage == "generate":
        generate(args.output)
        return
    if any(getattr(args, key) is None for key in fit_inputs):
        parser.error("fit requires " + ", ".join("--" + key for key in fit_inputs))
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    model, report = frozen_model(args.bank)
    templates, rows, pins = load_data(args, model, report)
    grammar = Grammar(*templates)
    viable = [row for row in rows["train"] if row[2] is not None]
    proposals = {"frozen": head_like(model)}
    for name, penalty, human_weight in (("regularized", 0.50, 1), ("contrast", 0.10, 1),
                                          ("human_8", 0.10, 8), ("human_24", 0.10, 24)):
        torch.manual_seed(SEED)
        proposals[name] = adapt(head_like(model), grammar, viable, 19, penalty, human_weight)
    passes = {}
    for name, head in proposals.items():
        artifact = args.output / f"{name}.npz"
        if name != "frozen":
            quantize(head, artifact)
        cutoff = calibrate(head, grammar, rows["dev"])
        passes[name] = {
            "threshold": cutoff, "matrix": {split: matrix(head, grammar, values, cutoff)
                                            for split, values in rows.items()},
            "human_affirmatives": {split: human_rejections(head, grammar, rows[split], cutoff)
                                   for split in ("train", "dev")},
        }
        if name != "frozen":
            passes[name]["head_sha256"] = file_digest(artifact)
    result = {"bank_weights": report["weights_sha256"], "source_sha256": file_digest(Path(__file__)),
              "paths": templates, "pins": pins, "rules": passes}
    with (args.output / "freeze.json").open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps({"rules": passes, "audio_rows": {key:len(value) for key,value in rows.items()}}, indent=2))


if __name__ == "__main__":
    main()
