"""Distill a host-only compact phoneme model and compare its weight-quantized development score."""

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import keyword_lab as kw
from phoneme_teacher_lab import SUPPORTED_LABELS, collapse, edit_distance
from time_lab import file_digest


WIDTH = 28
RECIPE = "bb-phoneme-student-1"
EPOCHS = 72
BATCH = 24
SEED = 20261127


class PhoneBlock(nn.Module):
    def __init__(self, dilation, width=WIDTH):
        super().__init__()
        self.temporal = nn.Conv1d(width, width, 5, padding=2 * dilation, dilation=dilation, groups=width)
        self.channel = nn.Conv1d(width, width, 1)

    def forward(self, sequence):
        return F.relu(sequence + self.channel(self.temporal(sequence)))


class Student(nn.Module):
    def __init__(self, labels, width=WIDTH):
        super().__init__()
        self.input = nn.Conv1d(kw.BANDS, width, 5, padding=2)
        self.blocks = nn.ModuleList(PhoneBlock(dilation, width) for dilation in (1, 2, 4))
        self.phones = nn.Conv1d(width, labels, 1)
        self.intent = nn.Linear(width * 5, len(kw.LABELS))
        self.negation = nn.Linear(width * 5, 1)

    def forward(self, frames, lengths):
        positions = torch.arange(frames.shape[-1])[None, :]
        mask = positions < lengths[:, None]
        sequence = F.relu(self.input(frames)) * mask[:, None]
        for block in self.blocks:
            sequence = block(sequence) * mask[:, None]
        turns = positions * 4 // lengths[:, None]
        pools = [sequence.sum(dim=-1) / lengths[:, None]]
        pools.extend(
            sequence.masked_fill(~(mask & (turns == part))[:, None], 0).amax(dim=-1) for part in range(4)
        )
        summary = torch.cat(pools, dim=-1)
        return self.phones(sequence).transpose(1, 2), self.intent(summary), self.negation(summary).squeeze(-1)


def load_cache(directory):
    data = json.loads((directory / "index.json").read_text())
    if data["schema"] != "bb-phoneme-development-1" or data["split_origin"] != "synthetic_mac_voices_and_ald":
        raise ValueError("Expected the synthetic teacher development cache")
    labels = data["vocabulary"]
    if len(labels) > SUPPORTED_LABELS or labels[0] != "<pad>" or labels[-1] != "<other>":
        raise ValueError("Invalid compact CTC vocabulary")
    groups = {"train": [], "dev": []}
    hashes = {key: set() for key in groups}
    for key, row in sorted(data["records"].items()):
        if not key.startswith(f"{row['split']}-") or not all(c.isalnum() or c == "-" for c in key):
            raise ValueError("Invalid synthetic utterance identifier")
        path = directory / f"{key}.npz"
        if file_digest(path) != row["artifact_sha256"]:
            raise ValueError("Development target differs from its seal")
        with np.load(path, allow_pickle=False) as artifact:
            frames, teacher, phones = (artifact[name] for name in ("features", "teacher", "phones"))
        n = row["frames"]
        if (frames.shape != (n, kw.BANDS) or frames.dtype != np.int8 or teacher.shape != (n, len(labels))
                or not np.isfinite(teacher).all() or not np.allclose(teacher.sum(axis=-1), 1, atol=2e-3)
                or phones.ndim != 1 or not len(phones) or (phones < 1).any() or (phones >= len(labels)).any()
                or len(phones) + np.count_nonzero(phones[1:] == phones[:-1]) > n):
            raise ValueError(f"Invalid phoneme cache geometry or CTC target for {key}")
        groups[row["split"]].append((row, frames, teacher, phones))
        hashes[row["split"]].add(row["pcm_sha256"])
    if not all(groups.values()) or hashes["train"] & hashes["dev"]:
        raise ValueError("Empty split or duplicate synthetic waveform across splits")
    return labels, groups


def pack(samples):
    lengths = torch.tensor([row["frames"] for row, *_ in samples])
    widths = torch.tensor([len(phones) for _, _, _, phones in samples])
    frames = torch.zeros((len(samples), kw.BANDS, int(lengths.max())))
    targets = torch.zeros((len(samples), int(lengths.max()), samples[0][2].shape[-1]))
    for i, (_, feature, teacher, _) in enumerate(samples):
        frames[i, :, :len(feature)] = torch.from_numpy(feature.T) / 32
        probs = torch.from_numpy(teacher.astype(np.float32))
        targets[i, :len(teacher)] = probs / probs.sum(dim=-1, keepdim=True)
    labels = torch.tensor([kw.LABELS.index(row["intent"]) for row, *_ in samples])
    denied = torch.tensor([row["denied"] for row, *_ in samples], dtype=torch.float32)
    phones = torch.from_numpy(np.concatenate([phone for _, _, _, phone in samples]).astype(np.int64))
    return frames, lengths, widths, targets, labels, denied, phones


def supervised_loss(model, batch, class_weights):
    x, n, widths, teacher, expected, denied, phones = pack(batch)
    logits, intent, negation = model(x, n)
    logprobs = logits.log_softmax(dim=-1)
    valid = torch.arange(x.shape[-1])[None, :] < n[:, None]
    soft = F.kl_div(logprobs[valid], teacher[valid], reduction="batchmean")
    hard = F.ctc_loss(logprobs.transpose(0, 1), phones, n, widths)
    loss = soft + 0.35 * hard + F.cross_entropy(intent, expected, weight=class_weights)
    return loss + 0.5 * F.binary_cross_entropy_with_logits(negation, denied)


def fit(labels, samples):
    torch.manual_seed(SEED)
    model = Student(len(labels))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.001)
    counts = torch.bincount(torch.tensor([kw.LABELS.index(row["intent"]) for row, *_ in samples]))
    if len(counts) != len(kw.LABELS) or (counts == 0).any():
        raise ValueError("Missing intention in training")
    class_weights = counts.float().rsqrt()
    for _ in range(EPOCHS):
        order = torch.randperm(len(samples)).tolist()
        for offset in range(0, len(order), BATCH):
            batch = [samples[i] for i in order[offset:offset + BATCH]]
            loss = supervised_loss(model, batch, class_weights)
            if not loss.isfinite():
                raise ValueError("Invalid distillation loss")
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0, error_if_nonfinite=True)
            optimizer.step()
    return model.eval()


def quantize(model):
    restored = deepcopy(model)
    saved = {}
    state = {}
    for name, values in model.state_dict().items():
        if not name.endswith("weight"):
            saved[name] = values.numpy()
            state[name] = values
            continue
        axes = tuple(range(1, values.ndim))
        scale = values.abs().amax(dim=axes, keepdim=True).clamp_min(1e-8) / 127
        integer = (values / scale).round().clamp(-127, 127).to(torch.int8)
        saved[name] = integer.numpy()
        saved[name + ".scale"] = scale.numpy()
        state[name] = integer.float() * scale
    restored.load_state_dict(state)
    return saved, restored


def gate(activations, denied):
    ranked, choices = activations.softmax(dim=-1).topk(2, dim=-1)
    intents = choices[:, 0].clone()
    intents[(ranked[:, 0] < 0.70) | (ranked[:, 0] - ranked[:, 1] < 0.20) | (denied.sigmoid() >= 0.50)] = 0
    return intents


@torch.inference_mode()
def measure(model, samples):
    report = Counter()
    raw_matrix = Counter()
    guarded_matrix = Counter()
    for offset in range(0, len(samples), BATCH):
        batch = samples[offset:offset + BATCH]
        frames, lengths, _, _, _, _, _ = pack(batch)
        logits, intents, denied = model(frames, lengths)
        for i, ((row, _, _, phones), raw, guarded) in enumerate(
                zip(batch, intents.argmax(dim=-1), gate(intents, denied), strict=True)):
            predicted = collapse(logits[i, :lengths[i]].argmax(dim=-1).tolist())
            report["utterances"] += 1
            report["reference_phones"] += len(phones)
            report["edit_operations"] += edit_distance(predicted, phones)
            raw_matrix[f"{row['intent']}:{kw.LABELS[raw]}"] += 1
            observed = kw.LABELS[guarded]
            guarded_matrix[f"{row['intent']}:{observed}"] += 1
            if row["denied"]:
                report["negated_turns"] += 1
                report["negated_activated"] += bool(observed)
    return dict(report, raw_confusion=raw_matrix, guarded_confusion=guarded_matrix)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=lambda p: Path(p).resolve(strict=True), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    labels, groups = load_cache(args.cache)
    model = fit(labels, groups["train"])
    arrays, restored = quantize(model)
    args.output.mkdir(mode=0o700)
    path = args.output / "student-int8-weights.npz"
    np.savez_compressed(path, **arrays)
    report = {
        "schema": RECIPE, "teacher_cache_sha256": file_digest(args.cache / "index.json"),
        "weights_sha256": file_digest(path), "labels": len(labels), "seed": SEED, "epochs": EPOCHS,
        "train_turns": len(groups["train"]), "weight_scale_bias_bytes": sum(item.nbytes for item in arrays.values()),
        "development_float": measure(model, groups["dev"]),
        "development_weight_int8_inference_float": measure(restored, groups["dev"]),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
