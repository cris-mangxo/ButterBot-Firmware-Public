"""Fit a small temporal CNN and export an integer-only C++ bank."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import re
import struct
import subprocess

import numpy as np
import torch
from torch import nn

from ald_round import ALD_SHA256, canonical, digest
from phrases import too_long
from pilot import feature, read_pcm
from model_sources import SHARVARD_SHA256

FIXTURE = Path(__file__).with_name("fixture.json")
BANK_BYTES = 6468
LABELS = {"outside": 0, "positive": 1, "other_die": 2, "negated": 3}
DILATIONS = (1, 2, 3)


class TemporalDice(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv1d(20, 24, 5, padding=2),
            nn.Conv1d(24, 24, 3, padding=2, dilation=2),
            nn.Conv1d(24, 24, 3, padding=3, dilation=3),
        ])
        self.dropout = nn.Dropout(0.10)
        self.final = nn.Linear(72, 4)

    def forward(self, signal):
        for layer in self.layers:
            signal = layer(signal).relu()
        return self.final(self.dropout(pool(signal)))


def pool(signal):
    return signal.reshape(-1, 24, 3, 8).max(axis=3)[0].transpose(1, 2).reshape(-1, 72)


def read_manifest(folder, sharvard=False):
    manifest = json.loads((folder / "manifest.json").read_text("utf-8"))
    field, voice_hash = ("sharvard_sha256", SHARVARD_SHA256) if sharvard else ("ald_sha256", ALD_SHA256)
    if manifest["fixture_sha256"] != digest(FIXTURE) or manifest.get(field) != voice_hash:
        raise ValueError("Corpus de síntesis fuera de la referencia fijada")
    rows = manifest["samples"]
    ids = set()
    texts = {}
    for row in rows:
        if (row["split"] not in {"train", "dev", "test"}
                or not re.fullmatch(fr'{row["split"]}-[0-9]{{4}}-[0-9]', row["id"])
                or row["id"] in ids or LABELS.get(row["category"]) != row["label"]):
            raise ValueError("Manifiesto ambiguo o mal formado")
        ids.add(row["id"])
        text = canonical(row["text"])
        previous = texts.setdefault(text, row["split"])
        if previous != row["split"]:
            raise ValueError("Texto compartido entre particiones")
    return rows


def collect(corpus, rows, program):
    wavs = [(row, corpus / "audio" / (row["id"] + ".wav"))
            for row in rows if row["split"] in {"train", "dev"}]
    observed = {"train": [], "dev": []}
    omitted = Counter()

    def one(item):
        _, wav = item
        if too_long(wav):
            return None
        return np.frombuffer(feature(program, wav), dtype=np.int8).reshape(24, 20).T

    with ThreadPoolExecutor(max_workers=6) as workers:
        for (row, wav), vector in zip(wavs, workers.map(one, wavs), strict=True):
            if vector is None:
                omitted[f'{row["split"]}/{row["category"]}'] += 1
            else:
                observed[row["split"]].append((vector, row["label"], wav))
    return observed, dict(omitted)


def choose_limits(scores, labels, unit, priority_tail=0):
    margin = scores[:, 1] - scores[:, [0, 2, 3]].max(axis=1)
    min_score = scores[:, 1]
    best = None
    for distance in (0.25, 0.5, 0.9, 1.4, 2.0, 2.7, 3.5, 4.5):
        gap = max(1, round(distance * unit))
        near = min_score[(labels != 1) & (margin >= gap)]
        threshold = near.max() + max(1, round(0.12 * unit)) if near.size else min_score.min() - 1
        admit = (margin >= gap) & (min_score >= threshold)
        positive = int((admit & (labels == 1)).sum())
        false = int((admit & (labels != 1)).sum())
        priority_hits = int((admit[-priority_tail:] & (labels[-priority_tail:] == 1)).sum()) if priority_tail else 0
        candidate = (priority_hits, positive, -false, gap, int(threshold))
        if best is None or candidate > best:
            best = candidate
    return best


def fit(train, dev, labels, development, priority_tail=0):
    model = TemporalDice()
    optim = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.015)
    weights = torch.bincount(labels, minlength=4).float().pow(-0.65)
    if not torch.isfinite(weights).all():
        raise ValueError("Hay clases de entrenamiento sin ejemplos")
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=72, eta_min=0.0003)
    best = None
    for epoch in range(1, 73):
        model.train()
        for batch in torch.randperm(len(train)).split(64):
            signal = train[batch]
            shift = int(torch.randint(-1, 2, ()).item())
            if shift > 0:
                signal = torch.roll(signal, shift, dims=2)
                signal[:, :, :shift] = 0
            elif shift < 0:
                signal = torch.roll(signal, shift, dims=2)
                signal[:, :, shift:] = 0
            signal = signal + torch.randn_like(signal) * 0.027
            optim.zero_grad()
            loss = loss_fn(model(signal), labels[batch])
            loss.backward()
            optim.step()
        schedule.step()
        if epoch % 3 == 0:
            model.eval()
            with torch.no_grad():
                scores = model(dev).numpy()
            result = choose_limits(scores, development, 1, priority_tail)
            accuracy = int((scores.argmax(axis=1) == development).sum())
            rank = (*result[:2], accuracy)
            if best is None or rank > best[0]:
                best = (rank, epoch, {key: value.detach().clone()
                                       for key, value in model.state_dict().items()})
    model.load_state_dict(best[2])
    return model.eval(), best[1]


def exponent(bound, maximum):
    return math.floor(math.log2(bound / maximum))


def quantize(model, train):
    layers = []
    current = train
    previous = 5  # Existing signed log-mel values were divided by 32 for training.
    with torch.no_grad():
        for layer in model.layers:
            weight = layer.weight.numpy()
            bias = layer.bias.numpy()
            weighted = exponent(127, np.max(np.abs(weight)))
            current = layer(current).relu()
            activation = exponent(122, max(float(current.quantile(0.999)), 1e-5))
            activation = min(activation, previous + weighted - 1)
            shift = previous + weighted - activation
            if not 0 < shift <= 24:
                raise ValueError("Escala de activación fuera del rango del ejecutor")
            layers.append((np.rint(weight * 2.0 ** weighted).astype(np.int8),
                           np.rint(bias * 2.0 ** (previous + weighted)).astype("<i4"), shift))
            previous = activation
        weight = model.final.weight.numpy()
        weighted = exponent(127, np.max(np.abs(weight)))
        unit = 2.0 ** (previous + weighted)
        output = np.rint(weight * 2.0 ** weighted).astype(np.int8)
        bias = np.rint(model.final.bias.numpy() * unit).astype("<i4")
    return layers, output, bias, unit


def integer_scores(samples, layers, output, bias):
    signal = samples.astype(np.int32)
    for (weight, offset, shift), dilation in zip(layers, DILATIONS, strict=True):
        width = weight.shape[2]
        padding = (width // 2) * dilation
        padded = np.pad(signal, ((0, 0), (0, 0), (padding, padding)))
        index = np.arange(24)[:, None] + dilation * np.arange(width)
        signal = np.einsum("nctk,ock->not", padded[:, :, index], weight.astype(np.int32))
        signal += offset[None, :, None]
        signal = np.clip((signal.clip(0) + (1 << (shift - 1))) >> shift, 0, 127)
    features = signal.reshape(-1, 24, 3, 8).max(axis=3).transpose(0, 2, 1).reshape(-1, 72)
    return features @ output.astype(np.int32).T + bias


def wire_bank(layers, output, bias, threshold, gap):
    prefix = b"".join(weight.tobytes() + offset.tobytes() + struct.pack("<I", shift)
                      for weight, offset, shift in layers)
    bank = prefix + output.tobytes() + bias.tobytes() + struct.pack("<ii", threshold, gap)
    if len(bank) != BANK_BYTES:
        raise ValueError("Tamaño binario distinto del contrato C++")
    return bank


def verify_wire(program, bank, observations, exact):
    chosen = np.linspace(0, len(observations) - 1, 24, dtype=int)
    for index in chosen:
        run = subprocess.run([program, "acoustic-scores", bank],
                             input=read_pcm(observations[index][2]), capture_output=True,
                             check=True, timeout=10)
        if json.loads(run.stdout) != exact[index].tolist():
            raise ValueError("Difieren los logits enteros de Python y C++")
    return len(chosen)


def train_bank(args):
    torch.set_num_threads(4)
    torch.manual_seed(1118)
    observed, omitted = collect(args.corpus, read_manifest(args.corpus), args.program)
    human_dev = []
    dev_rows = []
    provenance = {}
    if args.continuous:
        from whole_orders import pooled_manifests, read_manifest as continuous_manifest
        if len(args.continuous) == 1:
            folder, = args.continuous
            rows, provenance = continuous_manifest(folder)
            groups = [(folder, rows)]
        else:
            groups, provenance = pooled_manifests(args.continuous)
        # Oversample the scarce human turns, with noise/time masking applied by fit().
        human_train = []
        for folder, rows in groups:
            human, skipped = collect(folder, rows, args.program)
            if skipped:
                raise ValueError("Una toma humana quedó fuera del rango de entrada")
            human_train.extend(human["train"])
            human_dev.extend(human["dev"])
            dev_rows.extend(row for row in rows if row["split"] == "dev")
        observed["train"].extend(human_train * 8)
        observed["dev"].extend(human_dev)
    sets = {}
    for split, values in observed.items():
        signal, label, _ = zip(*values)
        sets[split] = np.stack(signal), np.array(label)
    pcm, truth = sets["train"]
    dev, dev_truth = sets["dev"]
    dataset = torch.from_numpy(pcm).float() / 32
    priority = len(human_dev) if args.prioritize_human else 0
    net, epoch = fit(dataset, torch.from_numpy(dev).float() / 32, torch.from_numpy(truth), dev_truth, priority)
    layers, output, bias, unit = quantize(net, dataset)
    exact = integer_scores(dev, layers, output, bias)
    _, positives, negative_false_count, gap, threshold = choose_limits(exact, dev_truth, unit, priority)
    with args.bank.open("xb") as handle:
        handle.write(wire_bank(layers, output, bias, threshold, gap))
    parity = verify_wire(args.program, args.bank, observed["dev"], exact)
    if provenance:
        matched = exact[-len(human_dev):]
        decisions = (matched[:, 1] >= threshold) & (matched[:, 1] - matched[:, [0, 2, 3]].max(axis=1) >= gap)
        human_truth = dev_truth[-len(human_dev):]
        provenance.update({
            "train_repeat_per_tempo": 8, "dev_turns": len(human_dev),
            **({"priority": "human_development"} if priority else {}),
            "dev_positive_detected": int((decisions & (human_truth == 1)).sum()),
            "dev_false_activations": int((decisions & (human_truth != 1)).sum()),
        })
        if "positive_by_die_basis" in provenance:
            by_basis = {}
            for row, admitted in zip(dev_rows, decisions, strict=True):
                if row["label"] != 1:
                    continue
                counts = by_basis.setdefault(row.get("die_basis", "unannotated"), {"eligible": 0, "detected": 0})
                counts["eligible"] += 1
                counts["detected"] += int(admitted)
            provenance["dev_positive_results_by_die_basis"] = by_basis
    class_names = tuple(LABELS)
    summary = {split: dict(Counter(class_names[int(label)] for label in values[1]))
               for split, values in sets.items()}
    print(json.dumps({"fixture_sha256": digest(FIXTURE), "ald_sha256": ALD_SHA256,
                      "bank_sha256": digest(args.bank), "bank_bytes": BANK_BYTES,
                      "chosen_epoch": epoch, "calibration": summary, "omissions_by_length": omitted,
                      "dev_positives": int((dev_truth == 1).sum()),
                      "dev_d6_detected": positives, "dev_false_activations": -negative_false_count,
                      "minimum_score": threshold, "minimum_margin": gap,
                      "parity_cpp_cases": parity, "torch_version": torch.__version__,
                      **({"continuous": provenance} if provenance else {})}, indent=2))


def emit_header(bank, header):
    data = bank.read_bytes()
    if len(data) != BANK_BYTES:
        raise ValueError("Modelo binario inválido")
    with header.open("x", encoding="utf-8") as file:
        file.write('#pragma once\n#include "AcousticDice.h"\n#include <bit>\n')
        file.write(f'inline constexpr auto PilotAcousticDice = '
                   f'std::bit_cast<EsMx::Kws::AcousticDice>(std::array<uint8_t, {BANK_BYTES}>{{\n')
        for start in range(0, len(data), 20):
            file.write("    " + ", ".join(map(str, data[start:start + 20])) + ",\n")
        file.write("});\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("fit", "header"))
    for name in ("program", "corpus", "bank", "header"):
        parser.add_argument(f"--{name}", type=Path, required=name == "bank")
    parser.add_argument("--continuous", type=Path, action="append",
                        help="Use a verified continuous speech corpus. Repeat to combine corpora.")
    parser.add_argument("--prioritize-human", action="store_true")
    args = parser.parse_args()
    if args.prioritize_human and (args.operation != "fit" or not args.continuous):
        parser.error("--prioritize-human requiere fit y un corpus humano continuo")
    if args.operation == "fit":
        if args.program is None or args.corpus is None:
            parser.error("Faltan --program y --corpus")
        train_bank(args)
    else:
        if args.header is None:
            parser.error("Falta --header")
        emit_header(args.bank, args.header)


if __name__ == "__main__":
    main()
