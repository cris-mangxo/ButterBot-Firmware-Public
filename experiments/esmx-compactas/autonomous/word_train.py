"""Fit an int8 word detector from Ald's local phoneme alignments."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch import nn

from ald_round import canonical, digest
from phrases import too_long
from pilot import read_pcm
from train import FIXTURE, exponent, read_manifest

WORDS = (
    set(),
    set(("tira tirar tires tiras tirame tirara tire tirada lanza lanzar lanzame lances lanzas "
         "lanzarme lanzara lance avienta aventar avientas avientes avientame aviente avientas "
         "echa echar echame echas eches echale eche rueda rodar ruedas ruedes").split()),
    {"dado"},
    {"seis"},
    {"no", "nunca", "tampoco", "jamas", "ni", "ninguna", "ningun"},
    set("uno dos tres cuatro cinco siete ocho nueve diez doce dieciseis veinte sesenta setenta cien dados".split()),
    set(("dijo dije dijeron cuando si cuando luego despues y ayer es tiene son pregunta dime "
         "mientras quedamos guardar pintar resultado color precio salio pienso sabes tampoco").split()),
)
VOCABULARY = {word: label for label, group in reversed(list(enumerate(WORDS))) for word in group}
WINDOW = 17
HIDDEN = 24
LABELS = len(WORDS)
MAX_FRAMES = 199
BYTES = 9140


def word_ranges(phones):
    result = []
    start = None
    cursor = 0
    for phone, length in phones:
        if not isinstance(length, int) or length < 0:
            raise ValueError("Duración de fonema inválida")
        if phone in {" ", ".", ",", ";", ":", "?", "!", "¿", "¡", "^", "$"}:
            if start is not None:
                result.append((start, cursor))
                start = None
        elif start is None:
            start = cursor
        cursor += length
    if start is not None:
        result.append((start, cursor))
    return result


def prepare(item, program):
    row, wav = item
    if too_long(wav):
        return "length"
    align = json.loads(wav.with_suffix(".json").read_text("utf-8"))
    tokens = canonical(row["text"]).split()
    intervals = word_ranges(align["phones"])
    if len(tokens) != len(intervals):
        return "alignment"
    pcm = read_pcm(wav)
    wire = subprocess.run([program, "word-frames"], input=pcm, capture_output=True,
                          check=True, timeout=10).stdout
    signal = np.frombuffer(wire, dtype=np.int8).reshape(-1, 20)
    if not 0 < len(signal) <= MAX_FRAMES:
        raise ValueError("Número de frames inválido")
    # At 16 kHz the first analysis window's midpoint is 240 samples.
    centers = (np.arange(len(signal)) * 320 + 240) * align["sample_rate"] / 16000
    targets = np.zeros(len(signal), np.int64)
    for token, (start, end) in zip(tokens, intervals, strict=True):
        label = VOCABULARY.get(token, 0)
        if label:
            inset = (end - start) // 6
            targets[(centers >= start + inset) & (centers < end - inset)] = label
    return signal, targets, row["label"], wav


def load(corpus, program):
    rows = read_manifest(corpus)
    pairs = [(row, corpus / "audio" / (row["id"] + ".wav")) for row in rows if row["split"] != "test"]
    groups = {"train": [], "dev": []}
    omitted = Counter()
    with ThreadPoolExecutor(max_workers=6) as workers:
        for (row, _), item in zip(pairs, workers.map(lambda pair: prepare(pair, program), pairs), strict=True):
            if isinstance(item, str):
                omitted[f'{row["split"]}/{item}'] += 1
            else:
                groups[row["split"]].append(item)
    return groups, dict(omitted)


def tensors(group):
    x = torch.zeros((len(group), 20, MAX_FRAMES), dtype=torch.float32)
    target = torch.full((len(group), MAX_FRAMES), -100, dtype=torch.long)
    for index, (signal, mask, _, _) in enumerate(group):
        x[index, :, :len(signal)] = torch.from_numpy(signal.T.copy()).float() / 32
        target[index, :len(mask)] = torch.from_numpy(mask)
    return x, target


class RecognizeWords(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Conv1d(20, HIDDEN, WINDOW, padding=WINDOW // 2)
        self.hidden = nn.Conv1d(HIDDEN, HIDDEN, 1)
        self.output = nn.Conv1d(HIDDEN, LABELS, 1)

    def forward(self, x):
        return self.output(self.hidden(self.input(x).relu()).relu())


def decode(labels):
    stage = 1
    last_token = 0
    last_frame = -1
    selected = np.flatnonzero((labels[1:] == labels[:-1]) & (labels[1:] != 0)) + 1
    for moment in selected:
        if moment >= 2 and labels[moment] == labels[moment - 2]:
            continue
        token = int(labels[moment])
        if token >= 4:
            return False
        if token == last_token and moment - last_frame <= 10:
            last_frame = moment
            continue
        if token != stage or (stage > 1 and moment - last_frame > 50):
            return False
        last_token, last_frame = token, moment
        stage += 1
    return stage == 4


def calibrate(logits, labels, unit):
    scored = []
    for signal in logits:
        best = signal.argmax(axis=1)
        ranked = np.partition(signal, -2, axis=1)
        margin = ranked[:, -1] - ranked[:, -2]
        scored.append((best, margin))
    best = None
    for word in (0.18, 0.35, 0.6, 0.95, 1.3, 1.8, 2.4, 3.2, 4.0, 5.5, 7.0, 9.0):
        for veto in (0.10, 0.25, 0.5, 0.8, 1.2, 1.8, 2.4, 3.2, 4.0):
            if veto > word:
                continue
            word_gap = max(1, round(word * unit)) if unit > 1 else word
            veto_gap = max(1, round(veto * unit)) if unit > 1 else veto
            detected = [decode(np.where(margin >= np.where(tokens >= 4, veto_gap, word_gap), tokens, 0))
                        for tokens, margin in scored]
            recall = sum(wants == 1 and detects for wants, detects in zip(labels, detected, strict=True))
            false = sum(wants != 1 and detects for wants, detects in zip(labels, detected, strict=True))
            rank = (-false, recall, -veto_gap, word_gap)
            if best is None or rank > best[0]:
                best = (rank, word_gap, veto_gap, detected)
    return best


def fit(groups):
    train, y_train = tensors(groups["train"])
    dev_x, _ = tensors(groups["dev"])
    lengths = [len(item[0]) for item in groups["dev"]]
    truth = [item[2] for item in groups["dev"]]
    weights = torch.bincount(y_train[y_train >= 0], minlength=LABELS).float().pow(-0.6)
    if not torch.isfinite(weights).all():
        raise ValueError("Hay etiquetas sin ejemplos")
    model = RecognizeWords()
    opt = torch.optim.AdamW(model.parameters(), lr=0.0025, weight_decay=0.008)
    loss_fn = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
    best = None
    for epoch in range(1, 37):
        model.train()
        for batch in torch.randperm(len(train)).split(48):
            signal = train[batch]
            signal = signal * torch.empty((len(batch), 1, 1)).uniform_(0.9, 1.1)
            signal = signal + torch.randn_like(signal) * 0.023
            opt.zero_grad()
            loss = loss_fn(model(signal), y_train[batch])
            loss.backward()
            opt.step()
        if epoch % 4 != 0:
            continue
        model.eval()
        with torch.no_grad():
            raw = model(dev_x).transpose(1, 2).numpy()
        decoded = calibrate([entry[:len_] for entry, len_ in zip(raw, lengths, strict=True)], truth, 1)
        rank = decoded[0][:2]
        if best is None or rank > best[0]:
            best = (rank, epoch, {key: value.detach().clone() for key, value in model.state_dict().items()})
            print(f"dev epoch={epoch} d6={rank[1]} false={-rank[0]}", file=sys.stderr, flush=True)
    model.load_state_dict(best[2])
    return model.eval(), best[1], train


def quantize(model, train):
    layers = []
    previous = 5
    frame = train[::3]
    with torch.no_grad():
        for layer in (model.input, model.hidden):
            weights = layer.weight.numpy()
            scale = exponent(127, float(np.abs(weights).max()))
            frame = layer(frame).relu()
            activation = exponent(122, max(float(frame.quantile(0.999)), 1e-5))
            shift = previous + scale - min(activation, previous + scale - 1)
            if not 0 < shift <= 24:
                raise ValueError("Desplazamiento inválido")
            scale_input = 2.0 ** (previous + scale)
            layers.append(np.rint(weights * 2.0 ** scale).astype(np.int8).tobytes()
                          + np.rint(layer.bias.numpy() * scale_input).astype("<i4").tobytes()
                          + struct.pack("<I", shift))
            previous = previous + scale - shift
        weights = model.output.weight.numpy()
        out_scale = exponent(127, float(np.abs(weights).max()))
        unit = 2.0 ** (previous + out_scale)
        output = (np.rint(weights * 2.0 ** out_scale).astype(np.int8).tobytes()
                  + np.rint(model.output.bias.numpy() * unit).astype("<i4").tobytes())
    weights = b"".join(layers) + output
    if len(weights) + 8 != BYTES or unit <= 1:
        raise ValueError("Dimensiones cuantizadas inválidas")
    return weights, unit


def score_dev(program, bank, groups, weights, unit):
    def invoke(mode, wav, model):
        return subprocess.run([program, mode, model], input=read_pcm(wav),
                              check=True, capture_output=True, timeout=10).stdout
    development = groups["dev"]
    with TemporaryDirectory(prefix="bb-word-probe-") as scratch:
        probe = Path(scratch) / "probe.bin"
        probe.write_bytes(weights + struct.pack("<ii", 1, 1))
        with ThreadPoolExecutor(max_workers=6) as workers:
            raw = list(workers.map(lambda row: invoke("word-scores", row[3], probe), development))
        logits = [np.frombuffer(wire, dtype="<i4").reshape(-1, LABELS) for wire in raw]
        for score, (signal, _, _, _) in zip(logits, development, strict=True):
            if len(score) != len(signal):
                raise ValueError("Desfase de frames en el ejecutor")
        rank, word_margin, veto_margin, decisions = calibrate(logits, [row[2] for row in development], unit)
    with bank.open("xb") as out:
        out.write(weights + struct.pack("<ii", word_margin, veto_margin))
    with ThreadPoolExecutor(max_workers=6) as workers:
        verdict = list(workers.map(lambda row: int(invoke("word-accept", row[3], bank)), development))
    if verdict != decisions:
        raise ValueError("Desacuerdo entre la gramática de calibración y C++")
    return {"dev_d6_detected": rank[1], "dev_false_activations": -rank[0],
            "word_margin": word_margin, "veto_margin": veto_margin, "parity_cpp_turns": len(decisions)}


def header(bank, target):
    raw = bank.read_bytes()
    if len(raw) != BYTES:
        raise ValueError("Modelo binario inválido")
    with target.open("x", encoding="utf-8") as output:
        output.write('#pragma once\n#include "WordDice.h"\n#include <bit>\n')
        output.write(f'inline constexpr auto PilotWordDice = '
                     f'std::bit_cast<EsMx::Kws::WordBank>(std::array<uint8_t, {BYTES}>{{\n')
        for start in range(0, len(raw), 20):
            output.write("    " + ", ".join(map(str, raw[start:start + 20])) + ",\n")
        output.write("});\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("fit", "header"))
    for name in ("program", "corpus", "bank", "header"):
        parser.add_argument(f"--{name}", type=Path, required=name == "bank")
    args = parser.parse_args()
    if args.operation == "header":
        if args.header is None:
            parser.error("Falta --header")
        header(args.bank, args.header)
        return
    if args.program is None or args.corpus is None:
        parser.error("Faltan --program y --corpus")
    torch.set_num_threads(4)
    torch.manual_seed(1151)
    groups, omitted = load(args.corpus, args.program)
    net, epoch, training = fit(groups)
    weights, unit = quantize(net, training)
    results = score_dev(args.program, args.bank, groups, weights, unit)
    print(json.dumps({"model_bytes": BYTES, "model_sha256": digest(args.bank), "fixture_sha256": digest(FIXTURE),
                      "chosen_epoch": epoch, "torch_version": torch.__version__,
                      "train_turns": len(groups["train"]), "dev_turns": len(groups["dev"]),
                      "dev_d6_total": sum(row[2] == 1 for row in groups["dev"]),
                      "omitted": omitted, **results}, indent=2))


if __name__ == "__main__":
    main()
