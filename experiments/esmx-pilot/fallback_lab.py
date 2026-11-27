"""Evaluate a three-command int8 fallback in desktop lab only, with fixed provenance."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from torch import nn

EXPERIMENTS = Path(__file__).resolve().parents[1] / "esmx-compactas"
sys.path.extend((str(EXPERIMENTS), str(EXPERIMENTS / "autonomous")))

from ald_round import ALD_SHA256, canonical, digest, render_ald
from pilot import feature, read_pcm, render
from train import TemporalDice, integer_scores, quantize, wire_bank
from replay import evaluate, panel, PUBLIC_PANELS

LABELS = {"reject": 0, "butter": 1, "joke": 3, "time": 5}
LABEL_ORDER = tuple(LABELS.values())
SYNTHETIC_SPLITS = {
    "train": (("Paulina", 176), ("Eddy (Español (México))", 159), ("Sandy (Español (México))", 190)),
    "dev": (("Grandpa (Español (México))", 169), ("Shelley (Español (México))", 185)),
    "test": (("Flo (Español (México))", 163), ("Rocko (Español (México))", 191)),
}


def prepare(args, fixture):
    texts = [canonical(text) for groups in fixture.values() for texts in groups.values() for text in texts]
    if len(set(texts)) != len(texts) or set(fixture) != set(SYNTHETIC_SPLITS):
        raise ValueError("duplicate phrase or unexpected split")
    records = []
    for split, voices in SYNTHETIC_SPLITS.items():
        for category, phrases in fixture[split].items():
            for number, text in enumerate(phrases):
                for i, (voice, rate) in enumerate(voices):
                    key = f"{split}-{category}-{number:02}-{i}"
                    records.append((split, category, key, text, voice, rate))
                if split == "train":
                    key = f"{split}-{category}-{number:02}-ald"
                    records.append((split, category, key, text, "Piper Ald", 0.92 + (number % 5) * 0.04))
    waves = args.output / "wav"
    waves.mkdir()
    def convert(item):
        split, category, key, text, voice, rate = item
        wav = (render_ald(str(args.piper_python), args.ald, waves, key, text, rate)
               if voice == "Piper Ald" else render(waves, key, voice, text, rate))
        try:
            data = vector(args, wav)
        except ValueError:
            return split, category, key, wav, None
        return split, category, key, wav, data

    with ThreadPoolExecutor(max_workers=6) as executor:
        rows = list(executor.map(convert, records))
    return rows


def vector(args, wav):
    return np.frombuffer(feature(args.feature_tool, wav), dtype=np.int8).reshape(24, 20).T


def fit(train_vectors, classes):
    torch.set_num_threads(3)
    torch.manual_seed(20261127)
    model = TemporalDice()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.015)
    labels = torch.tensor(classes)
    loss = nn.CrossEntropyLoss(weight=torch.bincount(labels, minlength=4).float().pow(-0.65))
    for _ in range(54):
        for batch in torch.randperm(len(train_vectors)).split(40):
            base = train_vectors[batch]
            augmented = base + torch.randn_like(base) * 0.045
            optimizer.zero_grad()
            loss(model(augmented), labels[batch]).backward()
            optimizer.step()
    return model.eval()


def calibrate(scores, targets, unit):
    result = {}
    for i, label in enumerate(LABEL_ORDER[1:], start=1):
        margin = scores[:, i] - np.delete(scores, i, axis=1).max(axis=1)
        wrong = targets != label
        min_margin = max(round(0.55 * unit), int(margin[wrong].max()) + max(1, round(0.08 * unit)))
        result[label] = min_margin
    return result


def decide(scores, limits):
    winner = int(np.argmax(scores))
    label = LABEL_ORDER[winner]
    other = max(value for i, value in enumerate(scores) if i != winner)
    return label if label in limits and scores[winner] - other >= limits[label] else 0


def check_rows(args, prepared, limits, bank, layers):
    def observe(key, wav, expect, kind, input_data=None):
        samples = read_pcm(wav)
        cxx = json.loads(subprocess.run([args.feature_tool, "acoustic-scores", bank],
                                       input=samples, capture_output=True, check=True, timeout=10).stdout)
        if input_data is None:
            input_data = vector(args, wav)
        expected = integer_scores(input_data[None, ...], *layers)[0].tolist()
        if expected != cxx:
            raise ValueError(f"C++/Python mismatch: {key}")
        reference = evaluate(args.reference, wav)
        rescue = decide(cxx, limits)
        return {"id": key, "panel": kind, "expected": expect, "baseline": reference,
                "fallback": rescue, "combined": reference or rescue}

    output = [observe(key, wav, LABELS[category], "synthetic-test", data)
              for split, category, key, wav, data in prepared if split == "test" and data is not None]
    for index, folder in enumerate(PUBLIC_PANELS):
        for row in panel(args.experiment_root, index):
            if row["overlong"]:
                continue
            key = row["id"]
            wav = args.experiment_root / folder / f"{key}.wav"
            output.append(observe(key, wav, row["expected"], "public-replay"))
    for group in ("003", "042"):
        key = f"hora-mty-{group}-esmx"
        wav = args.experiment_root / "primer-candidato-voces" / f"{key}.wav"
        output.append(observe(key, wav, 5, "public-time-reserved"))
    root = args.experiment_root / "d6-round1" / "audio"
    output.append(observe("test-0379", root / "test-0379.wav", 0, "fact-veto"))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("feature-tool", "reference", "experiment-root", "ald", "piper-python", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.feature_tool = args.feature_tool.resolve(strict=True)
    args.reference = args.reference.resolve(strict=True)
    args.experiment_root = args.experiment_root.resolve(strict=True)
    args.ald = args.ald.resolve(strict=True)
    args.piper_python = args.piper_python.absolute()
    args.output = args.output.resolve()
    if digest(args.ald) != ALD_SHA256:
        parser.error("unexpected Piper Ald checkpoint")
    args.output.mkdir(mode=0o700)
    fixture = Path(__file__).with_name("fallback_cases.json")
    prepared = prepare(args, json.loads(fixture.read_text()))
    train = [(data, LABEL_ORDER.index(LABELS[category]))
             for split, category, _, _, data in prepared if split == "train" and data is not None]
    adjust = args.experiment_root / "primer-candidato-voces" / "hora-ajuste-spanishpod-a0056-esmx.wav"
    adjust_sha = "0319e9b81c27930015d381c7f6a6c5aa31bc862509c46815814ba681936d3751"
    if digest(adjust) != adjust_sha:
        parser.error("unexpected SpanishPod adjustment clip")
    train += [(vector(args, adjust), LABEL_ORDER.index(LABELS["time"]))] * 6
    features = torch.from_numpy(np.array([data for data, _ in train])).float() / 32
    model = fit(features, [label for _, label in train])
    layers, weights, offsets, unit = quantize(model, features)
    dev = [(data, LABELS[category]) for split, category, _, _, data in prepared
           if split == "dev" and data is not None]
    scores = integer_scores(np.array([data for data, _ in dev]), layers, weights, offsets)
    limits = calibrate(scores, np.array([label for _, label in dev]), unit)
    bank = args.output / "fallback.bin"
    bank.write_bytes(wire_bank(layers, weights, offsets, 0, 1))
    config = {"bank_sha256": digest(bank), "fixture_sha256": digest(fixture),
              "train_adjustment_sha256": adjust_sha, "ald_model_sha256": ALD_SHA256, "limits": limits,
              "seed": 20261127, "bank_bytes": bank.stat().st_size,
              "calibration": "synthetic-dev only", "dev_count": len(dev), "train_count": len(train),
              "cpp_backend": digest(args.feature_tool), "recognizer": digest(args.reference)}
    (args.output / "frozen.json").write_text(json.dumps(config, indent=2) + "\n")
    results = check_rows(args, prepared, limits, bank, (layers, weights, offsets))
    missing = [key for _, _, key, _, data in prepared if data is None]
    print(json.dumps({"config": config, "excluded_unusable": missing, "rows": results}, indent=2))


if __name__ == "__main__":
    main()
