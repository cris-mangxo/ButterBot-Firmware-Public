"""Reproduce the laboratory D6 experiment with shorter spoken forms."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from ald_round import ALD_SHA256, canonical, digest
from model_sources import SHARVARD_CONFIG_SHA256, SHARVARD_SHA256

ALD_CONFIG_SHA256 = "f0c958eed7d2de607746a9b3b140f0eafa9c58ab08eef472d0105f852ed5d3c4"

PATTERNS = {
    "train": (
        "Tíralo de {}", "Lánzalo de {}", "Avienta uno de {}", "Oye, echa el de {}", "Tira tu de {}",
        "Hazme un tiro de {}", "A ver, tira el dadito de {} aquí", "Tira el de {} ahí", "Mira, usa el de {}",
        "Ya, rueda uno de {}", "¿Me lanzas un de {}?", "Échalo de {}", "Tírale un de {}",
        "Tira un dado de {} caras, porfa", "Ahora tú, avienta el de {}", "Lanza un dado de {} de una vez",
    ),
    "dev": (
        "Porfa tira el de {}", "Ayúdame a tirar uno de {}", "Tíralo de {} cuando puedas", "Va, echa un de {}",
        "¿Me tiras un dado de {} caras?", "Ahora haz un tiro con el de {}", "Vuelve a tirar el de {}",
    ),
}
EXTRA_VERBS = ("tiralo", "lanzalo", "avientalo", "echalo", "tirale", "lanzale",
               "aventarlo", "aventarle", "tiro", "tiren", "tiramos", "lanzamos")
EXTRA_DENIALS = ("cancela", "cancelen", "olvida", "olvidalo", "deten", "evita")
CATEGORIES = ("outside", "positive", "other_die", "negated")


def rows():
    seen = {}
    index = 0
    for split, patterns in PATTERNS.items():
        for position, pattern in enumerate(patterns):
            positive = pattern.format("seis")
            declarative = positive.lstrip("¿").rstrip("?")
            lower = declarative[:1].lower() + declarative[1:]
            contrasts = (
                (1, positive), (2, pattern.format("ocho" if position % 2 else "veinte")),
                (3, "No " + lower + "."), (3, positive + " No, mejor no."),
                (3, "Cancela lo de seis. No " + lower + "."), (2, pattern.format("dieciséis")),
                (0, "Dijo que " + lower + " hace rato."), (0, "Mi dado de seis está junto al tuyo."),
                (2, "Tira dos dados de seis." if position % 2 else "Aviéntame tres dados de seis."),
            )
            for label, phrase in contrasts:
                key = canonical(phrase)
                if key in seen:
                    if seen[key] != label:
                        raise ValueError("El texto tiene etiquetas en conflicto")
                    continue
                seen[key] = label
                for voice in range(3):
                    for speed in ((0.84, 1.13) if split == "train" else (0.93,)):
                        yield {"id": f"{split}-{index:04d}-0", "text": phrase, "split": split,
                               "label": label, "voice": voice, "speed": speed, "category": CATEGORIES[label]}
                        index += 1


def render(args):
    from corpus import initialize, waveform

    models = (args.ald, args.sharvard)
    hashes = [digest(model) for model in models]
    config_hashes = [digest(model.with_suffix(".onnx.json")) for model in models]
    if hashes != [ALD_SHA256, SHARVARD_SHA256] or config_hashes != [ALD_CONFIG_SHA256, SHARVARD_CONFIG_SHA256]:
        raise ValueError("Los modelos Piper difieren de las voces fijadas")
    output = args.output / "audio"
    args.output.mkdir(mode=0o700)
    output.mkdir(mode=0o700)
    manifest = list(rows())
    for model_index, model in enumerate(models):
        tasks = [(output, row, args.aligned, row["voice"] - 1 if row["voice"] else None)
                 for row in manifest if bool(row["voice"]) == model_index]
        with ProcessPoolExecutor(2, initializer=initialize, initargs=(model, args.aligned)) as workers:
            list(workers.map(waveform, tasks, chunksize=8))
    (args.output / "manifest.json").write_text(json.dumps({
        "synthesis_sha256": hashes, "samples": manifest,
    }, indent=2, ensure_ascii=False), encoding="utf-8")


def synthetic_rows(root):
    info = json.loads((root / "manifest.json").read_text("utf-8"))
    expected = list(rows())
    if info != {"samples": expected, "synthesis_sha256": [ALD_SHA256, SHARVARD_SHA256]}:
        raise ValueError("El corpus adicional difiere de su receta")
    return expected


def fit_acoustic(args):
    import numpy as np
    import torch
    from train import (choose_limits, collect, fit, integer_scores, quantize,
                       read_manifest, verify_wire, wire_bank)
    from whole_orders import read_manifest as human_manifest

    torch.set_num_threads(4)
    torch.manual_seed(1187)
    groups, original = collect(args.ald_turns, read_manifest(args.ald_turns), args.program)
    human, human_omissions = collect(args.human, human_manifest(args.human)[0], args.program)
    added, new_omissions = collect(args.corpus, synthetic_rows(args.corpus), args.program)
    for split, repeats in (("train", 8), ("dev", 1)):
        groups[split].extend(added[split])
        groups[split].extend(human[split] * repeats)
    train, train_labels, _ = zip(*groups["train"], strict=True)
    dev, dev_labels, _ = zip(*groups["dev"], strict=True)
    train_tensor = torch.from_numpy(np.stack(train)).float() / 32
    dev_features, dev_labels = np.stack(dev), np.asarray(dev_labels)
    human_dev = len(human["dev"])
    model, epoch = fit(train_tensor, torch.from_numpy(dev_features).float() / 32,
                       torch.from_numpy(np.asarray(train_labels)), dev_labels, priority_tail=human_dev)
    stages, weights, bias, scale = quantize(model, train_tensor)
    scores = integer_scores(dev_features, stages, weights, bias)
    _, hits, false, gap, threshold = choose_limits(scores, dev_labels, scale, human_dev)
    with args.bank.open("xb") as output:
        output.write(wire_bank(stages, weights, bias, threshold, gap))
    return {
        "model_sha256": digest(args.bank), "epoch": epoch, "development_d6": hits,
        "development_positives": int(sum(dev_labels == 1)), "development_false_d6": -false,
        "threshold": threshold, "gap": gap, "train": len(train_labels), "dev": len(dev_labels),
        "parity_cpp_turns": verify_wire(args.program, args.bank, groups["dev"], scores),
        "omitted": [original, human_omissions, new_omissions],
    }


def fit_words(args):
    import torch
    import word_train as words
    from human_words import read_manifest
    from unittest.mock import patch

    torch.set_num_threads(4)
    torch.manual_seed(1151)
    vocabulary = {**words.VOCABULARY, **dict.fromkeys(EXTRA_VERBS, words.VOCABULARY["tira"]),
                  **dict.fromkeys(EXTRA_DENIALS, words.VOCABULARY["no"])}
    with patch.object(words, "VOCABULARY", vocabulary):
        groups, original = words.load(args.ald_turns, args.program)
        human, human_omissions = words.load_rows(args.human, args.program, read_manifest(args.human)[0])
        added, new_omissions = words.load_rows(args.corpus, args.program, synthetic_rows(args.corpus))
    groups["train"].extend(human["train"] * 2 + added["train"])
    groups["dev"].extend(human["dev"] + added["dev"])
    model, epoch, training = words.fit(groups, multi_voice=True)
    weights, unit = words.quantize(model, training)
    result = words.score_dev(args.program, args.bank, groups, weights, unit)
    return {
        **result, "epoch": epoch, "model_sha256": digest(args.bank),
        "train": len(groups["train"]), "dev": len(groups["dev"]),
        "omitted": [original, human_omissions, new_omissions],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    operations = parser.add_subparsers(dest="operation", required=True)
    renderer = operations.add_parser("render")
    for name in ("ald", "sharvard", "output"):
        renderer.add_argument(f"--{name}", type=Path, required=True)
    renderer.add_argument("--aligned", action="store_true")
    for name in ("acoustic", "words"):
        fitting = operations.add_parser(name)
        for option in ("program", "ald-turns", "human", "corpus", "bank"):
            fitting.add_argument(f"--{option}", type=Path, required=True)
    args = parser.parse_args()
    if args.operation == "render":
        render(args)
    else:
        fit = fit_words if args.operation == "words" else fit_acoustic
        print(json.dumps(fit(args), indent=2))


if __name__ == "__main__":
    main()
