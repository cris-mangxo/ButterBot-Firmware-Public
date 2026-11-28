"""Adapt only the phoneme output to public, transcribed Mexican Spanish."""

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch.nn import functional as F

import keyword_lab as kw
from phoneme_reject_lab import BASE_SHA256, NEGATIVE_CACHE_SHA256, load_negatives
from phoneme_replay_lab import frozen_model, load_int8_weights
from phoneme_student_lab import RECIPE, load_cache
from phoneme_teacher_lab import collapse, edit_distance, phonemes
from time_lab import file_digest


TARGET_SHA256 = "adc7fd33c18d28f7ce13395089ae658b55014e1677d7dabf1e818a32c65576dc"
METHOD = "bb-natural-phoneme-head-1"
EPOCHS = 30
SEED = 20261129
BATCH = 24
HEAD_ARRAYS = {"phones.weight", "phones.weight.scale", "phones.bias"}


def build_targets(args, synthetic, natural):
    import duckdb
    from phoneme_chm150_lab import CORPUS_SHA256

    if file_digest(args.corpus) != CORPUS_SHA256:
        raise ValueError("Unexpected CHM150 source")
    with duckdb.connect(config={"autoload_known_extensions": False, "autoinstall_known_extensions": False}) as db:
        transcripts = dict(db.execute(
            "SELECT audio_id, normalized_text FROM read_parquet(?) WHERE audio_id=ANY(?)",
            [str(args.corpus), sorted(natural)],
        ).fetchall())
    if transcripts.keys() != natural.keys():
        raise ValueError("CHM150 metadata missing transcripts for designated roles")
    lookup = {symbol: index for index, symbol in enumerate(synthetic["vocabulary"])}
    unknown = lookup["<other>"]
    records = {}
    for key, row in sorted(natural.items()):
        tokens = phonemes(str(args.espeak), transcripts[key].strip() + ".")
        phones = [lookup.get(symbol, unknown) for symbol in tokens]
        if (not phones or phones.count(unknown) * 5 > len(phones)
                or len(phones) + sum(a == b for a, b in zip(phones, phones[1:])) > row["frames"]):
            continue
        if 0 in phones:
            raise ValueError("CTC target contains the blank symbol")
        records[key] = {"split": row["split"], "phones": phones, "features_sha256": row["feature_sha256"]}
    data = {"schema": "bb-chm150-natural-phones-1", "native_negative_index_sha256": NEGATIVE_CACHE_SHA256,
            "synthetic_sha256": file_digest(args.cache / "index.json"), "vocabulary": synthetic["vocabulary"],
            "records": records}
    with args.targets.open("x") as output:
        output.write(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return {"targets_sha256": file_digest(args.targets), "counts": Counter(row["split"] for row in records.values())}


def direct(model, frames):
    data = torch.from_numpy(frames.astype(np.float32).T[None] / 32)
    return model(data, torch.tensor([len(frames)]))


@torch.inference_mode()
def encoded_groups(model, speech, targets, natural, directory):
    capture = []
    hook = model.phones.register_forward_pre_hook(lambda _, args: capture.append(args[0][0].detach().clone()))
    groups = {"train": [], "dev": []}
    try:
        for split, rows in speech.items():
            for _, frames, _, phones in rows:
                direct(model, frames)
                groups[split].append(("synthetic", capture.pop(), tuple(phones.tolist()), frames))
        for key, row in targets.items():
            parent = natural[key]
            path = directory / f"{key}.npz"
            if row["split"] != parent["split"] or row["features_sha256"] != file_digest(path):
                raise ValueError("Natural phone target differs from its assigned role or waveform seal")
            with np.load(path, allow_pickle=False) as archive:
                frames = archive["features"]
            direct(model, frames)
            groups[row["split"]].append(("CHM150", capture.pop(), tuple(row["phones"]), frames))
    finally:
        hook.remove()
    return groups


def train_head(initial, samples):
    """The optimizer receives only the phoneme layer, with a cached frozen encoder."""
    head = copy.deepcopy(initial)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.003, weight_decay=0.001)
    for _ in range(EPOCHS):
        order = torch.randperm(len(samples)).tolist()
        for start in range(0, len(samples), BATCH):
            batch = [samples[index] for index in order[start:start + BATCH]]
            lengths = torch.tensor([row[1].shape[1] for row in batch])
            frames = torch.zeros(len(batch), head.in_channels, int(lengths.max()))
            for i, (_, hidden, _, _) in enumerate(batch):
                frames[i, :, :hidden.shape[1]] = hidden
            phones = torch.tensor([symbol for row in batch for symbol in row[2]])
            expected = torch.tensor([len(row[2]) for row in batch])
            scores = head(frames).log_softmax(dim=1)
            loss = F.ctc_loss(scores.permute(2, 0, 1), phones, lengths, expected)
            if not loss.isfinite():
                raise ValueError("Natural phone adaptation produced non-finite loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return head


def export_head(head, original):
    weights = head.weight.detach()
    scale = weights.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-8) / 127
    quantized = (weights / scale).round().clamp(-127, 127).to(torch.int8)
    with np.load(original, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays.update({"phones.weight": quantized.numpy(), "phones.weight.scale": scale.numpy(),
                   "phones.bias": head.bias.detach().numpy()})
    return arrays


@torch.inference_mode()
def measure_heads(initial, updated, rows):
    counts = {bank: {"synthetic": Counter(), "CHM150": Counter()} for bank in ("initial", "updated")}
    invariance = Counter()
    for origin, _, expected, frames in rows:
        before, after = direct(initial, frames), direct(updated, frames)
        if not torch.equal(before[1], after[1]) or not torch.equal(before[2], after[2]):
            raise ValueError("Head-only adjustment changed the intention or negation classifier")
        invariance[origin] += 1
        for name, output in (("initial", before), ("updated", after)):
            prediction = collapse(output[0][0].argmax(-1).tolist())
            counts[name][origin].update(turns=1, phones=len(expected), edits=edit_distance(prediction, expected))
    return counts, invariance


def fit(args, synthetic, natural):
    if file_digest(args.targets) != TARGET_SHA256:
        raise ValueError("Unexpected natural phoneme targets")
    target_data = json.loads(args.targets.read_text())
    synthetic_sha = file_digest(args.cache / "index.json")
    if (target_data["synthetic_sha256"] != synthetic_sha or target_data["vocabulary"] != synthetic["vocabulary"]
            or target_data["native_negative_index_sha256"] != NEGATIVE_CACHE_SHA256):
        raise ValueError("Natural target vocabulary or provenance differs")
    torch.manual_seed(SEED)
    initial, base_report = frozen_model(args.base)
    if base_report["weights_sha256"] != BASE_SHA256 or base_report["teacher_cache_sha256"] != synthetic_sha:
        raise ValueError("Unexpected student initialization")
    _, speech = load_cache(args.cache)
    targets = target_data["records"]
    features = encoded_groups(initial, speech, targets, natural, args.negatives)
    head = train_head(initial.phones, features["train"])
    archive = export_head(head, args.base / "student-int8-weights.npz")
    with TemporaryDirectory(dir=args.output.parent) as temporary:
        bank = Path(temporary) / "student-int8-weights.npz"
        np.savez_compressed(bank, **archive)
        labels = base_report["labels"]
        evaluation, invariance = measure_heads(initial, load_int8_weights(bank, labels), features["dev"])
        report = {"schema": RECIPE, "procedure": METHOD, "teacher_cache_sha256": synthetic_sha,
                  "negative_cache_sha256": file_digest(args.negatives / "index.json"), "target_sha256": TARGET_SHA256,
                  "initial_weights_sha256": BASE_SHA256, "weights_sha256": file_digest(bank),
                  "labels": labels, "epochs": EPOCHS, "seed": SEED, "training_turns": len(features["train"]),
                  "versions": {"torch": str(torch.__version__), "numpy": np.__version__},
                  "evaluation": evaluation, "unchanged_intent_and_negation": invariance}
        args.output.mkdir(mode=0o700)
        bank.rename(args.output / bank.name)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def existing(path):
    return Path(path).resolve(strict=True)


def main():
    common = argparse.ArgumentParser(add_help=False)
    for key in ("cache", "negatives", "speaker-split"):
        common.add_argument("--" + key, type=existing, required=True)
    common.add_argument("--targets", type=Path, required=True)
    parser = argparse.ArgumentParser(description=__doc__)
    children = parser.add_subparsers(dest="stage", required=True)
    prepare = children.add_parser("prepare", parents=[common])
    prepare.add_argument("--corpus", type=existing, required=True)
    prepare.add_argument("--espeak", type=existing, required=True)
    train = children.add_parser("train", parents=[common])
    train.add_argument("--base", required=True, type=existing)
    train.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    synthetic = json.loads((args.cache / "index.json").read_text())
    load_negatives(args.negatives, args.speaker_split, synthetic["frontend_sha256"])
    natural = json.loads((args.negatives / "index.json").read_text())["records"]
    result = build_targets(args, synthetic, natural) if args.stage == "prepare" else fit(args, synthetic, natural)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
