"""Fine-tune the frozen desktop phoneme student with unrelated Mexican Spanish speech."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import keyword_lab as kw
from phoneme_replay_lab import frozen_model
from phoneme_student_lab import BATCH, RECIPE, gate, load_cache, measure, quantize, supervised_loss
from time_lab import file_digest


NEGATIVE_CACHE_SHA256 = "b82de523cdb315a779333a20bd56d5c66634a9e89aafced8a291c96a7dbe1193"
BASE_SHA256 = "8129251b438936609c613a5baea3a1f4cda8014c93013a3485f56400c96307aa"
PROCEDURE = "bb-phoneme-reject-chm150-1"
EPOCHS = 40
SEED = 20261128


def load_negatives(directory, split_manifest, frontend_sha256):
    path = directory / "index.json"
    data = json.loads(path.read_text())
    assignment = json.loads(split_manifest.read_text())
    if (file_digest(path) != NEGATIVE_CACHE_SHA256 or data["schema"] != "bb-chm150-negative-frames-1"
            or data["split_sha256"] != file_digest(split_manifest)
            or data["source_sha256"] != assignment["parquet_sha256"]
            or data["extractor_sha256"] != frontend_sha256):
        raise ValueError("Negative speech labels, corpus assignment or frontend differ from their seals")
    groups = {"train": [], "dev": []}
    pcm_hashes = {key: set() for key in groups}
    for identifier, record in sorted(data["records"].items()):
        split, speaker, frames = (record[key] for key in ("split", "speaker", "frames"))
        if (split not in groups or assignment["speaker_roles"].get(speaker) != split
                or record["label"] != "out_of_domain"
                or not all(char.isalnum() or char in "-_" for char in identifier)):
            raise ValueError("Negative speech needs a designated training or development speaker")
        artifact_path = directory / f"{identifier}.npz"
        if file_digest(artifact_path) != record["feature_sha256"]:
            raise ValueError(f"Negative feature archive differs from its seal: {identifier}")
        with np.load(artifact_path, allow_pickle=False) as artifact:
            feature = artifact["features"]
        if feature.shape != (frames, kw.BANDS) or feature.dtype != np.int8 or not 8 <= frames <= 199:
            raise ValueError(f"Unexpected feature geometry: {identifier}")
        pcm_hashes[split].add(record["effective_pcm_sha256"])
        groups[split].append(feature)
    if not all(groups.values()) or pcm_hashes["train"] & pcm_hashes["dev"]:
        raise ValueError("Empty split or duplicate voice waveform across splits")
    return groups


def pack_negative(samples):
    lengths = torch.tensor([len(sample) for sample in samples])
    batch = torch.zeros((len(samples), kw.BANDS, int(lengths.max())))
    for index, sample in enumerate(samples):
        batch[index, :, :len(sample)] = torch.from_numpy(sample.T) / 32
    return batch, lengths


@torch.inference_mode()
def measure_rejection(model, samples):
    activated = Counter()
    for offset in range(0, len(samples), BATCH):
        frames, lengths = pack_negative(samples[offset:offset + BATCH])
        _, intent, negation = model(frames, lengths)
        activated.update(kw.LABELS[label] for label in gate(intent, negation).tolist() if label)
    return {"utterances": len(samples), "activated": sum(activated.values()), "activation_by_intent": activated}


def adapt(model, commands, negatives):
    torch.manual_seed(SEED)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00075, weight_decay=0.001)
    class_counts = torch.bincount(
        torch.tensor([kw.LABELS.index(row["intent"]) for row, *_ in commands]), minlength=len(kw.LABELS),
    )
    if (class_counts == 0).any():
        raise ValueError("The synthetic replay must retain all intention classes")
    weights = class_counts.float().rsqrt()
    for _ in range(EPOCHS):
        speech_order = torch.randperm(len(commands)).tolist()
        reject_order = torch.randperm(len(negatives)).tolist()
        for offset in range(0, len(speech_order), BATCH):
            reject_index = reject_order[offset % len(reject_order):offset % len(reject_order) + BATCH]
            if len(reject_index) < BATCH:
                reject_index += reject_order[:BATCH - len(reject_index)]
            speech = [commands[i] for i in speech_order[offset:offset + BATCH]]
            x, n = pack_negative([negatives[i] for i in reject_index])
            _, intent, _ = model(x, n)
            # Unrelated speech is a negative intention. It has no command-cancellation
            # label and must not supervise the phoneme or negation heads directly.
            rejection = F.cross_entropy(intent, torch.zeros(len(x), dtype=torch.long))
            loss = supervised_loss(model, speech, weights) + rejection
            if not loss.isfinite():
                raise ValueError("Non-finite rejection adaptation loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0, error_if_nonfinite=True)
            optimizer.step()
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "negatives", "speaker-split", "base"):
        parser.add_argument("--" + name, type=lambda path: Path(path).resolve(strict=True), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    labels, speech = load_cache(args.cache)
    synthetic_index = args.cache / "index.json"
    synthetic_sha = file_digest(synthetic_index)
    frontend_sha = json.loads(synthetic_index.read_text())["frontend_sha256"]
    human = load_negatives(args.negatives, args.speaker_split, frontend_sha)
    base, prior = frozen_model(args.base)
    if prior["weights_sha256"] != BASE_SHA256 or prior["teacher_cache_sha256"] != synthetic_sha:
        raise ValueError("Unexpected initialization checkpoint or synthetic training corpus")
    before = measure_rejection(base, human["dev"])
    optimized = adapt(base, speech["train"], human["train"])
    arrays, model = quantize(optimized)
    args.output.mkdir(mode=0o700)
    path = args.output / "student-int8-weights.npz"
    np.savez_compressed(path, **arrays)
    report = {
        "schema": RECIPE, "procedure": PROCEDURE, "seed": SEED, "epochs": EPOCHS,
        "teacher_cache_sha256": synthetic_sha, "negative_cache_sha256": NEGATIVE_CACHE_SHA256,
        "speaker_split_sha256": file_digest(args.speaker_split), "initial_weights_sha256": BASE_SHA256,
        "weights_sha256": file_digest(path), "labels": len(labels),
        "weight_scale_bias_bytes": sum(item.nbytes for item in arrays.values()),
        "train_turns": len(speech["train"]), "train_ood": len(human["train"]),
        "development_ood_baseline": before, "development_ood": measure_rejection(model, human["dev"]),
        "development_synthetic": measure(model, speech["dev"]),
    }
    document = json.dumps(report, indent=2) + "\n"
    (args.output / "report.json").write_text(document)
    print(document, end="")


if __name__ == "__main__":
    main()
