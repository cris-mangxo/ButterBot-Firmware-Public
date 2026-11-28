"""Train the existing compact phone student on transcribed Mexican speech."""
from argparse import ArgumentParser
from collections import Counter
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import re

import duckdb
import numpy as np
import torch
from torch.nn import functional as F

import keyword_lab as kw
from phoneme_chm150_lab import CORPUS_SHA256, SPEAKER_SPLIT_SHA256, decode
from phoneme_reject_lab import BASE_SHA256
from phoneme_replay_lab import frozen_model, native_frames
from phoneme_student_lab import BATCH, RECIPE, gate, load_cache, measure, quantize, supervised_loss
from phoneme_teacher_lab import collapse, edit_distance, known_codes
from time_lab import file_digest


EPOCHS = 16
SEED = 20261128
ARMS = {"reject_only": 0.0, "transcript_035": 0.35, "transcript_100": 1.0}
# Keep even dialectal mentions of the target terms out of this adaptation.
RESERVED = re.compile(r"\b(?:mantequilla|chistes?|horas?|dados?|seis)\b")


def write_json(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def prepare(args, roles, vocabulary):
    options = {"autoload_known_extensions": False, "autoinstall_known_extensions": False}
    speakers = sorted(key for key, role in roles.items() if role in ("train", "dev"))
    with duckdb.connect(config=options) as conn:
        rows = conn.execute(
            "SELECT audio_id, speaker_id, normalized_text, audio.bytes FROM read_parquet(?) "
            "WHERE speaker_id=ANY(?) AND duration BETWEEN 0.35 AND 4 ORDER BY audio_id",
            [str(args.corpus), speakers],
        ).fetchall()
    phones = {item: i for i, item in enumerate(vocabulary)}

    def extract(row):
        identifier, speaker, text, wav = row
        text = text.lower()
        if not re.fullmatch(r"[a-záéíóúüñ ]+", text) or RESERVED.search(text):
            return None
        if not re.fullmatch(r"[A-Za-z0-9_-]+", identifier):
            raise ValueError("Unexpected audio identifier")
        pcm = decode(args.ffmpeg, wav)
        matrix = native_frames(args.frontend, pcm)
        if matrix is None:
            return None
        target = known_codes(args.espeak, text, phones)
        repeats = sum(a == b for a, b in zip(target, target[1:]))
        if len(target) < 3 or len(target) + repeats > len(matrix) or target.count(phones["<other>"]) * 4 > len(target):
            return None
        return (identifier, speaker, matrix, target, sha256(kw.trimmed(pcm)).hexdigest())

    groups = {"train": [], "dev": []}
    seen = set()
    pcm_sets = {split: set() for split in groups}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for row in pool.map(extract, rows):
            if row is None:
                continue
            identifier, speaker, _, _, checksum = row
            if identifier in seen:
                raise ValueError("Duplicate input identifier")
            seen.add(identifier)
            part = roles[speaker]
            if checksum not in pcm_sets[part]:
                pcm_sets[part].add(checksum)
                groups[part].append(row)
    if not all(groups.values()) or pcm_sets["train"] & pcm_sets["dev"]:
        raise ValueError("Empty human split or repeated PCM across splits")
    return groups


def pack_human(rows):
    lengths = torch.tensor([len(row[2]) for row in rows])
    widths = torch.tensor([len(row[3]) for row in rows])
    x = torch.zeros((len(rows), kw.BANDS, int(max(lengths))))
    for index, (_, _, frames, _, _) in enumerate(rows):
        x[index, :, :len(frames)] = torch.from_numpy(frames.T.copy()) / 32
    target = torch.tensor([code for row in rows for code in row[3]])
    return x, lengths, widths, target


@torch.inference_mode()
def validate_human(model, rows):
    stats = Counter()
    for start in range(0, len(rows), BATCH):
        batch = rows[start:start + BATCH]
        x, n, _, _ = pack_human(batch)
        phones, intents, negation = model(x, n)
        for i, (choice, row) in enumerate(zip(gate(intents, negation), batch, strict=True)):
            hypothesis = collapse(phones[i, :n[i]].argmax(dim=-1).tolist())
            stats["phone_edit_distance"] += edit_distance(hypothesis, row[3])
            stats["phones"] += len(row[3])
            stats["false_dispatches"] += bool(choice)
            stats["utterances"] += 1
    return {**stats, "speaker_count": len({row[1] for row in rows})}


def fit(model, synthetic, human, phone_weight):
    torch.manual_seed(SEED)
    model = deepcopy(model).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0006, weight_decay=0.001)
    classes = torch.tensor([kw.LABELS.index(row["intent"]) for row, *_ in synthetic])
    class_counts = torch.bincount(classes, minlength=len(kw.LABELS))
    if (class_counts == 0).any():
        raise ValueError("Missing intention in synthetic replay")
    weights = class_counts.float().rsqrt()
    for _ in range(EPOCHS):
        a = torch.randperm(len(synthetic)).tolist()
        b = torch.randperm(len(human)).tolist()
        for start in range(0, len(b), BATCH):
            first = start % len(a)
            selected = (a + a)[first:first + BATCH]
            commands = [synthetic[i] for i in selected]
            x, lengths, widths, target = pack_human([human[i] for i in b[start:start + BATCH]])
            phones, intent, _ = model(x, lengths)
            rejection = F.cross_entropy(intent, torch.zeros(len(x), dtype=torch.long))
            loss = supervised_loss(model, commands, weights) + 0.65 * rejection
            if phone_weight:
                ctc = F.ctc_loss(phones.log_softmax(-1).transpose(0, 1), target, lengths, widths)
                loss = loss + phone_weight * ctc
            if not loss.isfinite():
                raise ValueError("Non-finite adaptation loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0, error_if_nonfinite=True)
            optimizer.step()
    return model.eval()


def main():
    parser = ArgumentParser(description=__doc__)
    for key in ("corpus", "speakers", "speech", "bank", "frontend", "ffmpeg", "espeak", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    if file_digest(args.corpus) != CORPUS_SHA256 or file_digest(args.speakers) != SPEAKER_SPLIT_SHA256:
        raise ValueError("Expected the sealed CHM150 corpus and speaker assignment")
    base, original = frozen_model(args.bank)
    frontend = file_digest(args.frontend)
    speech = args.speech / "index.json"
    if (original["weights_sha256"] != BASE_SHA256 or original["teacher_cache_sha256"] != file_digest(speech)
            or json.loads(speech.read_text())["frontend_sha256"] != frontend):
        raise ValueError("Original student and synthetic cache must match")
    vocabulary, synthetic = load_cache(args.speech)
    roles = json.loads(args.speakers.read_text())["speaker_roles"]
    human = prepare(args, roles, vocabulary)
    args.output.mkdir(mode=0o700)
    manifest = {
        "schema": "bb-chm150-transcript-features-1",
        "corpus": CORPUS_SHA256,
        "speaker_assignment": SPEAKER_SPLIT_SHA256,
        "tools": {key: file_digest(getattr(args, key)) for key in ("frontend", "espeak", "ffmpeg")},
        "entries": {part: [{"id": name, "speaker_partition_id": speaker, "frames":len(frames),
                            "phones":len(target), "pcm_sha256": checksum,
                            "feature_sha256":sha256(frames.tobytes()).hexdigest()}
                           for name, speaker, frames, target, checksum in rows] for part, rows in human.items()},
    }
    index = args.output / "local-input.json"
    write_json(index, manifest)
    result = {"baseline": {"human_development": validate_human(base, human["dev"]),
                           "synthetic_development": measure(base, synthetic["dev"])}}
    for name, phone_weight in ARMS.items():
        arrays, model = quantize(fit(base, synthetic["train"], human["train"], phone_weight))
        directory = args.output / name
        directory.mkdir(mode=0o700)
        weights = directory / "student-int8-weights.npz"
        np.savez_compressed(weights, **arrays)
        report = {
            "schema": RECIPE,
            "procedure": "bb-chm150-transcript-ctc-1",
            "train_turns": len(human["train"]),
            "train_speakers": len({row[1] for row in human["train"]}),
            "epochs": EPOCHS,
            "seed": SEED,
            "weight_scale_bias_bytes": sum(value.nbytes for value in arrays.values()),
            "labels": len(vocabulary),
            "weights_sha256": file_digest(weights),
            "initial_weights_sha256": BASE_SHA256,
            "teacher_cache_sha256": file_digest(speech),
            "human_manifest_sha256": file_digest(index),
            "human_phone_loss_weight": phone_weight,
            "human_development": validate_human(model, human["dev"]),
            "synthetic_development": measure(model, synthetic["dev"]),
        }
        write_json(directory / "report.json", report)
        result[name] = report
        print(name, json.dumps(report["human_development"]), flush=True)
    write_json(args.output / "summary.json", result)


if __name__ == "__main__":
    main()
