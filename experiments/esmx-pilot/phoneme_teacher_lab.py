"""Compare local phoneme evidence on synthetic development data and prepare distillation targets."""

import argparse
from collections import Counter
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

import keyword_lab as kw
from capture_replay import pcm_from
from cv27_word_prep import SANDBOX
from time_lab import file_digest, invoke


WEIGHTS_SHA256 = "04366b6c8d24099ef313cf02f0e58d26f5dddfda16edbfc8eb2c713d94a9f551"
SAMPLE_RATE = 16_000
SUPPORTED_LABELS = 33
UNKNOWN = "<other>"
STRIDE = 320
TEACHER_WINDOW = 400
FRONTEND_WINDOW = 480


@lru_cache(maxsize=None)
def phonemes(espeak, phrase):
    stresses = str.maketrans("", "", "ˈˌ")
    result = subprocess.run(
        ["/usr/bin/sandbox-exec", "-p", SANDBOX, espeak, "-q", "--ipa=1", "-v", "es-419", "--stdin"],
        input=phrase, capture_output=True, check=True, timeout=10, text=True,
    )
    words = result.stdout.translate(stresses).split()
    return tuple(phone for word in words for phone in word.split("_") if phone)


def teacher_labels(espeak, phrases, vocabulary):
    known = sorted({phone for phrase in phrases for phone in phonemes(espeak, phrase)})
    missing = set(known) - vocabulary.keys()
    if missing or not known or len(known) + 2 > SUPPORTED_LABELS:
        raise ValueError(f"Training inventory exceeds compact teacher coverage: {sorted(missing)}")
    return ("<pad>", *known, UNKNOWN)


def collapse(indices):
    out = []
    previous = None
    for index in indices:
        if index and index != previous:
            out.append(index)
        previous = index
    return tuple(out)


def edit_distance(left, right):
    costs = list(range(len(right) + 1))
    for i, item in enumerate(left, 1):
        row = [i]
        for j, other in enumerate(right, 1):
            row.append(min(row[-1] + 1, costs[j] + 1, costs[j - 1] + int(item != other)))
        costs = row
    return costs[-1]


def nearest_intent(hypothesis, templates):
    """A fixed diagnostic over training text only. Abstain on ties or weak matches."""
    if not hypothesis:
        return 0
    best = {}
    for codes, label in templates:
        distance = edit_distance(hypothesis, codes) / max(len(codes), len(hypothesis))
        best[label] = min(best.get(label, 1.0), distance)
    (distance, intent), (rival, _) = sorted((value, key) for key, value in best.items())[:2]
    return intent if distance <= 0.4 and rival - distance >= 0.05 else 0


def project(logits, lookup, classes):
    """Merge unfamiliar symbols into a separate class, keeping CTC blank distinct."""
    probabilities = logits.softmax(dim=-1)
    compact = torch.zeros((*logits.shape[:-1], classes), dtype=probabilities.dtype)
    return compact.scatter_add_(-1, lookup.expand_as(probabilities), probabilities).numpy()


def align_frontend(probabilities, frame_count):
    offset = (FRONTEND_WINDOW - TEACHER_WINDOW) / (2 * STRIDE)
    positions = np.arange(frame_count, dtype=np.float32) + offset
    before = positions.astype(np.int64)
    after = np.minimum(before + 1, len(probabilities) - 1)
    fraction = (positions - before).astype(np.float32)[:, None]
    return probabilities[before] * (1 - fraction) + probabilities[after] * fraction


def prepared_pcm(row):
    if row.voice:
        with TemporaryDirectory(prefix="bb-phoneme-synth-") as scratch:
            voice, rate = row.voice
            path = kw.render(Path(scratch), row.key, voice, row.phrase, rate)
            return kw.trimmed(pcm_from(path))
    return kw.trimmed(pcm_from(row.wav, row.sha))


def known_codes(espeak, phrase, index):
    return tuple(index.get(phone, index[UNKNOWN]) for phone in phonemes(espeak, phrase))


def feature_rows(program, cases):
    for row in cases:
        pcm = prepared_pcm(row)
        if not pcm:
            raise ValueError(f"{row.key}: synthetic development audio outside the frontend window")
        data = invoke(program, pcm, "word-frames")
        if len(data) % kw.BANDS:
            raise ValueError(f"{row.key}: incomplete frontend frame")
        features = np.frombuffer(data, dtype=np.int8).reshape(-1, kw.BANDS)
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32_768
        expected = 1 + (len(samples) - FRONTEND_WINDOW) // STRIDE
        if len(features) < 8 or len(features) != expected:
            raise ValueError(f"{row.key}: unexpected frontend frame geometry")
        yield row, samples, features


def load_teacher(directory):
    blob = directory / "pytorch_model.bin"
    if file_digest(blob) != WEIGHTS_SHA256:
        raise ValueError("Teacher weights differ from their official SHA-256")
    model, info = Wav2Vec2ForCTC.from_pretrained(
        directory, local_files_only=True, use_safetensors=False, weights_only=True, output_loading_info=True,
    )
    if any(info[key] for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("Teacher tensor inventory differs from the pinned model")
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(directory, local_files_only=True)
    if extractor.sampling_rate != SAMPLE_RATE or not extractor.do_normalize:
        raise ValueError("Teacher frontend normalization is incompatible")
    vocabulary = json.loads((directory / "vocab.json").read_text())
    if (sorted(vocabulary.values()) != list(range(model.config.vocab_size)) or vocabulary.get("<pad>") != 0
            or model.config.pad_token_id != 0 or tuple(model.config.conv_stride) != (5, 2, 2, 2, 2, 2, 2)
            or tuple(model.config.conv_kernel) != (10, 3, 3, 3, 3, 2, 2)):
        raise ValueError("Teacher vocabulary or temporal geometry is incompatible")
    return model.eval(), extractor, vocabulary


def teacher_outputs(model, extractor, inputs, lookup, classes):
    packed = extractor(
        [item[1] for item in inputs], sampling_rate=SAMPLE_RATE, padding=True, return_tensors="pt",
    )
    with torch.inference_mode():
        logits = model(**packed).logits
        for (_, pcm, _), prediction in zip(inputs, logits, strict=True):
            frames = 1 + (len(pcm) - TEACHER_WINDOW) // STRIDE
            yield project(prediction[:frames], lookup, classes)


def run(args):
    args.skip_adjustment = True
    model, extractor, vocab = load_teacher(args.teacher)
    cases = list(kw.rows(args, ("train", "dev")))
    train = tuple({(row.phrase, row.intent): None for row in cases if row.split == "train"})
    labels = teacher_labels(str(args.espeak), (phrase for phrase, _ in train), vocab)
    index = {word: i for i, word in enumerate(labels)}
    translation = torch.empty(len(vocab), dtype=torch.long)
    for phone, origin in vocab.items():
        translation[origin] = index.get(phone, index[UNKNOWN])
    templates = [(known_codes(str(args.espeak), phrase, index), intent) for phrase, intent in train]
    prepared = sorted(feature_rows(args.features, cases), key=lambda item: len(item[1]))
    args.output.mkdir(mode=0o700)
    measures = {split: Counter() for split in ("train", "dev")}
    decisions = {split: Counter() for split in ("train", "dev")}
    rejects = {split: Counter() for split in ("train", "dev")}
    saved = {}
    for start in range(0, len(prepared), args.batch):
        batch = prepared[start:start + args.batch]
        outputs = teacher_outputs(model, extractor, batch, translation, len(labels))
        for (row, pcm, features), probabilities in zip(batch, outputs, strict=True):
            expected = known_codes(str(args.espeak), row.phrase, index)
            predicted = collapse(probabilities.argmax(axis=-1).tolist())
            observed = nearest_intent(predicted, templates)
            bucket = measures[row.split]
            bucket["turns"] += 1
            bucket["expected_phones"] += len(expected)
            bucket["edit_operations"] += edit_distance(predicted, expected)
            bucket["oov_reference_phones"] += expected.count(index[UNKNOWN])
            decisions[row.split][f"{row.intent}:{observed}"] += 1
            if row.denied():
                rejects[row.split]["negative_turns"] += 1
                rejects[row.split]["wrong_dispatch"] += bool(observed)
            cache = args.output / f"{row.key}.npz"
            np.savez_compressed(
                cache, features=features, teacher=align_frontend(probabilities, len(features)).astype(np.float16),
                phones=np.asarray(expected, dtype=np.int16),
            )
            saved[row.key] = {
                "split": row.split, "intent": row.intent, "denied": bool(row.denied()), "frames": len(features),
                "pcm_sha256": sha256(pcm.tobytes()).hexdigest(), "artifact_sha256": file_digest(cache),
            }
    report = {
        "schema": "bb-phoneme-development-1", "teacher_weights_sha256": WEIGHTS_SHA256, "vocabulary": labels,
        "teacher_metadata_sha256": {
            key: file_digest(args.teacher / key) for key in ("config.json", "preprocessor_config.json", "vocab.json")
        },
        "frontend_sha256": file_digest(args.features), "espeak_sha256": file_digest(args.espeak),
        "split_origin": "synthetic_mac_voices_and_ald",
        "numbers": measures, "confusion_expected_observed": decisions, "negation": rejects,
        "sample_rate": SAMPLE_RATE, "stride": STRIDE, "teacher_window": TEACHER_WINDOW,
        "firmware_frontend_window": FRONTEND_WINDOW,
    }
    (args.output / "index.json").write_text(json.dumps({**report, "records": saved}, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "wav", "teacher", "espeak", "features", "output"):
        parser.add_argument("--" + key, type=lambda path: Path(path).resolve(), required=True)
    parser.add_argument("--batch", type=int, default=4, choices=range(1, 9))
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(args)


if __name__ == "__main__":
    main()
