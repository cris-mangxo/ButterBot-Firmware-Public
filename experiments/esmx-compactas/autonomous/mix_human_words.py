"""Place aligned MSWC words in continuous Ald turns to keep their context."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import wave

import numpy as np

from ald_round import canonical, digest
from human_words import read_manifest
from pilot import read_pcm
from phrases import too_long
from train import read_manifest as read_ald_manifest
from word_train import VOCABULARY, word_ranges


def donors(folder, rows):
    values = defaultdict(dict)
    for row in rows:
        wav = folder / "audio" / f'{row["id"]}.wav'
        pcm = np.frombuffer(read_pcm(wav), dtype="<i2")
        alignment = json.loads(wav.with_suffix(".json").read_text("utf-8"))
        for (word, start, end), (speaker, source) in zip(alignment["word_intervals"], row["sources"], strict=True):
            key = canonical(word)
            if key in VOCABULARY:
                values[row["split"], key].setdefault(source, (speaker, pcm[start:end]))
    return {key: sorted(pool.items()) for key, pool in values.items()}


def render(wav, row, available, rng, output):
    alignment = json.loads(wav.with_suffix(".json").read_text("utf-8"))
    tokens = canonical(row["text"]).split()
    ranges = word_ranges(alignment["phones"])
    if len(tokens) != len(ranges) or too_long(wav):
        return None
    pcm = np.frombuffer(read_pcm(wav), dtype="<i2")
    intervals = [(round(start * 16000 / alignment["sample_rate"]), round(end * 16000 / alignment["sample_rate"]))
                 for start, end in ranges]
    if (not intervals or intervals[-1][1] > len(pcm)
            or any(end <= start for start, end in intervals)):
        return None
    eligible = [i for i, token in enumerate(tokens) if (row["split"], token) in available]
    if not eligible:
        return None
    # Preserve a random subset of the original words, including every token
    # without a human match. Align the splice to the original word boundaries.
    replaced = set(rng.sample(eligible, rng.randint(1, len(eligible))))
    segments, annotations, origins = [], [], []
    displacement = previous = 0
    for i, (token, (start, end)) in enumerate(zip(tokens, intervals, strict=True)):
        segments.append(pcm[previous:start])
        if i in replaced:
            clip_id, (speaker, speech) = rng.choice(available[row["split"], token])
            peak = int(np.max(np.abs(pcm[start:end].astype(np.int32))))
            gain = max(1200, peak) / max(1, int(np.max(np.abs(speech.astype(np.int32)))))
            scaled = np.rint(speech.astype(np.float32) * min(gain, 2.0))
            segment = np.clip(scaled, -32768, 32767).astype("<i2")
            origins.append([speaker, clip_id])
        else:
            segment = pcm[start:end]
        segments.append(segment)
        annotations.append([token, start + displacement, start + displacement + len(segment)])
        displacement += len(segment) - (end - start)
        previous = end
    segments.append(pcm[previous:])
    if not 480 <= len(pcm) + displacement <= 64000:
        return None
    with wave.open(str(output), "wb") as dst:
        dst.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        dst.writeframes(np.concatenate(segments).astype("<i2").tobytes())
    output.with_suffix(".json").write_text(json.dumps({
        "sample_rate": 16000, "word_intervals": annotations,
    }) + "\n", encoding="utf-8")
    return origins


def build(human, carrier, out):
    rows, evidence = read_manifest(human)
    available = donors(human, rows)
    source = read_ald_manifest(carrier)
    rng = random.Random(6082026)
    out.mkdir(mode=0o700)
    audio = out / "audio"
    audio.mkdir(mode=0o700)
    mixed = []
    for split, maximum in (("train", 160), ("dev", 40)):
        counter = 0
        for label in range(4):
            selected = [row for row in source if row["split"] == split and row["label"] == label]
            rng.shuffle(selected)
            written = 0
            for row in selected:
                name = f"{split}-{counter:04d}"
                wav = audio / f"{name}.wav"
                origins = render(carrier / "audio" / f'{row["id"]}.wav', row, available, rng, wav)
                if origins is None:
                    continue
                mixed.append({
                    "id": name, "split": split, "label": label, "text": row["text"], "sources": origins,
                    "wav_sha256": digest(wav), "annotation_sha256": digest(wav.with_suffix(".json")),
                })
                counter += 1
                written += 1
                if written == maximum:
                    break
    (out / "manifest.json").write_text(json.dumps({
        "revision": evidence["revision"], "reference_sha256": evidence["reference_sha256"],
        "reserved_common_voice_ids": evidence["reserved_common_voice_ids"],
        "eligible_word_counts": evidence["eligible_word_counts"],
        "method": "mswc_words_spliced_into_ald",
        "human_word_manifest_sha256": evidence["manifest_sha256"],
        "ald_carrier_manifest_sha256": digest(carrier / "manifest.json"),
        "samples": mixed,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    read_manifest(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("human", "carrier", "out"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    args = parser.parse_args()
    build(args.human, args.carrier, args.out)
