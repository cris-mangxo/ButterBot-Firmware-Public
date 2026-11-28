"""Compare forced alignment with the previous Anáhuac word-coverage result."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
from statistics import median
import subprocess
from tempfile import TemporaryDirectory
import wave

from cv27_word_prep import MANIFEST_SHA, SAMPLE_RATE, SANDBOX, digest, normalized_words, prepare, wave_bytes


MODEL_PINS = {
    "acoustic": "e081a65fbb7982eff4257dbeb801d66f677a4250ea99966761060ab42afe8677",
    "dictionary": "cf4bb8ea69c6042d17264ae1ef335a528092d4187e174dde274f9bd636601474",
    "reference": "79851e2ea743c1312d6a0c58eb4e0751bab618b1de02dfde2240371d0a9529bc",
}
FLAGS = ("--single_speaker", "--num_jobs", "2", "--output_format", "json", "--clean")
MSWC_MANIFEST_SHA = "93aafb8eac37120ba493d3a4c340060a81a01ef568df08e625ef9a16c19de243"


def word_intervals(tiers):
    intervals = []
    for name, tier in tiers.items():
        if name == "words" or name.endswith(" - words"):
            intervals += tier["entries"]
    observed = []
    for start, end, text in intervals:
        tokens = normalized_words(text)
        if tokens:
            # Keep multi-word entries together so they cannot match one label.
            observed.append((" ".join(tokens), round(start * SAMPLE_RATE), round(end * SAMPLE_RATE)))
    return observed


def target_span(tiers, word, transcript, sample_count):
    observed = word_intervals(tiers)
    labels = [item[0] for item in observed]
    if labels.count(word) != 1 or transcript.count(word) != 1:
        return None, "target_missing_or_repeated"
    actual, expected = labels.index(word), transcript.index(word)
    for step in (-1, 1):
        if 0 <= expected + step < len(transcript):
            if not 0 <= actual + step < len(labels) or labels[actual + step] != transcript[expected + step]:
                return None, "neighbor_unmatched"
    _, start, end = observed[actual]
    if not 0 <= start < end <= sample_count or not 1440 <= end - start <= 14_400:
        return None, "bounds_or_duration"
    return (start, end), "development_only"


def align(args, work, corpus, log_file):
    annotations = work / "aligned"
    env = os.environ.copy()
    env["MFA_ROOT_DIR"] = str(work / "state")
    env["PATH"] = f"{args.mfa.parent}:{env['PATH']}"
    command = ["/usr/bin/sandbox-exec", "-p", SANDBOX, args.mfa, "align",
               *FLAGS, corpus, args.dictionary, args.acoustic, annotations]
    with log_file.open("wb") as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=240)
    return annotations


def benchmark_mswc(args):
    manifest = args.mswc / "manifest.json"
    if digest(manifest) != MSWC_MANIFEST_SHA:
        raise ValueError("MSWC alignment fixture SHA-256 mismatch")
    metadata = json.loads(manifest.read_text())["samples"]
    dev = sorted((row for row in metadata if row["split"] == "dev"), key=lambda row: row["id"])
    cases = [row for category in range(4) for row in
             [item for item in dev if item["label"] == category][:4]]
    if len(cases) != 16:
        raise ValueError("MSWC fixture must include four utterances per category")
    reference = {}
    with TemporaryDirectory(prefix="bb-mfa-mswc-") as scratch:
        work = Path(scratch)
        corpus = work / "corpus"
        corpus.mkdir(mode=0o700)
        for row in cases:
            identifier = row["id"]
            wav = args.mswc / "audio" / f"{identifier}.wav"
            timing = wav.with_suffix(".json")
            if digest(wav) != row["wav_sha256"] or digest(timing) != row["annotation_sha256"]:
                raise ValueError("MSWC development sample differs from its seal")
            annotation = json.loads(timing.read_text())
            if annotation["sample_rate"] != SAMPLE_RATE:
                raise ValueError("unexpected MSWC sample rate")
            intervals = annotation["word_intervals"]
            reference[identifier] = intervals
            shutil.copyfile(wav, corpus / wav.name)
            (corpus / f"{identifier}.lab").write_text(" ".join(word for word, _, _ in intervals) + "\n")
        annotations = align(args, work, corpus, args.output / "mswc.log")
        overlaps = []
        errors_ms = []
        for identifier, known in reference.items():
            path = annotations / f"{identifier}.json"
            if not path.exists():
                continue
            observed = word_intervals(json.loads(path.read_text())["tiers"])
            if [word for word, _, _ in observed] != [word for word, _, _ in known]:
                continue
            for (_, start, end), (_, actual_start, actual_end) in zip(known, observed, strict=True):
                common = max(0, min(end, actual_end) - max(start, actual_start))
                union = max(end, actual_end) - min(start, actual_start)
                overlaps.append(common / union)
                errors_ms.extend(abs(error) * 1000 / SAMPLE_RATE
                                 for error in (start - actual_start, end - actual_end))
    return {
        "mswc_manifest_sha256": MSWC_MANIFEST_SHA,
        "sample_type": "development_word_montages",
        "utterances": len(cases),
        "reference_words": sum(map(len, reference.values())),
        "ordered_words_measured": len(overlaps),
        "words_overlap_at_least_half_union": sum(value >= 0.5 for value in overlaps),
        "median_boundary_error_ms": round(median(errors_ms), 3) if errors_ms else None,
    }


def run(args):
    args.output.mkdir(mode=0o700)
    reference = json.loads(args.reference.read_text())
    previous = {row["id"]: row["interval_ms"] for row in reference["clips"]}
    rows = []
    with TemporaryDirectory(prefix="bb-mfa-cv27-") as scratch:
        work = Path(scratch)
        corpus = work / "corpus"
        corpus.mkdir(mode=0o700)
        inputs = prepare(args, corpus)
        if Counter(word for _, word, *_ in inputs) != reference["attempted"]:
            raise ValueError("forced alignment must evaluate the same panel as the reference")
        for _, _, transcript, wav, _ in inputs:
            wav.with_suffix(".lab").write_text(" ".join(transcript) + "\n")
        annotations = align(args, work, corpus, args.output / "mfa.log")
        for index, word, transcript, wav, source_hash in inputs:
            identifier = f"anahuac-{index}"
            row = {"id": identifier, "word": word, "agreement_ms": previous.get(identifier),
                   "source_sha256": source_hash}
            rows.append(row)
            annotation = annotations / f"{index}.json"
            if not annotation.exists():
                row["status"] = "unaligned"
                continue
            tiers = json.loads(annotation.read_text())["tiers"]
            with wave.open(str(wav), "rb") as source:
                bounds, status = target_span(tiers, word, transcript, source.getnframes())
                row["status"] = status
                if bounds is None:
                    continue
                start, end = bounds
                source.setpos(start)
                pcm = source.readframes(end - start)
            output = args.output / f"{identifier}.wav"
            wave_bytes(output, pcm)
            row.update({"file": output.name, "wav_sha256": digest(output), "forced_samples": bounds,
                        "interval_ms": [round(value * 1000 / SAMPLE_RATE) for value in bounds],
                        "split": "development_only", "speaker_verified": False,
                        "locale_basis": "self_reported_mexicano"})
    clips = [row for row in rows if row["status"] == "development_only"]
    result = {"source_manifest_sha256": MANIFEST_SHA, "license": "CC0-1.0",
              "mfa_version": "3.4.0", "mfa_flags": FLAGS, "model_pins": MODEL_PINS,
              "ffmpeg_sha256": digest(args.ffmpeg), "attempted": len(rows),
              "accepted": Counter(row["word"] for row in clips),
              "statuses": Counter(row["status"] for row in rows), "clips": clips, "rows": rows}
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    if args.mswc:
        validation = benchmark_mswc(args)
        (args.output / "mswc-validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in {"clips", "rows"}}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output", "ffmpeg", "mfa", *MODEL_PINS):
        parser.add_argument(f"--{name}", type=lambda path: Path(path).resolve(), required=True)
    parser.add_argument("--mswc", type=lambda path: Path(path).resolve(strict=True))
    args = parser.parse_args()
    for name, checksum in MODEL_PINS.items():
        if digest(getattr(args, name)) != checksum:
            parser.error(f"incorrect {name} SHA-256")
    version = subprocess.check_output([args.mfa, "version"], text=True, timeout=20).strip()
    if version != "3.4.0":
        parser.error("requires MFA 3.4.0")
    run(args)


if __name__ == "__main__":
    main()
