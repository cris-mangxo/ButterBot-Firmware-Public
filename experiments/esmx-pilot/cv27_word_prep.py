"""Prepare development-only human word spans from the CC0 Anáhuac subset."""

import argparse
from collections import Counter
from hashlib import file_digest, sha256
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
from tempfile import TemporaryDirectory
from unicodedata import normalize
import wave


MANIFEST_SHA = "3dcae81b890a72a73cb06d1fc25169b55e34b27434d391256cdf6307c7edd77b"
SHARDS = {
    5: "2d648a69b55d0788296e4e4a17609412981dd56d20c9abc32aa9317aafbe9da9",
    9: "1461def97dd549c6a2df7f4514d384c5aa0a317ead29c919017f397d3b429436",
    12: "eb64946713b716f7025cce1a2ce96cb72d6aade8ea804530cd37deebf3275d85",
    13: "4e5a6b83b442053e2ab0d120497c47db5231f83659c62dc6182e89c803f7d770",
    23: "f91dd45781f134055edae67a63157fab9d9d62088485770bb66f27923c87a329",
    36: "b4fa1264c3f115b27861aedf4d3634877ef8a4b55bbc56d082b99e43ade6b770",
    45: "e85b85ed05d9c0381d8a90032f4473e3013c0d2507d19b51fc5ba62826235ba5",
    59: "77f1ef42c298c6c7b729ff7ecca68a4d2e1fdbfee1ec88a01a5418a4ecf23668",
}
MODELS = {
    "small": "ae85e4a935d7a567bd102fe55afc16bb595bdb618e11b2fc7591bc08120411bb",
    "turbo": "394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2",
}
FLAGS = ("-l", "es", "-nf", "-mc", "0", "-bs", "5", "-bo", "5", "-tp", "0",
         "-t", "2", "-np", "-ml", "1", "-sow", "-oj")
SANDBOX = '(version 1) (allow default) (deny network*) (deny file-read* file-write* (regex #"^/dev/(cu|tty)"))'
WORDS = {"chiste": 4, "mantequilla": 4, "hora": 3, "dado": 3, "seis": 3, "no": 3}
BUTTER_ONLY_SHARDS = {9, 12, 13, 23}
SAMPLE_RATE = 16_000


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def normalized_words(text):
    return re.findall(r"[^\W\d_]+|\d+", normalize("NFKC", text).casefold())


def select(metadata, shard):
    count = Counter()
    selected = {}
    targets = {"mantequilla": WORDS["mantequilla"]} if shard in BUTTER_ONLY_SHARDS else WORDS
    for index in range(shard * 2000, (shard + 1) * 2000):
        item = metadata[index]
        tokens = normalized_words(item["sentence"])
        for word, limit in targets.items():
            if count[word] < limit and tokens.count(word) == 1:
                selected[item["original_file"]] = (index, word)
                count[word] += 1
                break
    return selected


def wave_bytes(path, pcm):
    with wave.open(str(path), "wb") as target:
        target.setparams((1, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
        target.writeframes(pcm)


def prepare(args, temp):
    path = args.source / "manifest.json"
    if digest(path) != MANIFEST_SHA:
        raise ValueError("upstream manifest SHA-256 mismatch")
    metadata = json.loads(path.read_text())
    rows = []
    for shard, checksum in SHARDS.items():
        path = args.source / f"shard-{shard}.tar"
        if digest(path) != checksum:
            raise ValueError(f"shard {shard} does not match its pinned digest")
        selected = select(metadata, shard)
        observed = set()
        with tarfile.open(path) as data:
            for item in data:
                if not item.isfile() or not item.name.endswith(".json"):
                    continue
                record = json.load(data.extractfile(item))
                target = selected.get(record["original_file"])
                if target is None:
                    continue
                index, word = target
                observed.add(record["original_file"])
                if any(record[key] != value for key, value in metadata[index].items()):
                    raise ValueError("shard metadata disagrees with the audit manifest")
                if record["license"] != "CC0-1.0" or record["accent_tag"] != "mexicano":
                    raise ValueError("word source license or dialect outside requested panel")
                member = data.getmember(item.name.removesuffix(".json") + ".mp3")
                if not member.isfile() or not 0 < member.size < 1_500_000:
                    raise ValueError("unexpected source audio")
                source = data.extractfile(member).read()
                pcm = subprocess.run(
                    [args.ffmpeg, "-nostdin", "-v", "error", "-i", "pipe:0",
                     "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "pipe:1"],
                    input=source, capture_output=True, check=True, timeout=20,
                ).stdout
                if not 0 < len(pcm) <= 30 * SAMPLE_RATE * 2 or len(pcm) % 2:
                    raise ValueError("source must be complete PCM16 bounded to 30 s")
                wav = temp / f"{index}.wav"
                wave_bytes(wav, pcm)
                rows.append((index, word, normalized_words(record["sentence"]), wav, sha256(source).hexdigest()))
        if observed != selected.keys():
            raise ValueError(f"shard {shard} is missing a selected utterance")
    if len(rows) != len({row[0] for row in rows}):
        raise ValueError("duplicate source utterance")
    return sorted(rows)


def decode(args, model, rows, target):
    target.mkdir(mode=0o700)
    inputs = []
    for index, _, _, wav, _ in rows:
        name = target / f"{index}.wav"
        shutil.copyfile(wav, name)
        inputs.append(name)
    subprocess.run(["/usr/bin/sandbox-exec", "-p", SANDBOX, args.whisper, "-m",
                    getattr(args, model), *FLAGS, *inputs], capture_output=True, check=True, timeout=240)
    return {index: json.loads((target / f"{index}.wav.json").read_text())["transcription"]
            for index, *_ in rows}


def interval(word, truth, segments):
    decoded = []
    for segment in segments:
        tokens = normalized_words(segment["text"])
        if len(tokens) > 1:
            return None
        if tokens:
            offsets = segment["offsets"]
            decoded.append((tokens[0], offsets["from"], offsets["to"]))
    labels = [row[0] for row in decoded]
    if labels.count(word) != 1:
        return None
    actual, expected = labels.index(word), truth.index(word)
    # Keep only matches with every available adjacent source token confirmed.
    for step in (-1, 1):
        if 0 <= expected + step < len(truth):
            if not 0 <= actual + step < len(labels) or labels[actual + step] != truth[expected + step]:
                return None
    return decoded[actual][1:]


def aligned(first, second):
    if first is None or second is None or any(abs(x - y) > 120 for x, y in zip(first, second)):
        return None
    start, end = min(first[0], second[0]), max(first[1], second[1])
    overlap = min(first[1], second[1]) - max(first[0], second[0])
    if start < 0 or not 90 <= end - start <= 900 or overlap * 2 < end - start:
        return None
    return start, end


def run(args):
    args.output.mkdir(mode=0o700)
    accepted = []
    counts = Counter()
    with TemporaryDirectory(prefix="bb-cv27-align-") as temp:
        work = Path(temp)
        rows = prepare(args, work)
        small = decode(args, "small", rows, work / "small")
        turbo = decode(args, "turbo", rows, work / "turbo")
        for index, word, truth, wav, source_hash in rows:
            counts[word] += 1
            first = interval(word, truth, small[index])
            second = interval(word, truth, turbo[index])
            bounds = aligned(first, second)
            if bounds is None:
                continue
            start, end = bounds
            with wave.open(str(wav), "rb") as source:
                if end * 16 > source.getnframes():
                    continue
                source.setpos(start * 16)
                pcm = source.readframes((end - start) * 16)
            path = args.output / f"anahuac-{index}.wav"
            wave_bytes(path, pcm)
            accepted.append({"id": f"anahuac-{index}", "word": word, "split": "development_only",
                             "locale_basis": "self_reported_mexicano", "speaker_verified": False,
                             "file": path.name, "wav_sha256": digest(path), "source_sha256": source_hash,
                             "interval_ms": bounds, "small_ms": first, "turbo_ms": second})
    report = {"dataset_commit": "2a9d611ee5c268fe8fd155adbb37eba790ef8e08",
              "license": "CC0-1.0", "source_manifest_sha256": MANIFEST_SHA,
              "decoder_sha256": digest(args.whisper), "models": MODELS, "flags": FLAGS,
              "attempted": counts, "accepted": Counter(row["word"] for row in accepted), "clips": accepted}
    (args.output / "alignment.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "clips"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output", "ffmpeg", "whisper", *MODELS):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    for key, expected in MODELS.items():
        if digest(getattr(args, key)) != expected:
            parser.error(f"wrong {key} model hash")
    if subprocess.check_output([args.whisper, "--version"], stderr=subprocess.STDOUT,
                               text=True, timeout=10).strip() != "whisper.cpp version: 1.9.6":
        parser.error("requires whisper.cpp 1.9.6")
    run(args)


if __name__ == "__main__":
    main()
