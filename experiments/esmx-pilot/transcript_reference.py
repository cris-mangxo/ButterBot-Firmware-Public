"""Measure an offline transcript-and-dispatch reference without keeping decoded text."""

import argparse
from hashlib import file_digest, sha256
import json
from pathlib import Path
import subprocess
import re
import sys
from tempfile import TemporaryDirectory
from unicodedata import normalize
import wave

from capture_replay import DISPATCH, evaluate as audio_decision, pcm_from, recordings

COMPACT = Path(__file__).resolve().parents[1] / "esmx-compactas"
sys.path.extend((str(COMPACT), str(COMPACT / "transcript")))
from evaluate import MODELS, WHISPER_FLAGS, decode_batch

TURBO = ("394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2", 574041195)
DECODE_FLAGS = (*WHISPER_FLAGS, "-bs", "5", "-bo", "5", "-tp", "0", "-mc", "0")


def digest(path):
    with path.open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


def words(text):
    return re.findall(r"[a-z0-9]+", normalize("NFKD", text.casefold())
                      .encode("ascii", errors="ignore").decode("ascii"))



def reference(executable, text):
    data = text.encode("utf-8")
    if len(data) > 192:
        data = b""
    result = subprocess.run([executable], input=data, capture_output=True, check=True, timeout=3)
    result = json.loads(result.stdout)
    if result["route"] != DISPATCH.get(result["intent"], "none"):
        raise ValueError("text reference selected an inconsistent factory")
    return result


def check_grammar(executable):
    fixture = (
        ("¿Qué horas son?", 5), ("Me puede decir qué hora es.", 5),
        ("Échame un chiste.", 3), ("Pásame la mantequilla.", 1),
        ("Tira un dado de seis, por favor.", 6), ("Tira un dado de 6.", 6),
        ("No me pases la mantequilla.", 0), ("No me cuentes un chiste.", 0),
        ("No me digas qué horas son.", 0), ("¿A qué hora empieza?", 0),
        ("Tira un dado de seis. No.", 0), ("No tires un dado de seis.", 0),
        ("Tira dos dados de seis.", 0), ("Tira un dado de ocho.", 0),
        ("Dime un dato curioso.", 0), ("No me digas un dato curioso.", 0),
    )
    if any(reference(executable, text)["intent"] != expected for text, expected in fixture):
        raise ValueError("reference grammar fixture failed")
    return len(fixture)


def selected(base):
    yield from (
        row for row in recordings(base)
        if row[1].startswith("public") or row[1] == "synthetic_aliases" or row[0] == "test-0379"
    )
    voices = base / "primer-candidato-voces"
    extra = json.loads((voices / "manifest-coerll-int02c-corte.json").read_text())
    reserved = (voices / extra["reserved_manifest"]).resolve(strict=True)
    wav = (voices / extra["sample"]).resolve(strict=True)
    if (reserved.parent != voices.resolve() or wav.parent != voices.resolve()
            or sha256(reserved.read_bytes()).hexdigest() != extra["reserved_manifest_sha256"]
            or extra["split"] != "heldout_supplemental" or extra["positive_class"] != "TimeNow"):
        raise ValueError("supplemental reserve differs from its sealed record")
    yield extra["sample_id"], extra["split"], "time", 5, wav, extra["wav_sha256"]


def known_texts(base):
    voices = base / "primer-candidato-voces"
    corpus = json.loads((voices / "manifest.json").read_text())
    supplementary = json.loads((voices / "manifest-coerll-int02c-corte.json").read_text())
    result = {row["id"]: row["text"] for row in corpus["clips"]}
    result[supplementary["sample_id"]] = supplementary["transcript"]
    return result


def capture_segment(program, samples, mode, work, index):
    flags = ("--nominal-hangover",) if mode == "hold" else ()
    result = audio_decision(program, samples, *flags)
    row = {key: result[key] for key in ("observed", "route", "reason")}
    if result["reason"] != "classified":
        return row, None
    command = "--pcm-with-hangover" if mode == "hold" else "--pcm"
    capture = subprocess.run([program, command], input=samples, capture_output=True,
                             check=True, timeout=10).stdout
    if not capture or len(capture) % 2 or len(capture) > 128_000:
        raise ValueError("unexpected PCM window from the capture probe")
    row["pcm_sha256"] = sha256(capture).hexdigest()
    row["samples"] = len(capture) // 2
    filename = work / f"capture-{index:03}-{mode}.wav"
    with wave.open(str(filename), "wb") as target:
        target.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        target.writeframes(capture)
    return row, filename


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("whisper", "small", "turbo", "reference", "recognizer", "capture", "experiment-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    version = subprocess.check_output([args.whisper, "--version"], stderr=subprocess.STDOUT,
                                      text=True, timeout=5).strip()
    if version != "whisper.cpp version: 1.9.6":
        parser.error("reference requires whisper.cpp 1.9.6")
    profiles = {"small": MODELS["small"], "turbo": TURBO}
    for key, (checksum, size) in profiles.items():
        weights = getattr(args, key)
        if weights.stat().st_size != size or digest(weights) != checksum:
            parser.error(f"model {key} differs from the pinned weights")
    grammar_checks = check_grammar(args.reference)
    clips = list(selected(args.experiment_root))
    decisions = []
    annotated = known_texts(args.experiment_root)
    elapsed = {}
    with TemporaryDirectory(prefix="bb-esmx-reference-") as scratch:
        work = Path(scratch)
        queue = []
        for index, (identifier, split, category, expected, wav, checksum) in enumerate(clips):
            samples = pcm_from(wav, checksum)
            raw = {"compact": audio_decision(args.recognizer, samples), "ref": {}}
            record = {"id": identifier, "split": split, "category": category,
                      "expected": expected, "wav_sha256": digest(wav), "raw": raw}
            decisions.append(record)
            gold = annotated.get(identifier)
            if gold is not None:
                record["annotation_grammar"] = reference(args.reference, gold)
            queue.append((raw, wav, gold))
            if "reserved" in split or split == "heldout_supplemental":
                for mode in ("immediate", "hold"):
                    found, path = capture_segment(args.capture, samples, mode, work, index)
                    record[mode] = found
                    if path is not None:
                        found["ref"] = {}
                        queue.append((found, path, gold))

        for key in profiles:
            texts, elapsed[key] = decode_batch(args.whisper, getattr(args, key),
                                               [path for _, path, _ in queue], work / key, DECODE_FLAGS)
            for (result, _, annotated_text), text in zip(queue, texts, strict=True):
                decision = reference(args.reference, text)
                if annotated_text is not None:
                    decision["annotation_words_match"] = words(annotated_text) == words(text)
                result["ref"][key] = decision
    print(json.dumps({
        "decoder": version, "decoder_flags": DECODE_FLAGS,
        "reference_executable_sha256": digest(args.reference),
        "model_pins": profiles, "grammar_checks": grammar_checks,
        "elapsed_seconds": {key: round(value, 3) for key, value in elapsed.items()},
        "rows": decisions,
    }, indent=2))


if __name__ == "__main__":
    main()
