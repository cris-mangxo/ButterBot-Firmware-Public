"""Render reproducible, paired Ald D6 contrasts for a tiny acoustic model."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import io
import json
from pathlib import Path
import wave

import numpy as np
from piper import PiperVoice, SynthesisConfig

from ald_round import ALD_SHA256, canonical, digest

FIXTURE = Path(__file__).with_name("fixture.json")
CLASSES = {"outside": 0, "positive": 1, "other_die": 2, "negated": 3}
NUMBERS = ("cuatro", "ocho", "diez", "doce", "veinte", "cien", "siete",
           "dieciséis", "tres", "dos", "cinco", "sesenta")
RATES = {"train": (0.89, 1.11), "dev": (0.93, 1.07), "test": (0.97, 1.03)}
VOICE = None


def development(spec):
    for i, stem in enumerate(spec["stems"]):
        for j, tail in enumerate(spec["tails"]):
            for k, politeness in enumerate(spec["polite_tails"]):
                template = f"{stem} de {{number}}{tail}{politeness}"
                request = template.format(number="seis")
                yield "positive", request + ("?" if request.startswith("¿") else ".")
                for offset in (0, 3, 7):
                    number = NUMBERS[(5 * i + 3 * j + k + offset) % len(NUMBERS)]
                    yield "other_die", template.format(number=number) + ("?" if stem.startswith("¿") else ".")
                lower = request.lstrip("¿")
                lower = lower[:1].lower() + lower[1:]
                for offset in (0, 3):
                    negation = spec["negated"][(i + j + k + offset) % len(spec["negated"])]
                    yield "negated", negation.format(request=lower) + "."
    yield from (("outside", text) for text in spec["outside"])


def rows():
    spec = json.loads(FIXTURE.read_text("utf-8"))
    participants = {split: set() for split in RATES}
    for split in RATES:
        if split == "test":
            cases = ((category, text) for category, texts in spec[split].items() for text in texts)
        else:
            cases = development(spec[split])
        unique = {}
        for category, text in cases:
            normalized = canonical(text)
            other = unique.setdefault(normalized, category)
            if other != category:
                raise ValueError("Una frase tiene dos etiquetas")
            if normalized in participants[split]:
                continue
            participants[split].add(normalized)
            index = len(participants[split])
            rate = RATES[split]
            speeds = rate if split != "test" else (rate[index % len(rate)],)
            for i, speed in enumerate(speeds):
                yield {"id": f"{split}-{index:04d}-{i}", "split": split, "label": CLASSES[category],
                       "category": category, "speed": speed, "text": text}
    if (participants["train"] & participants["dev"] or
            participants["test"] & (participants["train"] | participants["dev"])):
        raise ValueError("Se repitió el texto entre particiones")


def initialize(model, aligned):
    global VOICE
    VOICE = PiperVoice.load(model, include_alignments=aligned)


def waveform(task):
    output, row, aligned = task
    raw = io.BytesIO()
    synthesis = SynthesisConfig(length_scale=row["speed"], noise_scale=0, noise_w_scale=0)
    with wave.open(raw, "wb") as original:
        intervals = VOICE.synthesize_wav(row["text"], original, synthesis, include_alignments=aligned)
    raw.seek(0)
    with wave.open(raw, "rb") as source:
        rate = source.getframerate()
        pcm = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
    if aligned:
        if not intervals:
            raise ValueError("Ald no produjo alineaciones de entrenamiento")
        (output / (row["id"] + ".json")).write_text(json.dumps({
            "sample_rate": rate, "phones": [[entry.phoneme, int(entry.num_samples)] for entry in intervals],
        }, ensure_ascii=False), encoding="utf-8")
    # Windowed low pass before linear resampling; the reference frontend always
    # receives the same mono 16 kHz format used by the embedded path.
    if rate != 16000:
        kernel = np.sinc(np.arange(-15, 16) * (15000 / rate)) * np.hamming(31)
        clean = np.convolve(pcm, kernel / kernel.sum(), mode="same")
        grid = np.arange(round(len(pcm) * 16000 / rate)) * (rate / 16000)
        pcm = np.rint(np.interp(grid, np.arange(len(pcm)), clean)).astype("<i2")
    with wave.open(str(output / (row["id"] + ".wav")), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(pcm.tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ald", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--aligned", action="store_true")
    args = parser.parse_args()
    if digest(args.ald) != ALD_SHA256:
        parser.error("Ald no coincide con la versión fijada")
    corpus = list(rows())
    args.out.mkdir(mode=0o700)
    audio = args.out / "audio"
    audio.mkdir(mode=0o700)
    with ProcessPoolExecutor(max_workers=3, initializer=initialize, initargs=(args.ald, args.aligned)) as pool:
        for _ in pool.map(waveform, ((audio, row, args.aligned) for row in corpus), chunksize=8):
            pass
    (args.out / "manifest.json").write_text(json.dumps({
        "fixture_sha256": digest(FIXTURE), "ald_sha256": ALD_SHA256, "samples": corpus,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
