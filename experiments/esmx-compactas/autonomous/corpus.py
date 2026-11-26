"""Render reproducible, paired Ald D6 contrasts for a tiny acoustic model."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import io
import json
from pathlib import Path
import re
import wave

import numpy as np
from piper import PiperVoice, SynthesisConfig

from ald_round import ALD_SHA256, canonical, digest
from model_sources import SHARVARD_CONFIG_SHA256, SHARVARD_SHA256

FIXTURE = Path(__file__).with_name("fixture.json")
FRESH = FIXTURE.with_name("fresh_suite.json")
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


def fresh_rows(previous):
    old = {canonical(row["text"]) for row in previous}
    items = json.loads(FRESH.read_text("utf-8"))
    index = 0
    for category, texts in items.items():
        for text in texts:
            key = canonical(text)
            if key in old:
                raise ValueError("Frase repetida en la prueba nueva")
            old.add(key)
            index += 1
            yield {"id": f"test-{index:04d}-0", "split": "test", "label": CLASSES[category],
                   "category": category, "speed": (0.95, 1.05)[index % 2], "text": text}


def shorthand_rows(previous):
    for row in previous:
        if row["split"] == "test" or row["category"] == "outside":
            continue
        text, changes = re.subn(r"\bun dado de\b", "un de", row["text"], count=1)
        if changes:
            yield dict(row, text=text)


def waveform(task):
    output, row, aligned, speaker = task
    raw = io.BytesIO()
    synthesis = SynthesisConfig(speaker_id=speaker, length_scale=row["speed"], noise_scale=0, noise_w_scale=0)
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
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--ald", type=Path)
    sources.add_argument("--sharvard", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--aligned", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="Genera solo el nuevo panel reservado")
    parser.add_argument("--shorthand", action="store_true", help="Genera contrastes sintéticos para «un de seis»")
    args = parser.parse_args()
    model = args.ald or args.sharvard
    expected = ALD_SHA256 if args.ald else SHARVARD_SHA256
    if digest(model) != expected:
        parser.error("La voz de síntesis no coincide con la versión fijada")
    if args.sharvard and digest(Path(str(model) + ".json")) != SHARVARD_CONFIG_SHA256:
        parser.error("La configuración Sharvard difiere de la referencia")
    if args.fresh and not args.ald:
        parser.error("El panel reservado usa Ald")
    if args.fresh and args.shorthand:
        parser.error("La variante abreviada se reserva al desarrollo")
    corpus = list(rows())
    if args.fresh:
        corpus = list(fresh_rows(corpus))
    if args.shorthand:
        corpus = list(shorthand_rows(corpus))
    if args.sharvard:
        corpus = [row for row in corpus if row["split"] != "test"]
    args.out.mkdir(mode=0o700)
    audio = args.out / "audio"
    audio.mkdir(mode=0o700)
    def speaker(row):
        if args.ald:
            return None
        _, sequence, replica = row["id"].split("-")
        return (int(sequence) + int(replica)) % 2

    with ProcessPoolExecutor(max_workers=3, initializer=initialize, initargs=(model, args.aligned)) as pool:
        for _ in pool.map(waveform, ((audio, row, args.aligned, speaker(row)) for row in corpus), chunksize=8):
            pass
    (args.out / "manifest.json").write_text(json.dumps({
        "fixture_sha256": digest(FRESH if args.fresh else FIXTURE),
        "ald_sha256" if args.ald else "sharvard_sha256": expected,
        "samples": corpus,
        **({"shorthand": True} if args.shorthand else {}),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
