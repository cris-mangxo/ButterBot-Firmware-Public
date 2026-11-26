"""Render the phrase gate fixture with pinned Ald, entirely on the Mac."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from shutil import copyfile

from ald_round import ALD_SHA256, canonical, digest
from d6_corpus import render_one

FIXTURE = Path(__file__).with_name("phrases_cases.json")
LABELS = {"d6": 6, "fact": 4, "deny_fact": 0, "known_deny": 0, "outside": 0, "known_d6_deny": 0}


def samples(spec):
    for kind, label in LABELS.items():
        for phrase in spec.get(kind, ()):
            yield kind, label, phrase
    for label, phrase in spec.get("legacy", ()):
        yield "legacy", label, phrase
    for pattern in spec.get("contrasts", ()):
        for number in spec["numbers"]:
            yield "other_die", 0, pattern.format(n=number)
        positive = pattern.format(n="seis")
        phrase = positive[:1].lower() + positive[1:]
        for pattern in spec["negations"]:
            yield "deny_d6", 0, pattern.format(p=phrase)


def fixture_rows():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    rows = []
    development = set()
    previous_evaluation = set()
    for split, spec in fixture.items():
        for category, label, phrase in samples(spec):
            if phrase:
                normalized = canonical(phrase)
                if split in {"enroll", "calibrate"}:
                    development.add(normalized)
                elif split == "test":
                    if normalized in development and category not in {"known_deny", "legacy"}:
                        raise ValueError(f"Frase de prueba repetida: {phrase}")
                    previous_evaluation.add(normalized)
                elif split == "confirm" and (normalized in development or normalized in previous_evaluation):
                    raise ValueError(f"Frase de confirmación repetida: {phrase}")
            if category == "known_d6_deny":
                speeds = spec["known_d6_deny_speeds"]
            else:
                speeds = [1.0] if category == "legacy" or phrase is None else spec["speeds"]
            for speed in speeds:
                rows.append({"id": f"{split}-{len(rows):04d}", "split": split,
                             "category": category, "label": label, "text": phrase, "speed": speed})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("piper-python", "ald", "out"):
        parser.add_argument(f"--{argument}", type=Path, required=True)
    parser.add_argument("--reuse", type=Path, help="Reutiliza tomas Ald del laboratorio local")
    args = parser.parse_args()
    if digest(args.ald) != ALD_SHA256:
        parser.error("Ald difiere del modelo fijado")
    rows = fixture_rows()
    args.out.mkdir(mode=0o700)
    audio = args.out / "audio"
    audio.mkdir(mode=0o700)
    cached = {}
    if args.reuse:
        prior = json.loads((args.reuse / "manifest.json").read_text("utf-8"))
        if prior["ald_sha256"] != ALD_SHA256:
            parser.error("Las tomas existentes tienen otro modelo Ald")
        for row in prior["samples"]:
            basename = Path(row["id"] + ".wav")
            if basename.name != str(basename):
                parser.error("Nombre de toma Ald inválido")
            cached[(row["text"], row["speed"])] = args.reuse / "audio" / basename
    pending = []
    for row in rows:
        source = cached.get((row["text"], row["speed"]))
        if source is not None:
            copyfile(source, audio / (row["id"] + ".wav"))
        else:
            pending.append((args.piper_python, args.ald, audio, row))
    with ProcessPoolExecutor(max_workers=4) as pool:
        for _ in pool.map(render_one, pending):
            pass
    (args.out / "manifest.json").write_text(json.dumps({
        "fixture_sha256": digest(FIXTURE), "ald_sha256": ALD_SHA256, "samples": rows,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
