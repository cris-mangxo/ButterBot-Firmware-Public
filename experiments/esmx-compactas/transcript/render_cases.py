"""Render frozen synthetic utterances with pinned Piper Ald."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from ald_round import ALD_SHA256, canonical, digest
from d6_corpus import render_one

FIXTURE = Path(__file__).with_name("cases.json")
CONFIRM = FIXTURE.with_name("confirm.json")
FINAL = FIXTURE.with_name("final.json")
HOLDOUT = FIXTURE.with_name("holdout.json")
FIXTURES = {"panel": FIXTURE, "confirm": CONFIRM, "final": FINAL, "holdout": HOLDOUT}


def cases(previous_fixture, fixture=FIXTURE):
    previous = json.loads(previous_fixture.read_text("utf-8"))
    registered = {canonical(text) for text in previous["enroll"]["d6"]}
    rows = []
    seen = set()
    for earlier in FIXTURES.values():
        if earlier == fixture:
            break
        for lines in json.loads(earlier.read_text("utf-8")).values():
            seen.update(canonical(text) for text in lines)
    for category, lines in json.loads(fixture.read_text("utf-8")).items():
        for phrase in lines:
            normalized = canonical(phrase)
            if normalized in seen or (category == "d6" and normalized in registered):
                raise ValueError("Caso repetido o D6 ya registrado")
            seen.add(normalized)
            label = 6 if category == "d6" else 0
            rows.append({"id": f"asr-{len(rows):03d}", "category": category,
                         "label": label, "text": phrase, "speed": (0.94, 1.06)[len(rows) % 2]})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("piper-python", "ald", "out"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--suite", choices=FIXTURES, default="panel")
    args = parser.parse_args()
    if digest(args.ald) != ALD_SHA256:
        parser.error("Ald difiere del modelo fijado")
    fixture = FIXTURES[args.suite]
    rows = cases(FIXTURE.parent.parent / "phrases_cases.json", fixture)
    args.out.mkdir(mode=0o700)
    audio = args.out / "audio"
    audio.mkdir(mode=0o700)
    with ProcessPoolExecutor(max_workers=4) as pool:
        for _ in pool.map(render_one, ((args.piper_python, args.ald, audio, row) for row in rows)):
            pass
    (args.out / "manifest.json").write_text(json.dumps({
        "fixture_sha256": digest(fixture), "ald_sha256": ALD_SHA256, "samples": rows,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
