"""Render the versioned Ald D6 contrasts locally. Each split holds distinct constructions."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from ald_round import ALD_SHA256, canonical, digest, render_ald


def records(spec):
    candidates = []
    for pattern in spec["patterns"]:
        positive = pattern.format(n="seis")
        candidates.append(("d6", 6, positive))
        candidates.extend(("other_die", 0, pattern.format(n=number)) for number in spec["numbers"])
        phrase = positive[:1].lower() + positive[1:]
        candidates.extend(("negation", 0, negation.format(p=phrase)) for negation in spec["negations"])
    candidates.extend(("outside", 0, utterance) for utterance in spec["outside"])
    candidates.extend(("legacy", label, utterance) for label, utterance in spec.get("legacy", ()))
    return candidates


def render_one(job):
    python, ald, folder, row = job
    render_ald(python, ald, folder, row["id"], row["text"], row["speed"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("piper-python", "ald", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--anchor", action="store_true")
    args = parser.parse_args()
    if digest(args.ald) != ALD_SHA256:
        parser.error("Ald difiere del modelo fijado")
    fixture = Path(__file__).with_name("d6_anchor_cases.json" if args.anchor else "d6_cases.json")
    specifications = json.loads(fixture.read_text("utf-8"))
    if args.anchor:
        patterns = specifications["patterns"]
        specifications = {
            "train": {"patterns": patterns, "numbers": [], "negations": [], "outside": [],
                      "speeds": specifications["enroll_speeds"]},
            "dev": {**specifications["calibration"], "patterns": patterns},
            "test": {**specifications["test"], "patterns": patterns},
        }
    samples = {split: records(spec) for split, spec in specifications.items()}
    texts = {split: {canonical(utterance) for category, _, utterance in rows
                     if utterance and category != "legacy"}
             for split, rows in samples.items()}
    if args.anchor:
        # This experiment holds out recordings and speeds of the enrolled phrases.
        speeds = [speed for spec in specifications.values() for speed in spec["speeds"]]
        if len(speeds) != len(set(speeds)):
            parser.error("Las velocidades del verificador deben ser disjuntas")
    elif texts["train"] & texts["dev"] or texts["test"] & (texts["train"] | texts["dev"]):
        parser.error("Hay frases repetidas entre particiones Ald")
    args.out.mkdir(mode=0o700)
    folder = args.out / "audio"
    folder.mkdir(mode=0o700)
    manifest = []
    for split, rows in samples.items():
        for category, label, utterance in rows:
            # Legacy regression checks use exactly the original speed and text.
            speeds = [1.0] if category == "legacy" or utterance is None else specifications[split]["speeds"]
            for speed in speeds:
                manifest.append({
                    "id": f"{split}-{len(manifest):04d}", "split": split,
                    "category": category, "label": label, "text": utterance, "speed": speed,
                })
    with ProcessPoolExecutor(max_workers=4) as pool:
        jobs = ((args.piper_python, args.ald, folder, row) for row in manifest)
        for _ in pool.map(render_one, jobs):
            pass
    (args.out / "manifest.json").write_text(json.dumps({
        "fixture_sha256": digest(fixture), "ald_sha256": ALD_SHA256, "samples": manifest,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
