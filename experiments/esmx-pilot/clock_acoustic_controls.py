"""Check native behavior on sealed public conversation and synthetic whole turns."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import duckdb

from capture_replay import pcm_from
from phoneme_chm150_lab import CORPUS_SHA256, SPEAKER_SPLIT_SHA256, eligible_rows, decode
from time_lab import file_digest, invoke


def main():
    parser = argparse.ArgumentParser()
    for arg in ("before","after","chm","speakers","ffmpeg","synthetic"):
        parser.add_argument("--" + arg, type=Path, required=True)
    a = parser.parse_args()
    if file_digest(a.chm) != CORPUS_SHA256 or file_digest(a.speakers) != SPEAKER_SPLIT_SHA256:
        raise ValueError("Unexpected conversation partition")
    roles = json.loads(a.speakers.read_text())["speaker_roles"]
    groups = {"conversation":[], "synthetic_clock_near_pairs":[]}
    with duckdb.connect(config={"autoload_known_extensions": False, "autoinstall_known_extensions": False}) as db:
        for i, (_, _, _, audio) in enumerate(eligible_rows(db, a.chm, {s for s, r in roles.items() if r == "heldout"})):
            groups["conversation"].append((i, False, audio))
    for row in json.loads((a.synthetic / "manifest.json").read_text()):
        if row["split"] != "train":
            pcm = pcm_from(a.synthetic / (row["id"] + ".wav"), row["sha256"])
            groups["synthetic_clock_near_pairs"].append((row["id"], row["goal"] == 1, pcm))
    output = {}
    with ThreadPoolExecutor(max_workers=7) as pool:
        for name, inputs in groups.items():
            def one(entry):
                key, wanted, data = entry
                sample = decode(a.ffmpeg, data) if name == "conversation" else data
                if not 0 < len(sample) <= 128_000:
                    return None
                old = int(invoke(a.before, sample))
                new = int(invoke(a.after, sample))
                return key, wanted, old, new
            counts = Counter()
            added = []
            for row in pool.map(one, inputs):
                if row is None:
                    continue
                key, wanted, old, new = row
                counts["turns"] += 1
                counts["negatives" if not wanted else "positive"] += 1
                if new != old:
                    added.append({"case":key,"expected_clock":wanted,"before":old,"after":new})
            output[name] = {"counts": dict(counts), "changes":added}
    print(json.dumps(output, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
