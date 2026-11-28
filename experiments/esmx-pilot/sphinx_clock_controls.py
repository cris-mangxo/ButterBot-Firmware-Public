"""Evaluate the exact clock path of the local acoustic diagnostic on known controls."""
import argparse
import json
from pathlib import Path
from collections import Counter

import duckdb
from pocketsphinx import Decoder

from sphinx_clock_reference import GRAMMAR
from capture_replay import pcm_from, recordings
from phoneme_chm150_lab import CORPUS_SHA256, SPEAKER_SPLIT_SHA256, decode, eligible_rows
from time_lab import file_digest, invoke

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("acoustic","dict","chm","speakers","ffmpeg","host","history"):
        parser.add_argument("--" + key, type=Path, required=True)
    a = parser.parse_args()
    if file_digest(a.chm) != CORPUS_SHA256 or file_digest(a.speakers) != SPEAKER_SPLIT_SHA256:
        raise ValueError("Conversation corpus or speaker split changed")
    decoder = Decoder(hmm=str(a.acoustic), dict=str(a.dict), lm=None, samprate=16000, loglevel="FATAL")
    decoder.add_jsgf_string("clock", GRAMMAR)
    decoder.activate_search("clock")
    outcomes = {}
    for panel in ("historical", "conversation"):
        counts = Counter()
        witnesses = []
        def assess(key, pcm, expected):
            if not 0 < len(pcm) <= 128_000:
                return
            decoder.start_utt()
            decoder.process_raw(pcm, False, True)
            decoder.end_utt()
            hypothesis = decoder.hyp()
            detail = json.loads(invoke(a.host, pcm, "--decision"))
            match = hypothesis is not None and hypothesis.hypstr == "qué hora es"
            counts["evaluated"] += 1
            counts["target_expected"] += expected == 5
            counts["raw_exact_match"] += match
            counts["eligible_after_pilot"] += detail["allows_fallback"]
            if match and detail["allows_fallback"]:
                counts["new_clock"] += 1
                if expected != 5:
                    witnesses.append({"case_index": key, "prob": hypothesis.prob, "score": hypothesis.best_score})
        if panel == "historical":
            for key, role, _, expected, path, pin in recordings(a.history):
                assess(key, pcm_from(path, pin), expected)
        else:
            roles = json.loads(a.speakers.read_text())["speaker_roles"]
            with duckdb.connect(config={"autoload_known_extensions": False,
                                        "autoinstall_known_extensions": False}) as db:
                for index, (_, _, _, audio) in enumerate(eligible_rows(db, a.chm, {s for s, r in roles.items() if r == "heldout"})):
                    assess(index, decode(a.ffmpeg, audio), 0)
        outcomes[panel] = {"counts":dict(counts), "new_wrong": witnesses}
    print(json.dumps({"host_sha256": file_digest(a.host), "corpus_sha256": CORPUS_SHA256,
                      "speaker_split_sha256": SPEAKER_SPLIT_SHA256, "panels": outcomes},
                     indent=2, allow_nan=False))
