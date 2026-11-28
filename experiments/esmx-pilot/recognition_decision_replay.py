"""Compare the pilot's final labels and check the fallback policy on local fixtures."""

import argparse
from array import array
from collections import Counter
import json
from pathlib import Path

from capture_replay import pcm_from, recordings
from late_cancel_replay import as_pcm, samples, without_zeros
from time_lab import file_digest, invoke, supplemental


SELECTED = {
    "public-03", "public-37", "public-77", "public-81", "public-88", "public-89",
    "veto-fact", "veto-d6", "mty-003-02-15", "mty-042-04-15", "coerll-int02c-hora-esmx",
}


def compare(old, new, pcm):
    report = json.loads(invoke(new, pcm, "--decision"))
    before = int(invoke(old, pcm))
    if report["intent"] != before or int(invoke(new, pcm)) != before:
        raise ValueError("Pilot intent changed")
    eligible = report["intent"] == 0 and report["disposition"] == "no_match"
    if report["allows_fallback"] != eligible:
        raise ValueError("Fallback policy conflicts with pilot disposition")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("old", "new", "old-capture", "new-capture", "root", "coerll", "late-cancel"):
        parser.add_argument("--" + name, type=lambda path: Path(path).resolve(strict=True), required=True)
    args = parser.parse_args()
    panel = list(recordings(args.root))
    panel.append(supplemental(args.root, args.coerll))
    counts = {mode: Counter() for mode in ("raw", "retained")}
    examples = {}
    for name, _, _, target, path, checksum in panel:
        pcm = pcm_from(path, checksum)
        retained = invoke(args.new_capture, pcm, "--pcm-with-hangover")
        if retained != invoke(args.old_capture, pcm, "--pcm-with-hangover"):
            raise ValueError(f"Capture PCM changed for {name}")
        for mode, audio in (("raw", pcm), ("retained", retained)):
            observed = compare(args.old, args.new, audio)
            counts[mode][observed["disposition"]] += 1
            if name in SELECTED:
                examples.setdefault(name, {"target": target})[mode] = observed

    sources = args.root / "phrases-final" / "audio"
    affirmative = without_zeros(samples(pcm_from(sources / "enroll-0000.wav")))
    cancellation = without_zeros(samples(pcm_from(args.late_cancel)))
    late = affirmative + array("h", [0]) * (16 * 250) + array("h", (int(value / 16) for value in cancellation))
    extra = {
        "empty": (b"", "invalid_audio"),
        "silence_300ms": (bytes(9_600), "invalid_audio"),
        "short_pcm": (as_pcm(array("h", [12_000]) * 479), "invalid_audio"),
        "fact_veto": (pcm_from(sources / "enroll-0014.wav"), "fact_veto"),
        "fact_affirmative": (pcm_from(sources / "test-0298.wav"), "accepted"),
        "synthetic_d6_prefix": (as_pcm(affirmative), "accepted"),
        "synthetic_late_d6_cancel": (as_pcm(late), "dice_follow_up"),
    }
    cases = {}
    for name, (audio, disposition) in extra.items():
        cases[name] = compare(args.old, args.new, audio)
        if cases[name]["disposition"] != disposition or cases[name]["allows_fallback"]:
            raise ValueError(f"Unexpected disposition for {name}")
    for mode in counts:
        if examples["public-37"][mode]["disposition"] != "dice_negation_veto":
            raise ValueError("Public D6 cancellation did not retain its veto")
        if examples["public-77"][mode]["disposition"] != "no_match":
            raise ValueError("Neutral control did not reach the abstention path")

    print(json.dumps({
        "panel_cases": len(panel),
        "compared_decisions": 2 * len(panel),
        "capture_pcm_parity_cases": len(panel),
        "host_sha256": {key: file_digest(getattr(args, key)) for key in
                        ("old", "new", "old_capture", "new_capture")},
        "counts": counts,
        "examples": examples,
        "focused_controls": cases,
    }, indent=2))


if __name__ == "__main__":
    main()
