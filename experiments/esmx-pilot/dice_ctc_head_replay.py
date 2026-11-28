"""Replay the isolated D6 phoneme head without changing the integrated recognizer."""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json

import duckdb
import numpy as np
import torch

from capture_replay import pcm_from, recordings
from dice_ctc_head_lab import accepted, encode, Grammar, head_like, metrics
from d6_teacher_guard_lab import pcm_seals, eligible_pcm
from phoneme_chm150_lab import CORPUS_SHA256, SPEAKER_SPLIT_SHA256, decode, eligible_rows
from phoneme_replay_lab import frozen_model, native_frames, strip_digital_edges
from time_lab import file_digest, invoke


def heads(model, frozen, directory):
    result = {}
    names = {"frozen", "regularized", "contrast", "human_8", "human_24"}
    if set(frozen["rules"]) != names:
        raise ValueError("Unexpected head variants")
    for name, spec in frozen["rules"].items():
        head = head_like(model)
        if name != "frozen":
            artifact = directory / f"{name}.npz"
            if file_digest(artifact) != spec["head_sha256"]:
                raise ValueError("Head does not match its recipe")
            with np.load(artifact, allow_pickle=False) as values:
                integer, scales, bias = (values[key] for key in ("weights", "scales", "bias"))
            if (integer.shape != tuple(head.weight.shape) or integer.dtype != np.int8
                    or scales.shape != tuple(head.bias.shape) or bias.shape != tuple(head.bias.shape)
                    or scales.dtype != np.float32 or bias.dtype != np.float32
                    or not np.isfinite(scales).all() or not np.isfinite(bias).all() or (scales <= 0).any()):
                raise ValueError("Invalid int8 head geometry")
            with torch.no_grad():
                head.weight.copy_(torch.from_numpy(integer).float() * torch.from_numpy(scales)[:, None])
                head.bias.copy_(torch.from_numpy(bias))
        result[name] = (head.eval(), spec["threshold"])
    return result


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("recipe", "bank", "pilot", "capture", "root", "frontend", "chm", "ffmpeg", "speech", "labelled"):
        parser.add_argument("--" + name, type=Path, required=True)
    programs = ("pilot", "capture", "ffmpeg")
    for name in programs:
        parser.add_argument(f"--{name}-sha256", required=True)
    args = parser.parse_args()
    hashes = {name: file_digest(getattr(args, name)) for name in programs}
    if any(value != getattr(args, key + "_sha256") for key, value in hashes.items()):
        raise ValueError("Replay executable fingerprint differs")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    frozen = json.loads(args.recipe.read_text())
    model, meta = frozen_model(args.bank)
    refs = {key: file_digest(path) for key, path in {
        "speech": args.speech / "index.json", "labelled": args.labelled / "index.json",
        "chm": args.chm / "native-negative-features/index.json",
        "engine": args.frontend, "speaker_split": args.chm / "speaker-roles.json",
        "synthesis": args.recipe.parent / "synthesis.json",
    }.items()}
    if any(frozen["pins"][key] != digest for key, digest in refs.items()) or (
            frozen["bank_weights"] != meta["weights_sha256"] or
            frozen["source_sha256"] != file_digest(Path(__file__).with_name("dice_ctc_head_lab.py"))):
        raise ValueError("Unexpected replay provenance")
    occupied = {row["pcm_sha256"] for row in json.loads((args.speech / "index.json").read_text())["records"].values()}
    for row in json.loads((args.labelled / "index.json").read_text())["records"].values():
        occupied.update(row["pcm_sha256"])
    occupied.update(row["effective_pcm_sha256"] for row in
                    json.loads((args.chm / "native-negative-features/index.json").read_text())["records"].values())
    for row in json.loads((args.recipe.parent / "synthesis.json").read_text()):
        if row["split"] != "test":
            seal = pcm_from(args.recipe.parent / f"{row['id']}.wav", row["sha256"])
            occupied.update(pcm_seals(seal))
    grammar = Grammar(*frozen["paths"])
    proposals = heads(model, frozen, args.recipe.parent)

    def observe(pcm):
        base = int(invoke(args.pilot, pcm))
        decision = json.loads(invoke(args.pilot, pcm, "--decision"))
        data = encode(model, native_frames(args.frontend, strip_digital_edges(pcm)))
        values = {}
        for name, (head, threshold) in proposals.items():
            measured = metrics(head, grammar, data)
            possible = accepted(measured, threshold)
            strict = (possible and decision["allows_fallback"] and decision["intent"] == 0
                      and decision["disposition"] == "no_match")
            conflict = possible and decision["intent"] == 0 and decision["disposition"] == "dice_negation_conflict"
            values[name] = {"acoustic": measured, "strict": base or (6 if strict else 0),
                            "conflict_lab": base or (6 if strict or conflict else 0)}
        return {"prior": base, "primary_disposition": decision["disposition"], "variants": values}

    cases, exclusions = {}, []
    for identifier, split, _, expected, wav, checksum in recordings(args.root):
        pcm = pcm_from(wav, checksum)
        views = {"raw": pcm, "retained": invoke(args.capture, pcm, "--pcm-with-hangover")}
        if any(set(pcm_seals(data)) & occupied for data in views.values()):
            if split.startswith("public_reserved_"):
                raise ValueError("A D6 reservation overlaps adjustment")
            exclusions.append(identifier)
            continue
        cases[identifier] = {"expected": expected, "split": split,
                             **{name: observe(payload) for name, payload in views.items()}}
    corpus = args.chm / "corpus.parquet"
    manifest = args.chm / "speaker-roles.json"
    if file_digest(corpus) != CORPUS_SHA256 or file_digest(manifest) != SPEAKER_SPLIT_SHA256:
        raise ValueError("Unexpected conversation sources")
    reserved = {key for key, role in json.loads(manifest.read_text())["speaker_roles"].items() if role == "heldout"}
    additions, counts, speakers = Counter(), Counter(), set()
    pcm_digest = sha256()
    with duckdb.connect(config={"autoload_known_extensions": False, "autoinstall_known_extensions": False}) as rows:
        for _, speaker, _, clip in eligible_rows(rows, corpus, reserved):
            pcm = decode(args.ffmpeg, clip)
            if not eligible_pcm(pcm):
                continue
            if occupied.intersection(pcm_seals(pcm)):
                raise ValueError("Conversation overlaps adjustment")
            sample = observe(pcm)
            counts[sample["prior"]] += 1
            speakers.add(speaker)
            pcm_digest.update(len(pcm).to_bytes(4, "little"))
            pcm_digest.update(pcm)
            for name, arm in sample["variants"].items():
                for gate in ("strict", "conflict_lab"):
                    additions[f"{name}:{gate}"] += int(arm[gate] != sample["prior"])
    print(json.dumps({"recipe_sha256": file_digest(args.recipe), "cases": cases, "excluded_calibration": exclusions,
                      "executable_sha256": hashes,
                      "conversation": {"cases": sum(counts.values()), "speakers": len(speakers),
                                       "pcm_sha256": pcm_digest.hexdigest(), "prior_labels": counts,
                                       "added": dict(additions)}}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
