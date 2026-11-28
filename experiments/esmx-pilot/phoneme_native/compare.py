"""Compare native logits and gate decisions against the reconstructed frozen Student."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capture_replay import pcm_from, recordings
from phoneme_replay_lab import frozen_model, native_frames
from phoneme_student_lab import gate
from keyword_lab import LABELS
from time_lab import file_digest, invoke, supplemental

BANDS, PHONES, INTENTS, MAX_FRAMES = 20, 32, 5, 199
RTOL, ATOL = 1e-5, 2e-4


def synthetic_cases():
    rng = np.random.default_rng(20261127)
    for count in (1, 7, 8, 9, 33, MAX_FRAMES):
        yield f"zeros-{count}", np.zeros((count, BANDS), dtype=np.int8)
        yield f"range-{count}", rng.integers(-128, 128, dtype=np.int8, size=(count, BANDS))
    yield "positive-extreme", np.full((MAX_FRAMES, BANDS), 127, dtype=np.int8)
    yield "negative-extreme", np.full((MAX_FRAMES, BANDS), -128, dtype=np.int8)


def cache_cases(folder):
    if folder is None:
        return
    index = json.loads((folder / "index.json").read_text())
    if index["schema"] != "bb-phoneme-development-1" or index["split_origin"] != "synthetic_mac_voices_and_ald":
        raise ValueError("Expected sealed synthetic-only lab cache")
    found = set()
    for name, row in sorted(index["records"].items()):
        if row["split"] != "dev" or row["intent"] in found:
            continue
        if not name.startswith("dev-") or any(not char.isalnum() and char != "-" for char in name):
            raise ValueError("Invalid synthetic sample ID")
        path = folder / f"{name}.npz"
        if sha256(path.read_bytes()).hexdigest() != row["artifact_sha256"]:
            raise ValueError("Synthetic feature cache integrity mismatch")
        with np.load(path, allow_pickle=False) as sample:
            frames = sample["features"]
        if (frames.dtype != np.int8 or frames.shape != (row["frames"], BANDS)
                or not 1 <= len(frames) <= MAX_FRAMES):
            raise ValueError("Invalid synthetic feature shape")
        found.add(row["intent"])
        yield name, frames
    if found != set(LABELS):
        raise ValueError("Missing synthetic diagnostic intention")


@torch.inference_mode()
def oracle(model, frames):
    data = torch.from_numpy(frames.T[None].astype(np.float32) / 32)
    phones, intents, denied = model(data, torch.tensor([len(frames)]))
    scores = np.concatenate([phones.flatten().numpy(), intents.flatten().numpy(), denied.numpy()])
    return LABELS[int(gate(intents, denied).item())], scores


def run(executable, model, name, frames):
    binary = subprocess.run([executable, "--logits"], input=frames.tobytes(),
                            stdout=subprocess.PIPE, check=True, timeout=10).stdout
    elements = len(frames) * PHONES + INTENTS + 1
    if len(binary) != elements * 4 + 1:
        raise ValueError(f"Unexpected native output length for {name}")
    native = np.frombuffer(binary[:-1], dtype="<f4")
    expected, reference = oracle(model, frames)
    np.testing.assert_allclose(native, reference, rtol=RTOL, atol=ATOL, err_msg=name)
    if binary[-1] != expected:
        raise ValueError(f"Gate mismatch {name}: native {binary[-1]}, expected {expected}")
    return {"case": name, "frames": len(frames), "intent": binary[-1], "expected_intent": expected,
            "max_abs_logit_error": float(np.abs(native - reference).max())}


def public_replay(args, executable, model):
    rows = [row for row in recordings(args.public_root)
            if row[1].startswith("public") and (args.include_past_reserves or "_reserved_" not in row[1])]
    if args.coerll:
        rows.append(supplemental(args.public_root, args.coerll.resolve(strict=True)))
    result = []
    panel_seal = sha256()
    for name, split, category, target, path, checksum in rows:
        pcm = pcm_from(path, checksum)
        panel_seal.update(json.dumps([name, split, category, target, sha256(pcm).hexdigest()]).encode("utf-8"))
        captured = invoke(args.capture, pcm, "--pcm-with-hangover")
        for mode, audio in (("raw", pcm), ("retained", captured)):
            frames = native_frames(args.frontend, audio)
            output = {"case": name, "frames": 0, "intent": None, "oracle_intent": None}
            if frames is not None:
                output = run(executable, model, name, frames)
                output["oracle_intent"] = output.pop("expected_intent")
            pilot = None
            if 0 < len(audio) <= 128_000:  # PCM16 in bytes, four seconds at 16 kHz.
                label = invoke(args.pilot, audio)
                if label:
                    pilot = int(label)
            result.append({**output, "mode": mode, "cohort": split, "category": category,
                           "pilot_intent": pilot, "target_intent": target})
    return {
        "scope": "historical_public_diagnostics",
        "past_reserves_included": args.include_past_reserves,
        "panel_sha256": panel_seal.hexdigest(),
        "supplemental_manifest_sha256": file_digest(args.coerll) if args.coerll else None,
        "frontend_sha256": file_digest(args.frontend),
        "capture_sha256": file_digest(args.capture),
        "pilot_sha256": file_digest(args.pilot),
        "student_sha256": file_digest(executable),
        "rows": result,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--public-root", type=Path)
    parser.add_argument("--frontend", type=Path)
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--coerll", type=Path)
    parser.add_argument("--include-past-reserves", action="store_true",
                        help="Include previously inspected reserved rows for historical regression only")
    args = parser.parse_args()
    if args.public_root:
        if not (args.frontend and args.capture and args.pilot):
            parser.error("--public-root requires --frontend, --capture and --pilot")
    elif args.frontend or args.capture or args.pilot or args.coerll or args.include_past_reserves:
        parser.error("public replay options require --public-root")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model, report = frozen_model(args.checkpoint)
    executable = args.executable.resolve(strict=True)
    results = [run(executable, model, name, frames)
               for name, frames in [*synthetic_cases(), *cache_cases(args.cache)]]
    for malformed in (b"", b"abc", bytes((MAX_FRAMES + 1) * BANDS)):
        result = subprocess.run([executable], input=malformed, capture_output=True, timeout=10)
        if result.returncode != 2 or result.stdout:
            raise ValueError("Invalid frame geometry was not rejected cleanly")
    summary = {
        "weight_sha256": report["weights_sha256"], "cases": len(results),
        "gate_parity": len(results), "malformed_cases_rejected": 3, "atol": ATOL, "rtol": RTOL,
        "max_abs_logit_error": max(row["max_abs_logit_error"] for row in results),
        "rows": results,
    }
    if args.public_root:
        summary["public_replay"] = public_replay(args, executable, model)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
