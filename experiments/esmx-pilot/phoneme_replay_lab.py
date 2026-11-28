"""Replay the frozen compact phoneme experiment on its separate desktop panel."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

import keyword_lab as kw
from capture_replay import pcm_from, recordings
from phoneme_student_lab import RECIPE, Student, gate
from time_lab import file_digest, invoke, supplemental


def frozen_model(directory):
    report = json.loads((directory / "report.json").read_text())
    weight_path = directory / "student-int8-weights.npz"
    if report["schema"] != RECIPE or file_digest(weight_path) != report["weights_sha256"]:
        raise ValueError("Frozen checkpoint differs from its seal")
    return load_int8_weights(weight_path, report["labels"]), report


def load_int8_weights(weight_path, labels):
    model = Student(labels).eval()
    values = {}
    with np.load(weight_path, allow_pickle=False) as weights:
        for name, tensor in model.state_dict().items():
            raw = weights[name]
            if raw.shape != tuple(tensor.shape):
                raise ValueError("Frozen tensor shape mismatch")
            if name.endswith("weight"):
                scale = weights[name + ".scale"]
                if (raw.dtype != np.int8 or scale.shape != (tensor.shape[0],) + (1,) * (tensor.ndim - 1)
                        or scale.dtype != np.float32 or not np.isfinite(scale).all() or not (scale > 0).all()):
                    raise ValueError("Frozen tensor has invalid int8 scale")
                raw = raw.astype(np.float32) * scale
            if raw.dtype != np.float32 or not np.isfinite(raw).all():
                raise ValueError("Frozen tensor has invalid values")
            values[name] = torch.from_numpy(raw)
    model.load_state_dict(values)
    return model


def native_frames(features, pcm):
    segment = kw.trimmed(pcm)
    if len(segment) < 800:
        return None
    stdout = invoke(features, segment, "word-frames")
    if len(stdout) % kw.BANDS:
        raise ValueError("Native audio frontend produced an incomplete frame")
    frames = np.frombuffer(stdout, dtype=np.int8).reshape(-1, kw.BANDS)
    return frames if len(frames) >= 8 else None


@torch.inference_mode()
def intent_from_frames(model, frames):
    if frames is None:
        return 0
    tensor = frames.astype(np.float32)
    _, intents, denial = model(torch.from_numpy(tensor.T[None] / 32.0), torch.tensor([len(tensor)]))
    return kw.LABELS[int(gate(intents, denial).item())]


def student_intent(model, features, pcm):
    return intent_from_frames(model, native_frames(features, pcm))


def overlaps_adjustment(pcm, hashes, native_hashes=()):
    segment = kw.trimmed(pcm)
    if sha256(segment).hexdigest() in native_hashes:
        return True
    trimmed = np.frombuffer(segment, dtype="<i2").astype(np.float32) / 32_768
    return sha256(trimmed.tobytes()).hexdigest() in hashes


def evaluate(args, model, adjustment_hashes, negative_hashes=(), predict=student_intent):
    rows = list(recordings(args.root))
    if len(rows) != 112:
        raise ValueError("Historical comparison panel must contain 112 rows")
    if args.coerll_supplemental:
        rows.append(supplemental(args.root, args.coerll_supplemental))
    for identifier, split, category, expected, path, seal in rows:
        pcm = pcm_from(path, seal)
        outcome = {"id": identifier, "split": split, "category": category, "expected": expected}
        captured = invoke(args.capture, pcm, "--pcm-with-hangover")
        for mode, sample in (("raw", pcm), ("retained", captured)):
            same_pcm = overlaps_adjustment(sample, adjustment_hashes, negative_hashes)
            if same_pcm and split.startswith(("public", "supplemental")):
                raise ValueError(f"Human evaluation intersects the fitting or development cache: {identifier}")
            baseline = candidate = 0
            if 0 < len(sample) <= 128_000:
                baseline = int(invoke(args.reference, sample))
                candidate = predict(model, args.features, sample)
            outcome[mode] = {
                "baseline": baseline, "candidate": candidate, "combined": baseline or candidate,
                "matches_adjustment_pcm": same_pcm,
            }
        yield outcome


def summarize(rows, mode):
    extra = []
    wrong = []
    wrong_intent = Counter()
    correct = {name: Counter() for name in ("baseline", "candidate", "combined")}
    for row in rows:
        observed = row[mode]
        for arm in correct:
            if observed[arm] == row["expected"]:
                correct[arm][row["expected"]] += 1
        if observed["baseline"] == 0 and observed["combined"]:
            if observed["combined"] == row["expected"]:
                extra.append(row["id"])
            else:
                wrong.append(row["id"])
                wrong_intent[observed["combined"]] += 1
    overlap = [row["id"] for row in rows if row[mode]["matches_adjustment_pcm"]]
    return {
        "correct": correct, "new_correct": extra, "new_wrong": wrong, "new_wrong_intent": wrong_intent,
        "matching_adjustment_pcm": overlap,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "features", "reference", "capture", "frozen", "cache", "coerll-supplemental", "negatives"):
        parser.add_argument("--" + name, type=lambda p: Path(p).resolve(strict=True),
                            required=name not in ("coerll-supplemental", "negatives"))
    parser.add_argument("--episode", help="Output only the decision from one reserved human turn after capture")
    args = parser.parse_args()
    torch.set_num_threads(2)
    model, data = frozen_model(args.frozen)
    if file_digest(args.cache / "index.json") != data["teacher_cache_sha256"]:
        raise ValueError("Frozen model training provenance differs")
    cache = json.loads((args.cache / "index.json").read_text())
    if cache["frontend_sha256"] != file_digest(args.features):
        raise ValueError("Frozen audio frontend differs")
    negative_index = args.negatives / "index.json" if args.negatives else None
    if data.get("negative_cache_sha256") != (file_digest(negative_index) if negative_index else None):
        raise ValueError("Frozen model negative speech provenance differs")
    negative_records = json.loads(negative_index.read_text())["records"] if negative_index else {}
    negative_hashes = {row["effective_pcm_sha256"] for row in negative_records.values()}
    rows = list(evaluate(args, model, {row["pcm_sha256"] for row in cache["records"].values()}, negative_hashes))
    if args.episode:
        reserved = [row for row in rows if row["id"] == args.episode and row["split"] in {
            "public_reserved_time", "public_reserved_d6", "supplemental_human_time",
        }]
        if len(reserved) != 1:
            parser.error("episode must refer to one sealed human evaluation recording")
        print(reserved[0]["retained"]["combined"])
        return
    historical = rows[:112]
    groups = {
        "historical_112": historical,
        "historical_without_matching_adjustment_pcm": [
            row for row in historical if not any(row[mode]["matches_adjustment_pcm"] for mode in ("raw", "retained"))
        ],
    }
    for name in ("public_reserved_time", "public_reserved_d6", "supplemental_human_time"):
        groups[name] = [row for row in rows if row["split"] == name]
    report = {
        "weights_sha256": data["weights_sha256"],
        "reference_sha256": file_digest(args.reference), "capture_sha256": file_digest(args.capture),
        "summary": {name: {"items": len(panel), **{mode: summarize(panel, mode) for mode in ("raw", "retained")}}
                    for name, panel in groups.items()},
        "regression": [row for row in rows if row["id"] in {
            "public-03", "public-37", "public-81", "public-88", "public-89", "veto-fact",
        }],
    }
    print(json.dumps(report, indent=2) + "\n", end="")


if __name__ == "__main__":
    main()
