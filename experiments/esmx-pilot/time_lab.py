"""Freeze and replay an independent, whole-turn binary Time acoustic bank."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import struct
import subprocess
import unicodedata

import numpy as np

from capture_replay import pcm_from, recordings

FRAMES, BANDS, WARP = 24, 20, 3
FEATURE_BYTES = FRAMES * BANDS
TIME = 5
BANK_HEADER = struct.Struct("<8sBBBBIi")
BANK_MAGIC = b"BBTIME01"
BUDGET = 20_000
SPANISHPOD_FILE = "hora-ajuste-spanishpod-a0056-esmx.wav"
SPANISHPOD_SHA256 = "0319e9b81c27930015d381c7f6a6c5aa31bc862509c46815814ba681936d3751"
SYNTHETIC_COUNTS = {"train": 3, "dev": 2, "test": 2}
FIXTURE = Path(__file__).with_name("fallback_cases.json")


def digest(data):
    return sha256(data).hexdigest()


def file_digest(path):
    return digest(path.read_bytes())


def invoke(program, data, *arguments):
    result = subprocess.run([program, *arguments], input=data, capture_output=True, timeout=10)
    if result.returncode == 3 and not result.stdout:
        return b""
    result.check_returncode()
    return result.stdout


def feature(program, pcm):
    if not pcm or len(pcm) > 128_000:
        return None
    output = invoke(program, pcm, "features")
    if not output:
        return None
    if len(output) != FEATURE_BYTES:
        raise ValueError("C++ feature encoder returned an unexpected size")
    return np.frombuffer(output, dtype=np.int8).reshape(FRAMES, BANDS)


def canonical(text):
    text = unicodedata.normalize("NFKD", text.casefold())
    return "".join(char for char in text if char.isalpha())


def synthetic_rows(args, split):
    fixture = json.loads(FIXTURE.read_text())
    phrases = [canonical(phrase) for groups in fixture.values() for category in groups.values()
               for phrase in category]
    if len(phrases) != len(set(phrases)) or set(fixture) != set(SYNTHETIC_COUNTS):
        raise ValueError("Repeated phrases or unexpected splits")
    for category, texts in fixture[split].items():
        voices = list(str(index) for index in range(SYNTHETIC_COUNTS[split]))
        if split == "train":
            voices.append("ald")
        for index in range(len(texts)):
            for voice in voices:
                key = f"{split}-{category}-{index:02}-{voice}"
                yield key, f"synthetic_{split}", category, TIME if category == "time" else 0, args.wav / f"{key}.wav", None


def normalized(values, centered):
    signed = values.astype(np.int16)
    if centered:
        signed -= (signed.sum(axis=-2, keepdims=True) // FRAMES).astype(np.int16)
    return signed


def distances(query, templates):
    """Integer DTW over the production signature, with the production warp."""
    sizes = len(templates)
    inf = 1 << 28
    previous = np.full((FRAMES + 1, sizes), inf, dtype=np.int32)
    previous[0] = 0
    for i in range(1, FRAMES + 1):
        current = np.full_like(previous, inf)
        lower, upper = max(1, i - WARP), min(FRAMES, i + WARP) + 1
        costs = np.abs(templates[:, lower - 1:upper - 1] - query[i - 1]).sum(axis=-1)
        for j in range(lower, upper):
            current[j] = costs[:, j - lower] + np.minimum(np.minimum(previous[j], previous[j - 1]), current[j - 1])
        previous = current
    return previous[FRAMES]


def farthest_first(pool, distance_matrix, maximum, seed):
    selected = [seed]
    nearest = distance_matrix[seed].copy()
    while len(selected) < min(maximum, len(pool)):
        nearest[selected] = -1
        chosen = int(np.argmax(nearest))
        selected.append(chosen)
        nearest = np.minimum(nearest, distance_matrix[chosen])
    return selected


def select_templates(train, centered):
    pos_keys, positive = zip(*[(key, data) for key, label, data in train if label == TIME])
    neg_keys, negative = zip(*[(key, data) for key, label, data in train if label != TIME])
    pos = normalized(np.stack(positive), centered)
    neg = normalized(np.stack(negative), centered)
    pos_matrix = np.stack([distances(template, pos) for template in pos])
    neg_to_pos = np.array([min(distances(template, pos)) for template in neg])
    # Keep the negatives nearest to a time command, then retain diverse members of that pool.
    hard = np.argsort(neg_to_pos, kind="stable")[:48]
    neg_matrix = np.stack([distances(neg[index], neg[hard]) for index in hard])
    pos_indices = farthest_first(pos, pos_matrix, 16, int(np.argmin(pos_matrix.sum(axis=1))))
    if "spanishpod_adjustment" in pos_keys:
        adjustment = pos_keys.index("spanishpod_adjustment")
        if adjustment not in pos_indices:
            pos_indices[-1] = adjustment
    neg_indices = hard[farthest_first(hard, neg_matrix, 24, 0)]
    keys = [*(pos_keys[i] for i in pos_indices), *(neg_keys[i] for i in neg_indices)]
    return keys, np.stack([*(positive[i] for i in pos_indices), *(negative[i] for i in neg_indices)]), len(pos_indices)


def evaluate_cost(data, transformed, positive_count, centered):
    path_costs = distances(normalized(data, centered), transformed)
    time, other = int(min(path_costs[:positive_count])), int(min(path_costs[positive_count:]))
    return time, other - time


def calibrate(train, dev):
    attempts = []
    for centered in (False, True):
        keys, templates, n_pos = select_templates(train, centered)
        transformed = normalized(templates, centered)
        costs = [(key, label, *evaluate_cost(data, transformed, n_pos, centered)) for key, label, data in dev]
        # Quantized radii are calibrated only on synthetic development positives.
        radii = sorted({((cost + 127) // 128) * 128 for _, label, cost, _ in costs if label == TIME})
        for radius in radii:
            negative_margins = [margin for _, label, cost, margin in costs if label != TIME and cost <= radius]
            gap = max(64, max(negative_margins, default=0) + 64)
            good = sum(label == TIME and cost <= radius and margin >= gap for _, label, cost, margin in costs)
            bad = sum(label != TIME and cost <= radius and margin >= gap for _, label, cost, margin in costs)
            if bad:
                raise AssertionError("Calibration must reject all development negatives")
            attempts.append((good, -radius, gap, centered, keys, templates, n_pos))
    good, minus_radius, gap, centered, keys, templates, n_pos = max(attempts, key=lambda a: a[:4])
    return templates, dict(centered=centered, radius=-minus_radius, min_margin=gap,
                           n_positive=n_pos, template_ids=keys, dev_accepted_positive=good)


def freeze(args):
    inputs = {split: list(synthetic_rows(args, split)) for split in ("train", "dev")}
    if not args.skip_adjustment:
        inputs["train"].append(("spanishpod_adjustment", "human_train", "time", TIME,
                                args.root / "primer-candidato-voces" / SPANISHPOD_FILE, SPANISHPOD_SHA256))
    counts, tensors, hashes, exclusions = {}, {}, {}, []
    for split, rows in inputs.items():
        vectors = []
        for key, _, _, label, path, expected_hash in rows:
            pcm = pcm_from(path, expected_hash)
            hashes[key] = digest(pcm)
            data = feature(args.features, pcm)
            if data is None:
                exclusions.append(key)
            else:
                vectors.append((key, label, data))
        tensors[split] = vectors
        counts[split] = dict(Counter("time" if label == TIME else "control" for _, label, _ in vectors))

    templates, config = calibrate(tensors["train"], tensors["dev"])
    blob = BANK_HEADER.pack(BANK_MAGIC, FRAMES, BANDS, config["n_positive"], len(templates),
                            config["radius"] | (int(config["centered"]) << 31), config["min_margin"]) + templates.tobytes()
    if len(blob) > BUDGET:
        raise ValueError(f"Bank exceeds the {BUDGET} byte limit")
    config.update({"bank_sha256": digest(blob), "bank_file_bytes": len(blob),
                   "dtw_rows_bytes_per_template_estimate": 2 * (FRAMES + 1) * 4,
                   "esp32s3_memory_and_latency_measured": False,
                   "feature_tool_sha256": file_digest(args.features), "recipe_sha256": file_digest(Path(__file__)),
                   "fixture_sha256": file_digest(FIXTURE), "split_counts": counts, "unusable": exclusions,
                   "training_pcm_hashes": hashes, "backend": "integer host DTW over production C++ features"})
    with (args.bank_dir / "time-bank.host-only").open("xb") as target:
        target.write(blob)
    with (args.bank_dir / "frozen.json").open("x") as target:
        json.dump(config, target, indent=2)
    print(json.dumps({key: value for key, value in config.items() if key != "training_pcm_hashes"}, indent=2))


def supplemental(root, manifest):
    voices = (root / "primer-candidato-voces").resolve(strict=True)
    if manifest.parent != voices:
        raise ValueError("Supplemental manifests belong in primer-candidato-voces")
    clip = json.loads(manifest.read_text())
    reservation = (voices / clip["reserved_manifest"]).resolve(strict=True)
    wave = (voices / clip["sample"]).resolve(strict=True)
    if any(path.parent != voices for path in (wave, reservation)):
        raise ValueError("Invalid supplemental source path")
    if file_digest(reservation) != clip["reserved_manifest_sha256"]:
        raise ValueError("Supplemental reservation mismatch")
    metadata = json.loads(reservation.read_text())
    if (metadata["locale"] != "es-MX" or metadata["sample_id"] != clip["sample_id"]
            or metadata["class"] != "TimeNow" or clip["split"] != "heldout_supplemental"
            or metadata["owner_voice"] or metadata["training_and_threshold_tuning"]
            or metadata["allow_machine_learning_gate_use"] or clip["counts_as_spontaneous_direct_order"]):
        raise ValueError("Unexpected supplemental clip provenance")
    return clip["sample_id"], "supplemental_human_time", "time", TIME, wave, clip["wav_sha256"]


def replay(args):
    config = json.loads((args.bank_dir / "frozen.json").read_text())
    blob = (args.bank_dir / "time-bank.host-only").read_bytes()
    if (digest(blob) != config["bank_sha256"] or file_digest(args.features) != config["feature_tool_sha256"]
            or file_digest(Path(__file__)) != config["recipe_sha256"] or file_digest(FIXTURE) != config["fixture_sha256"]):
        raise ValueError("Frozen artifact or recipe mismatch")
    magic, frames, bands, n_pos, count, radius_and_flag, gap = BANK_HEADER.unpack_from(blob)
    centered, radius = bool(radius_and_flag >> 31), radius_and_flag & 0x7FFFFFFF
    if (magic, frames, bands) != (BANK_MAGIC, FRAMES, BANDS) or not 0 < n_pos < count:
        raise ValueError("Invalid frozen model")
    templates = np.frombuffer(blob[BANK_HEADER.size:], dtype=np.int8).reshape(count, FRAMES, BANDS)
    transformed = normalized(templates, centered)
    training_hashes = set(config["training_pcm_hashes"].values())

    def observe(case):
        key, split, category, expected, path, pin = case
        pcm = pcm_from(path, pin)
        seen_pcm = digest(pcm) in training_hashes
        if seen_pcm and split not in ("synthetic_panel", "synthetic_aliases"):
            raise ValueError(f"Frozen holdout has an exact training PCM duplicate: {key}")
        row = {"id": key, "split": split, "category": category, "expected": expected,
               "synthetic_regression_seen_pcm": seen_pcm}
        for mode, capture_option in (("raw", None), ("captured", "--pcm"), ("captured_hold", "--pcm-with-hangover")):
            offered = pcm if capture_option is None else invoke(args.capture, pcm, capture_option)
            baseline = int(invoke(args.reference, offered)) if offered and len(offered) <= 128_000 else 0
            data = feature(args.features, offered)
            cost, margin = (None, None) if data is None else evaluate_cost(data, transformed, n_pos, centered)
            rescue = bool(baseline == 0 and cost is not None and cost <= radius and margin >= gap)
            row[mode] = {"samples": len(offered) // 2, "cost": cost, "margin": margin, "baseline": baseline,
                         "rescued": rescue, "combined": TIME if rescue else baseline}
        return row

    cases = list(recordings(args.root))
    if len(cases) != args.panel_count:
        raise ValueError(f"Unexpected prior panel size {len(cases)}")
    cases += list(synthetic_rows(args, "test"))
    for case in recordings(args.root, args.additional_reserved_manifest):
        if case[0] not in {base_case[0] for base_case in cases}:
            cases.append(case)
    if args.coerll_supplemental:
        cases.append(supplemental(args.root, args.coerll_supplemental))
    with ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(observe, cases))

    prior = rows[:args.panel_count]
    seen = [row for row in prior if row["synthetic_regression_seen_pcm"]]
    groups = {"full_panel_regression": prior,
              "panel_without_exact_overlap": [row for row in prior if not row["synthetic_regression_seen_pcm"]],
              "reserved_human_time": [row for row in rows if row["split"] == "public_reserved_time"],
              "supplemental_human_time": [row for row in rows if row["split"] == "supplemental_human_time"],
              "synthetic_test": [row for row in rows if row["split"] == "synthetic_test"]}
    summary = {}
    for group, members in groups.items():
        summary[group] = {}
        for mode in ("raw", "captured", "captured_hold"):
            summary[group][mode] = {
                "time_total": sum(row["expected"] == TIME for row in members),
                "time_baseline": sum(row["expected"] == TIME and row[mode]["baseline"] == TIME for row in members),
                "time_combined": sum(row["expected"] == TIME and row[mode]["combined"] == TIME for row in members),
                "non_time_total": sum(row["expected"] != TIME for row in members),
                "new_time_false_activations": [row["id"] for row in members if row["expected"] != TIME and row[mode]["rescued"]],
            }
    print(json.dumps({"bank_sha256": config["bank_sha256"], "bank_file_bytes": len(blob),
                      "esp32s3_memory_and_latency_measured": False,
                      "capture_tool_sha256": file_digest(args.capture), "reference_sha256": file_digest(args.reference),
                      "synthetic_regression_seen_pcm_ids": [row["id"] for row in seen],
                      "summary": summary, "rows": rows}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "replay"))
    for field in ("root", "features", "wav", "bank-dir"):
        parser.add_argument("--" + field, type=lambda path: Path(path).resolve(strict=True), required=True)
    parser.add_argument("--reference", type=lambda path: Path(path).resolve(strict=True))
    parser.add_argument("--capture", type=lambda path: Path(path).resolve(strict=True))
    parser.add_argument("--skip-adjustment", action="store_true")
    parser.add_argument("--additional-reserved-manifest", type=lambda path: Path(path).resolve(strict=True), action="append", default=[])
    parser.add_argument("--coerll-supplemental", type=lambda path: Path(path).resolve(strict=True))
    parser.add_argument("--panel-count", type=int, default=112)
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze(args)
    elif args.reference and args.capture:
        replay(args)
    else:
        parser.error("--reference and --capture are required for replay")


if __name__ == "__main__":
    main()
