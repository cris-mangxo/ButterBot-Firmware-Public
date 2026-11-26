"""Train and measure the experimental D6 guard; raw data and weights stay local."""

import argparse
from collections import Counter
import json
from pathlib import Path
import struct
import subprocess
from tempfile import TemporaryDirectory
import wave

from ald_round import ALD_SHA256, BASELINE_SHA256, digest
from common_voice import DATASETS, evaluate_selected, read_test_split, select_examples
from pilot import FEATURE_BYTES, MAX_COST, MIN_GAP, feature, read_pcm, score

CLASSES = ("outside", "d6", "other_die", "negation")
HIDDEN = 12
DIVISOR = 256
MAX_FRAMES = 64000
LOCAL_CV_SHA256 = "21b9f7021b12a78395d07a24a7222854dd8220c06710021e7c9ff86a68e145e0"


def fit(np, x, y):
    rng = np.random.default_rng(42)
    weights = rng.normal(0, 0.045, (FEATURE_BYTES, HIDDEN))
    bias = np.zeros(HIDDEN)
    output = rng.normal(0, 0.08, (HIDDEN, len(CLASSES)))
    output_bias = np.zeros(len(CLASSES))
    scale = y @ (len(y) / (len(CLASSES) * y.sum(axis=0)))
    for _ in range(1200):
        linear = x @ weights + bias
        hidden = np.maximum(0, linear)
        logits = hidden @ output + output_bias
        rates = np.exp(logits - logits.max(axis=1, keepdims=True))
        rates /= rates.sum(axis=1, keepdims=True)
        delta = (rates - y) * scale[:, None] / len(x)
        input_delta = (delta @ output.T) * (linear > 0)
        output -= 0.1 * (hidden.T @ delta + 0.005 * output)
        output_bias -= 0.1 * delta.sum(axis=0)
        weights -= 0.1 * (x.T @ input_delta + 0.005 * weights)
        bias -= 0.1 * input_delta.sum(axis=0)
    first_scale = 127 / np.abs(weights).max()
    second_scale = 127 / np.abs(output).max()
    return (
        np.rint(weights * first_scale).astype(np.int8).T.copy(),
        np.rint(bias * first_scale * 32).astype("<i4"),
        np.rint(output * second_scale).astype(np.int8).T.copy(),
        np.rint(output_bias * (first_scale * 32 / DIVISOR) * second_scale).astype("<i4"),
    )


def integer_margins(np, data, model):
    weights, bias, output, output_bias = model
    hidden = np.maximum(0, data.astype(np.int64) @ weights.T + bias) // DIVISOR
    logits = hidden @ output.T + output_bias
    return logits[:, 1] - logits[:, [0, 2, 3]].max(axis=1)


def build(samples, program, folder, weights):
    import numpy as np

    signatures = {"train": [], "dev": []}
    labels = {"train": [], "dev": []}
    rejected_by_length = Counter()
    for row in samples:
        split = row["split"]
        if split not in signatures:
            continue
        wav = folder / (row["id"] + ".wav")
        with wave.open(str(wav)) as stream:
            if not 0 < stream.getnframes() <= MAX_FRAMES:
                if row["label"]:
                    raise ValueError("Orden positiva fuera de la duración admitida")
                rejected_by_length[split] += 1
                continue
        signatures[split].append(np.frombuffer(feature(program, wav), dtype=np.int8))
        labels[split].append(CLASSES.index(row["category"]))
    train = np.stack(signatures["train"])
    ground_truth = np.array(labels["dev"])
    model = fit(np, train.astype(np.float64) / 32, np.eye(len(CLASSES))[labels["train"]])
    margins = integer_margins(np, np.stack(signatures["dev"]), model)
    negatives = margins[ground_truth != 1]
    threshold = int(max(1, negatives.max() + 1))
    with weights.open("xb") as stream:
        for array in model:
            stream.write(array.tobytes())
        stream.write(struct.pack("<i", threshold))
    return {
        "train": dict(Counter(CLASSES[i] for i in labels["train"])),
        "dev": dict(Counter(CLASSES[i] for i in labels["dev"])),
        "rejected_by_length": dict(rejected_by_length),
        "dev_d6_admitted_by_guard": int((margins[ground_truth == 1] >= threshold).sum()),
        "dev_d6_total": int((ground_truth == 1).sum()),
        "dev_other_admitted_by_guard": int((negatives >= threshold).sum()),
        "threshold": threshold,
        "model_bytes": weights.stat().st_size,
        "model_sha256": digest(weights),
    }


def guarded_score(program, baseline, weights, wav, mode="guard"):
    result = subprocess.run(
        [program, mode, baseline, str(MAX_COST), str(MIN_GAP), weights],
        input=read_pcm(wav), capture_output=True, timeout=10,
    )
    if result.returncode == 3:
        return {"label": 0, "no_feature": True}
    result.check_returncode()
    return json.loads(result.stdout)


def evaluate_ald(samples, program, baseline, weights, folder, split, mode="guard"):
    results = []
    for row in samples:
        if row["split"] != split:
            continue
        wav = folder / (row["id"] + ".wav")
        with wave.open(str(wav)) as stream:
            too_long = stream.getnframes() > MAX_FRAMES
        if too_long:
            original = candidate = {"label": 0, "rejected_by_length": True}
        else:
            original = score(program, baseline, wav)
            candidate = guarded_score(program, baseline, weights, wav, mode)
        results.append({
            "id": row["id"], "category": row["category"], "expected": row["label"],
            "baseline": original, "guard": candidate,
        })
    return results


def summarize_ald(rows):
    categories = {}
    for row in rows:
        totals = categories.setdefault(row["category"], {
            "trials": 0, "baseline_success": 0, "candidate_success": 0, "candidate_false_d6": 0,
        })
        expected = row["expected"]
        baseline, candidate = row["baseline"], row["guard"]
        totals["trials"] += 1
        totals["baseline_success"] += baseline["label"] == expected
        totals["candidate_success"] += candidate["label"] == expected
        totals["candidate_false_d6"] += expected == 0 and candidate["label"] == 6
        for decision in (baseline, candidate):
            if decision["label"] == 6 and (decision.get("scenario"), decision.get("data")) != (7, 2):
                raise ValueError("D6 debe usar el escenario y los datos originales")
    return categories


def fit_anchors(samples, program, baseline, folder, weights):
    prototypes = [feature(program, folder / (row["id"] + ".wav"))
                  for row in samples if row["split"] == "train"]
    if len(prototypes) != 2:
        raise ValueError("Se necesitan las dos plantillas D6")
    blob = b"".join(prototypes)
    with TemporaryDirectory(prefix="d6-calibration-") as temporary:
        probe = Path(temporary) / "probe.bin"
        probe.write_bytes(blob + struct.pack("<I", 1))
        rows = evaluate_ald(samples, program, baseline, probe, folder, "dev", "anchor")
    # A non-D6 route never enters the anchor gate. Count its errors separately.
    positives = [1 - row["guard"]["d6_margin"] for row in rows
                 if row["expected"] == 6 and row["baseline"]["label"] in (0, 6)
                 and "d6_margin" in row["guard"]]
    negatives = [1 - row["guard"]["d6_margin"] for row in rows
                 if not row["expected"] and row["baseline"]["label"] in (0, 6)
                 and "d6_margin" in row["guard"]]
    if len(positives) != sum(row["expected"] == 6 for row in rows) or not negatives:
        raise ValueError("La calibración quedó incompleta")
    # Use at most the largest positive cost plus 15%, and leave at least a
    # quarter of the closest measured negative cost outside the acceptance cap.
    limit = min(max(positives) * 115 // 100, min(negatives) * 3 // 4)
    if limit <= 0:
        raise ValueError("No se pudo fijar un límite positivo")
    with weights.open("xb") as output:
        output.write(blob + struct.pack("<I", limit))
    return {
        "train_templates": len(prototypes), "dev_trials": len(rows),
        "dev_max_positive_cost": max(positives), "dev_min_negative_cost": min(negatives),
        "dev_d6_admitted_by_guard": sum(cost <= limit for cost in positives),
        "dev_d6_total": len(positives),
        "dev_other_admitted_by_guard": sum(cost <= limit for cost in negatives),
        "dev_legacy_false_accepts": sum(not row["expected"] and row["baseline"]["label"] in range(1, 6)
                                        for row in rows),
        "max_cost": limit, "model_bytes": weights.stat().st_size,
        "model_sha256": digest(weights),
    }


def write_anchor_header(weights, header):
    content = weights.read_bytes()
    if len(content) != 2 * FEATURE_BYTES + 4 or not struct.unpack("<I", content[-4:])[0]:
        raise ValueError("Verificador binario de tamaño o límite inválido")
    with header.open("x", encoding="utf-8") as output:
        output.write('#pragma once\n#include "Dice6Guard.h"\n#include <bit>\n\n')
        output.write("inline constexpr auto PilotDice6Anchors = "
                     "std::bit_cast<EsMx::Kws::Dice6Anchors>(std::array<uint8_t, 964>{\n")
        for offset in range(0, len(content), 20):
            output.write("    " + ", ".join(map(str, content[offset:offset + 20])) + ",\n")
        output.write("});\n")


def evaluate_cv(archive, program, baseline, weights, mode, new_cohort):
    dataset_id, size, published_hash = DATASETS["female"]
    archive_hash = digest(archive)
    if archive.stat().st_size != size or archive_hash not in (published_hash, LOCAL_CV_SHA256):
        raise ValueError("Archivo Common Voice fuera de las dos procedencias registradas")
    official = archive_hash == published_hash
    prefix, rows = read_test_split(archive)
    # A fixed continuation of the previous 32 speakers. No speaker or clip is
    # shared with that evaluation and neither cohort is used for tuning.
    end = 88 if new_cohort else 64
    start = 64 if new_cohort else 32
    clips = select_examples(rows, end) - select_examples(rows, start)
    result = evaluate_selected(archive, prefix, clips, {
        "baseline": lambda wav: score(program, baseline, wav),
        "guard": lambda wav: guarded_score(program, baseline, weights, wav, mode),
    })
    return {
        "dataset_id": dataset_id, "archive_sha256": archive_hash,
        "catalog_sha256": published_hash, "catalog_hash_verified": official,
        "provisional": not official, "test_rows": len(rows), "prior_speakers_excluded": start,
        **result,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("fit", "ald", "common-voice", "header"))
    for name in ("program", "baseline", "weights"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--anchor", action="store_true")
    parser.add_argument("--broad", action="store_true", help="Evalúa las dos plantillas en el banco amplio")
    parser.add_argument("--new-cohort", action="store_true", help="Usa los hablantes de Common Voice 65–88")
    parser.add_argument("--header", type=Path)
    args = parser.parse_args()
    if digest(args.baseline) != BASELINE_SHA256:
        parser.error("El clasificador base no coincide con la ronda anterior")
    mode = "anchor" if args.anchor else "guard"
    if args.operation == "header":
        if not args.anchor or args.header is None:
            parser.error("Para generar el encabezado se requiere --anchor y --header")
        write_anchor_header(args.weights, args.header)
        result = {"header_sha256": digest(args.header)}
    elif args.operation == "common-voice":
        if args.archive is None:
            parser.error("--archive es obligatorio para Common Voice")
        result = evaluate_cv(args.archive, args.program, args.baseline, args.weights, mode, args.new_cohort)
    else:
        if args.corpus is None:
            parser.error("--corpus es obligatorio para Ald")
        manifest = json.loads((args.corpus / "manifest.json").read_text("utf-8"))
        fixture = Path(__file__).with_name("d6_anchor_cases.json" if args.anchor and not args.broad else "d6_cases.json")
        if manifest["fixture_sha256"] != digest(fixture) or manifest["ald_sha256"] != ALD_SHA256:
            parser.error("El manifiesto no coincide con el corpus fijado")
        samples = manifest["samples"]
        audio = args.corpus / "audio"
        if args.operation == "fit":
            if args.anchor:
                if args.broad:
                    parser.error("Las plantillas se ajustan en su propio banco")
                result = fit_anchors(samples, args.program, args.baseline, audio, args.weights)
            else:
                result = build(samples, args.program, audio, args.weights)
        else:
            rows = evaluate_ald(samples, args.program, args.baseline, args.weights, audio, "test", mode)
            result = {"test": rows, "categories": summarize_ald(rows)}
        result.update({"fixture_sha256": digest(fixture), "ald_sha256": ALD_SHA256})
    print(json.dumps({"baseline_sha256": BASELINE_SHA256, "guard_sha256": digest(args.weights),
                      **result}, indent=2))


if __name__ == "__main__":
    main()
