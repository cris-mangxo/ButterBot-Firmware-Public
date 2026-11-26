"""Calibrate and validate a small paired-phrase gate with the host C++ scorer."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import struct
import subprocess
from tempfile import TemporaryDirectory
import wave

from ald_round import ALD_SHA256, BASELINE_SHA256, canonical, digest
from d6_guard import MAX_FRAMES, guarded_score
from phrases_corpus import FIXTURE, fixture_rows
from pilot import feature, read_pcm, score

BANK_BYTES = 9676
FACT_GAP = 256
ANCHOR_SHA256 = "a2d967c01afd723c406a7c71cf2400da15763c444e607ad684ae9f6a2a4658b5"


def read_corpus(corpus):
    manifest = json.loads((corpus / "manifest.json").read_text("utf-8"))
    rows = fixture_rows()
    if manifest != {"fixture_sha256": digest(FIXTURE), "ald_sha256": ALD_SHA256, "samples": rows}:
        raise ValueError("El manifiesto no coincide con los casos publicados")
    return rows


def too_long(wav):
    with wave.open(str(wav)) as stream:
        return stream.getnframes() > MAX_FRAMES


def wire_bank(examples, dice_limits, deny_limits):
    def pack_bounded(signatures, radii):
        return b"".join(signature + struct.pack("<I", radius)
                        for signature, radius in zip(signatures, radii, strict=True))
    body = (pack_bounded(examples["d6"], dice_limits) + b"".join(examples["fact"])
            + pack_bounded(examples["deny_fact"], deny_limits) + struct.pack("<I", FACT_GAP))
    if len(body) != BANK_BYTES:
        raise ValueError("Tamaño incorrecto del banco")
    return body


def calibrate(rows, program, corpus, bank):
    examples = {category: [] for category in ("d6", "fact", "deny_fact")}
    denied_texts = []
    for row in rows:
        if row["split"] == "enroll":
            examples[row["category"]].append(feature(program, corpus / "audio" / (row["id"] + ".wav")))
            if row["category"] == "deny_fact":
                denied_texts.append(canonical(row["text"]))
    distance_rows = []
    omitted = Counter()
    with TemporaryDirectory(prefix="phrase-calibration-") as directory:
        probe = Path(directory) / "probe.bin"
        probe.write_bytes(wire_bank(examples, [0] * len(examples["d6"]), [0] * len(denied_texts)))
        for row in rows:
            if row["split"] != "calibrate":
                continue
            wav = corpus / "audio" / (row["id"] + ".wav")
            if too_long(wav):
                if row["label"]:
                    raise ValueError("Una orden positiva de calibración excede el turno permitido")
                omitted[row["category"]] += 1
                continue
            process = subprocess.run([program, "distances", probe], input=read_pcm(wav),
                                     capture_output=True, check=True, timeout=10)
            distance_rows.append((row, json.loads(process.stdout)))
    # Leave a 10% margin from the closest contrast for each dice phrase.
    dice_limits = [min(d["dice"][i] for row, d in distance_rows if row["label"] != 6) * 9 // 10
                   for i in range(len(examples["d6"]))]
    # Anchor the veto to same-text calibration takes. A nearby affirmative
    # reduces the radius further. The C++ gate still requires an affirmative gap.
    deny_limits = []
    for i, denial in enumerate(denied_texts):
        same_text = [d["deniedFacts"][i] for row, d in distance_rows
                     if row["category"] == "deny_fact" and canonical(row["text"]) == denial]
        if not same_text:
            raise ValueError("Faltan tomas separadas para un veto de Dato curioso")
        close_positive = min(d["deniedFacts"][i] for row, d in distance_rows if row["label"] == 4)
        deny_limits.append(min(max(same_text) * 115 // 100, close_positive * 85 // 100))
    with bank.open("xb") as output:
        output.write(wire_bank(examples, dice_limits, deny_limits))
    return {"dice_radii": dice_limits, "fact_denial_radii": deny_limits, "fact_gap": FACT_GAP,
            "calibration_in_range": len(distance_rows), "overlong_rejected": dict(omitted)}


def compare(program, baseline, anchor, bank, wav):
    if too_long(wav):
        rejected = {"label": 0, "rejected_by_length": True}
        return dict.fromkeys(("baseline", "previous", "phrases"), rejected)
    return {
        "baseline": score(program, baseline, wav),
        "previous": guarded_score(program, baseline, anchor, wav, mode="anchor"),
        "phrases": guarded_score(program, baseline, bank, wav, mode="phrases"),
    }


def totals(samples):
    result = {}
    for sample in samples:
        group = result.setdefault(sample["category"], {
            "trials": 0, "overlong": 0, "correct": dict.fromkeys(("baseline", "previous", "phrases"), 0),
            "false_accepts": dict.fromkeys(("baseline", "previous", "phrases"), 0),
            "false_accepts_by_label": {key: {} for key in ("baseline", "previous", "phrases")},
        })
        group["trials"] += 1
        group["overlong"] += sample["decisions"]["baseline"].get("rejected_by_length", False)
        for key, decision in sample["decisions"].items():
            label = decision["label"]
            if label == 6 and (decision.get("scenario"), decision.get("data")) != (7, 2):
                raise ValueError("D6 se desvió del escenario de CircuitMess")
            group["correct"][key] += label == sample["expected"]
            if not sample["expected"] and label:
                group["false_accepts"][key] += 1
                labels = group["false_accepts_by_label"][key]
                labels[str(label)] = labels.get(str(label), 0) + 1
    return result


def evaluate_ald(rows, program, baseline, anchor, bank, corpus, split="test"):
    results = []
    for row in rows:
        if row["split"] != split:
            continue
        wav = corpus / "audio" / (row["id"] + ".wav")
        results.append({"id": row["id"], "category": row["category"], "expected": row["label"],
                        "decisions": compare(program, baseline, anchor, bank, wav)})
    return {"categories": totals(results), "cases": results}


def replay(program, baseline, anchor, bank, corpus, fixture):
    manifest = json.loads((corpus / "manifest.json").read_text("utf-8"))
    if manifest["fixture_sha256"] != digest(fixture) or manifest["ald_sha256"] != ALD_SHA256:
        raise ValueError("El corpus anterior no coincide con su banco publicado")
    rows = manifest["samples"]
    if not all(re.fullmatch(r"(train|dev|test)-[0-9]{4}", row["id"]) for row in rows):
        raise ValueError("Nombre de toma anterior inválido")
    return evaluate_ald(rows, program, baseline, anchor, bank, corpus)


def write_header(bank, target):
    body = bank.read_bytes()
    if len(body) != BANK_BYTES:
        raise ValueError("Banco binario inválido")
    with target.open("x", encoding="utf-8") as out:
        out.write('#pragma once\n#include "PhraseGate.h"\n#include <bit>\n\n')
        out.write("inline constexpr auto PilotPhrases = "
                  f"std::bit_cast<EsMx::Kws::PhraseBank>(std::array<uint8_t, {BANK_BYTES}>{{\n")
        for start in range(0, len(body), 20):
            out.write("    " + ", ".join(map(str, body[start:start + 20])) + ",\n")
        out.write("});\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("fit", "ald", "confirm", "replay", "enrolled", "header"))
    for name in ("program", "baseline", "anchor", "bank", "corpus", "header"):
        parser.add_argument(f"--{name}", type=Path, required=name in ("program", "baseline", "bank"))
    parser.add_argument("--wide", action="store_true", help="Repite el banco amplio anterior")
    args = parser.parse_args()
    if digest(args.baseline) != BASELINE_SHA256:
        parser.error("La base difiere del piloto fijado")
    if args.operation == "header":
        if args.header is None:
            parser.error("--header es obligatorio")
        write_header(args.bank, args.header)
        result = {"header_sha256": digest(args.header)}
    else:
        if args.corpus is None:
            parser.error("--corpus es obligatorio")
        if args.operation == "fit":
            result = calibrate(read_corpus(args.corpus), args.program, args.corpus, args.bank)
        else:
            if args.anchor is None or digest(args.anchor) != ANCHOR_SHA256:
                parser.error("--anchor debe ser el verificador de dos frases fijado")
            if args.operation == "replay":
                fixture = FIXTURE.with_name("d6_cases.json" if args.wide else "d6_anchor_cases.json")
                result = replay(args.program, args.baseline, args.anchor, args.bank, args.corpus, fixture)
                result["corpus_fixture_sha256"] = digest(fixture)
            else:
                split = "test" if args.operation == "ald" else args.operation
                result = evaluate_ald(read_corpus(args.corpus), args.program, args.baseline,
                                      args.anchor, args.bank, args.corpus, split=split)
    print(json.dumps({"fixture_sha256": digest(FIXTURE), "ald_sha256": ALD_SHA256,
                      "bank_sha256": digest(args.bank), "bank_bytes": args.bank.stat().st_size,
                      **result}, indent=2))


if __name__ == "__main__":
    main()
