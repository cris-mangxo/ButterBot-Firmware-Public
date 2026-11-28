"""Reduce QEMU parity output to a JSON result bound to the bank and emulator image."""

import argparse
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re

BANK = re.compile(r"^PHONEME_BANK ([a-f0-9]{64}) ([0-9]+)$")
CASE = re.compile(r"^PHONEME_CASE (\S+) ([0-9]+) ([0-9]+) ([01]) (\S+) ([0-9]+)$")
DONE = re.compile(r"^PHONEME_DONE ([0-9]+) ([0-9]+)$")


def checksum(path):
    with path.open("rb") as stream:
        return sha256_bytes(stream.read())


def sha256_bytes(data):
    return sha256(data).hexdigest()


def parse(lines, expected_cases):
    rows = []
    seals = []
    summaries = []
    for line in lines:
        if match := BANK.fullmatch(line):
            seals.append((match[1], int(match[2])))
        elif match := CASE.fullmatch(line):
            case, observed, expected, logits, error, elapsed = match.groups()
            rows.append({"case": case, "intent": int(observed), "expected_intent": int(expected),
                         "logits_within_tolerance": logits == "1", "max_abs_logit_error": float(error),
                         "emulator_elapsed_us": int(elapsed)})
        elif match := DONE.fullmatch(line):
            summaries.append((int(match[1]), int(match[2])))
        elif line.startswith("PHONEME_"):
            raise ValueError(f"Unrecognized emulator result row: {line}")
    if len(seals) != 1 or summaries != [(len(rows), 0)] or len(rows) != expected_cases:
        raise ValueError("Missing or unsuccessful emulator completion")
    if len({row["case"] for row in rows}) != len(rows) or any(
            not row["logits_within_tolerance"] or row["intent"] != row["expected_intent"]
            or not isfinite(row["max_abs_logit_error"]) or row["max_abs_logit_error"] < 0 for row in rows):
        raise ValueError("Unexpected duplicate or parity failure in emulator output")
    return seals[0], rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("bank_report", type=Path)
    parser.add_argument("benchmark_binary", type=Path)
    parser.add_argument("--emulated-flash", type=Path)
    parser.add_argument("--expected-cases", type=int, choices=(14, 19), default=19)
    args = parser.parse_args()
    text = args.log.read_bytes()
    (bank, size), rows = parse(text.decode("utf-8").splitlines(), args.expected_cases)
    if bank != json.loads(args.bank_report.read_text())["weights_sha256"] or size != 9180:
        raise ValueError("QEMU result disagrees with the requested frozen bank")
    result = {
        "target": "esp32s3-qemu", "weight_sha256": bank, "bank_bytes": size,
        "benchmark_binary_sha256": checksum(args.benchmark_binary),
        "benchmark_binary_bytes": args.benchmark_binary.stat().st_size,
        "qemu_log_sha256": sha256_bytes(text), "case_count": len(rows), "checks_failed": 0,
        "maximum_absolute_logit_error": max(row["max_abs_logit_error"] for row in rows),
        "rows": rows,
    }
    if args.emulated_flash:
        result.update({"simulated_flash_sha256": checksum(args.emulated_flash),
                       "simulated_flash_bytes": args.emulated_flash.stat().st_size})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
