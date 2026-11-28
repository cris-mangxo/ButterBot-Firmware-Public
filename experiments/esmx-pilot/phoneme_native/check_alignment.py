"""Compare native CTC span and deficit against the Python reference."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from compare import PHONES, cache_cases, frozen_model, oracle, synthetic_cases
from phoneme_confirm_lab import local_ctc_alignment


def probe(executable, scores, tokens):
    output = subprocess.run(
        [executable, *(str(phone) for phone in tokens)],
        input=np.asarray(scores, dtype="<f4").tobytes(),
        capture_output=True, check=True, timeout=10,
    )
    return json.loads(output.stdout)


def compare_case(executable, name, scores, tokens):
    cost, interval = local_ctc_alignment(scores, tokens, track_span=True)
    result = probe(executable, scores, tokens)
    expected = None if interval is None else [cost, *interval]
    if result != expected:
        raise ValueError(f"{name}: native CTC {result} != Python {expected}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("bank", type=Path)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    executable = args.executable.resolve(strict=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model, report = frozen_model(args.bank)
    framesets = [*synthetic_cases(), *cache_cases(args.cache)]
    templates = [(1,), (1, 1), (2, 1, 2), (19, 8, 17, 18, 5), tuple(range(1, PHONES))]
    checks = 0
    for name, frames in framesets:
        _, logits = oracle(model, frames)
        phone_scores = logits[:len(frames) * PHONES].reshape(len(frames), PHONES)
        for tokens in templates:
            compare_case(executable, name, phone_scores, tokens)
            checks += 1
    # Ties, repeated tokens requiring a blank, and maximum width.
    uniform = np.zeros((199, PHONES), dtype=np.float32)
    for tokens in [*templates, (1,) * 32]:
        compare_case(executable, "uniform", uniform, tokens)
        checks += 1
    malformed = [np.empty(0), np.zeros(PHONES + 1), np.full((8, PHONES), np.nan),
                 np.zeros((200, PHONES)), np.zeros((8, PHONES))]
    for index, scores in enumerate(malformed):
        if probe(executable, scores, (0,) if index == 4 else (1,)) is not None:
            raise ValueError(f"Malformed input {index} produced a match")
    print(json.dumps({"native_python_parity": checks, "malformed_rejected": len(malformed),
                      "bank_sha256": report["weights_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
