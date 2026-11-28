"""Widen the original compact student and train on the same sealed Mexican turns."""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from chm_transcript_ctc_lab import EPOCHS, SEED, fit, pack_human, prepare, validate_human, write_json
from phoneme_chm150_lab import CORPUS_SHA256, SPEAKER_SPLIT_SHA256
from phoneme_reject_lab import BASE_SHA256
from phoneme_replay_lab import frozen_model
from phoneme_student_lab import RECIPE, WIDTH, Student, load_cache, measure, quantize
from time_lab import file_digest


WIDE = 96
PHONE_WEIGHT = 1.0
PROCEDURE = "bb-chm150-transcript-wider-1"


@torch.no_grad()
def widen(model):
    """Keep the trained subnet intact and let gradients reach additional channels."""
    torch.manual_seed(SEED)
    expanded = Student(model.phones.out_channels, width=WIDE)
    expanded.input.weight[:WIDTH].copy_(model.input.weight)
    expanded.input.bias[:WIDTH].copy_(model.input.bias)
    for source, target in zip(model.blocks, expanded.blocks, strict=True):
        target.temporal.weight[:WIDTH].copy_(source.temporal.weight)
        target.temporal.bias[:WIDTH].copy_(source.temporal.bias)
        target.channel.weight[:WIDTH].zero_()
        target.channel.weight[:WIDTH, :WIDTH].copy_(source.channel.weight)
        target.channel.bias[:WIDTH].copy_(source.channel.bias)
    expanded.phones.weight.zero_()
    expanded.phones.weight[:, :WIDTH].copy_(model.phones.weight)
    expanded.phones.bias.copy_(model.phones.bias)
    for source, target in ((model.intent, expanded.intent), (model.negation, expanded.negation)):
        target.weight.zero_()
        target.bias.copy_(source.bias)
        for part in range(5):
            target.weight[:, part * WIDE:part * WIDE + WIDTH].copy_(
                source.weight[:, part * WIDTH:(part + 1) * WIDTH])
    return expanded.eval()


@torch.inference_mode()
def verify_widening(before, after, rows):
    inputs, lengths, _, _ = pack_human(rows)
    errors = []
    for a, b in zip(before(inputs, lengths), after(inputs, lengths), strict=True):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)
        errors.append((a - b).abs().max().item())
    return errors


def verify_source_inputs(args, groups):
    manifest = json.loads(args.human_manifest.read_text())
    if (manifest["corpus"] != file_digest(args.corpus) or manifest["speaker_assignment"] != file_digest(args.speakers)
            or any(manifest["tools"][tool] != file_digest(getattr(args, tool)) for tool in ("frontend", "espeak", "ffmpeg"))):
        raise ValueError("CHM150 sources or tools differ from the 28-channel study")
    for phase, rows in groups.items():
        actual = [(name, speaker, len(frames), len(phones), checksum, sha256(frames.tobytes()).hexdigest())
                  for name, speaker, frames, phones, checksum in rows]
        keys = ("id", "speaker_partition_id", "frames", "phones", "pcm_sha256", "feature_sha256")
        expected = [tuple(row[key] for key in keys) for row in manifest["entries"][phase]]
        if actual != expected:
            raise ValueError(f"Human training cohort differs: {phase}")


def train(args):
    if file_digest(args.corpus) != CORPUS_SHA256 or file_digest(args.speakers) != SPEAKER_SPLIT_SHA256:
        raise ValueError("Corpus and speaker split must match the sealed CHM150 set")
    speech = args.speech / "index.json"
    voice_sha = file_digest(speech)
    base, original = frozen_model(args.bank)
    frontend = file_digest(args.frontend)
    if (original["weights_sha256"] != BASE_SHA256 or original["teacher_cache_sha256"] != voice_sha
            or json.loads(speech.read_text())["frontend_sha256"] != frontend):
        raise ValueError("Original checkpoint and frontend must match the synthetic cache")
    vocabulary, synthetic = load_cache(args.speech)
    roles = json.loads(args.speakers.read_text())["speaker_roles"]
    human = prepare(args, roles, vocabulary)
    verify_source_inputs(args, human)
    expanded = widen(base)
    errors = verify_widening(base, expanded, human["train"][:8] + human["dev"][:8])
    print(json.dumps({"initial_projection_max_abs_errors": errors}), flush=True)
    bank, model = quantize(fit(expanded, synthetic["train"], human["train"], PHONE_WEIGHT))
    args.output.mkdir(mode=0o700)
    weights = args.output / "student-int8-weights.npz"
    np.savez_compressed(weights, **bank)
    report = {
        "schema": RECIPE, "procedure": PROCEDURE, "width": WIDE, "labels": len(vocabulary),
        "weights_sha256": file_digest(weights), "seed": SEED, "epochs": EPOCHS,
        "initial_weights_sha256": BASE_SHA256, "teacher_cache_sha256": voice_sha,
        "human_manifest_sha256": file_digest(args.human_manifest),
        "human_phone_loss_weight": PHONE_WEIGHT,
        "initial_projection_max_abs_errors": errors,
        "weight_scale_bias_bytes": sum(value.nbytes for value in bank.values()),
        "native_activation_workspace_bytes_if_f32": 2 * 199 * WIDE * 4,
        "train_turns": len(human["train"]), "train_speakers": len({row[1] for row in human["train"]}),
        "human_development": validate_human(model, human["dev"]),
        "synthetic_development": measure(model, synthetic["dev"]),
        "source_sha256": file_digest(Path(__file__)),
    }
    write_json(args.output / "report.json", report)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("corpus", "speakers", "speech", "bank", "frontend", "ffmpeg", "espeak", "human-manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    train(parser.parse_args())
