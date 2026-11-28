"""Calibrate and replay a desktop D6-only reference with full-vocabulary CTC."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from capture_replay import pcm_from, recordings
import keyword_lab as kw
from phoneme_confirm_lab import local_ctc_cost, model_output
from phoneme_replay_lab import frozen_model, native_frames, strip_digital_edges
from phoneme_teacher_lab import WEIGHTS_SHA256, load_teacher, phonemes, prepared_pcm
from time_lab import file_digest, invoke


SCHEMA = "bb-d6-full-ctc-reference-1"
SYNTHETIC_SHA256 = "df9a9dac5e3003469516a0996242df70c887ebcd7f9ba22071216bfe1d8d44d4"
HUMAN_SHA256 = "37722cb8ce5a4cc3a4284126841d6e048cd84a62f704609b758daaf89eb1a93a"
STUDENT_SHA256 = "1785628e682569e77cdc4e934da8714cf5021b518a2bf745abc76aa24cfd0619"
PHRASES = {
    "order": ("tira un dado de seis", "tire un dado de seis", "tiren un dado de seis",
              "tírame un dado de seis", "lanza un dado de seis", "avienta un dado de seis"),
    "denial": ("no tires", "no avientes", "no lances", "no lo tires", "no quiero tirar", "mejor no tires"),
    "other_die": ("dado de ocho", "dado de diez", "dado de veinte", "dado de doce"),
    "multiple": ("dos dados de seis", "tres dados de seis", "cuatro dados de seis", "dados de seis"),
}


def paths(espeak, vocabulary):
    return {role: [[vocabulary[phone] for phone in phonemes(str(espeak), phrase + ".")]
                   for phrase in phrases] for role, phrases in PHRASES.items()}


def observation(scores, templates, negated):
    costs = {kind: min(local_ctc_cost(scores, seq) for seq in options)
             for kind, options in templates.items()}
    command = costs["order"]
    if not all(isfinite(value) for value in costs.values()):
        return None
    competitor = min(value for key, value in costs.items() if key != "order")
    return {"cost": command, "margin": competitor - command, "veto": negated, "costs": costs}


def accepted(item, limit, margin):
    return bool(item and not item["veto"] and item["cost"] <= limit and item["margin"] >= margin)


def eligible_pcm(data):
    return 0 < len(data) <= 128_000 and len(data) % 2 == 0


def select_limit(development):
    if not any(row["positive"] for row in development) or not any(not row["positive"] for row in development):
        raise ValueError("D6 calibration needs affirmatives and controls")
    boundaries = sorted({0.0} | {row["score"]["margin"] for row in development
                                  if row["score"] and not row["score"]["veto"] and row["score"]["margin"] > 0})
    # Within an interval bounded by measured margins, the eligible items are
    # unchanged. Use the midpoint instead of picking one observed item's margin.
    margins = [0.0, *((left + right) / 2 for left, right in zip(boundaries, boundaries[1:]))]
    proposals = []
    for margin in margins:
        ceiling = min((row["score"]["cost"] for row in development
                       if not row["positive"] and accepted(row["score"], float("inf"), margin)),
                      default=float("inf"))
        eligible = [row["score"]["cost"] for row in development
                    if row["positive"] and accepted(row["score"], float("inf"), margin)
                    and row["score"]["cost"] < ceiling]
        if eligible:
            proposals.append((len(eligible), -max(eligible), margin))
    if not proposals:
        return None, None
    _, neg_limit, margin = max(proposals)
    return -neg_limit, margin


class Scorer:
    def __init__(self, args):
        self.teacher, self.extractor, vocabulary = load_teacher(args.teacher)
        self.student, report = frozen_model(args.student)
        if report["weights_sha256"] != STUDENT_SHA256:
            raise ValueError("Unexpected compact negation model")
        self.templates = paths(args.espeak, vocabulary)
        self.features = args.features
        self.seal = {"teacher": WEIGHTS_SHA256, "student": STUDENT_SHA256,
                     "frontend": file_digest(args.features), "espeak": file_digest(args.espeak),
                     "templates": self.templates}
        self.cache = {}

    @torch.inference_mode()
    def run(self, pcm):
        if not eligible_pcm(pcm):
            return None
        pcm = strip_digital_edges(pcm)
        key = sha256(pcm).hexdigest()
        if key not in self.cache:
            self.cache[key] = self.predict(pcm)
        return self.cache[key]

    def predict(self, pcm):
        frames = native_frames(self.features, pcm)
        if frames is None:
            return None
        _, _, denial = model_output(self.student, frames)
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32_768
        packed = self.extractor(samples, sampling_rate=16_000, return_tensors="pt")
        logits = self.teacher(**packed).logits[0].numpy()
        return observation(logits, self.templates, bool(denial.item() >= 0))


def prepare(args, scorer):
    human_index = args.human / "manifest.json"
    if (file_digest(args.synthetic_index) != SYNTHETIC_SHA256 or file_digest(human_index) != HUMAN_SHA256):
        raise ValueError("Unexpected D6 development inputs")
    index = json.loads(args.synthetic_index.read_text())
    if index["frontend_sha256"] != scorer.seal["frontend"]:
        raise ValueError("Frontend differs from the sealed synthetic bank")
    roles = SimpleNamespace(root=args.root, wav=args.wav, skip_adjustment=True)
    data = []
    for row in kw.rows(roles, ("dev",)):
        pcm = prepared_pcm(row)
        signature = sha256((np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32_768).tobytes()).hexdigest()
        if signature != index["records"][row.key]["pcm_sha256"]:
            raise ValueError(f"Development PCM changed for {row.key}")
        data.append({"id": row.key, "group": "synthetic", "positive": row.intent == 6 and not row.denied(),
                     "score": scorer.run(pcm), "pcm_sha256": pcm_seals(pcm)})
    for row in json.loads(human_index.read_text())["samples"]:
        if row["split"] == "dev":
            pcm = pcm_from(args.human / "audio" / (row["id"] + ".wav"), row["wav_sha256"])
            data.append({"id": row["id"], "group": "human", "positive": row["label"] == 1,
                         "score": scorer.run(pcm), "pcm_sha256": pcm_seals(pcm)})
    limit, margin = select_limit(data)
    report = {"schema": SCHEMA, "seal": scorer.seal, "limit": limit, "margin": margin, "development": data}
    with args.guard.open("x") as output:
        output.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if limit is None:
        return {"guard_sha256": file_digest(args.guard), "status": "no_development_fit",
                "development_items": len(data)}
    counts = Counter((row["group"], row["positive"], accepted(row["score"], limit, margin)) for row in data)
    return {"guard_sha256": file_digest(args.guard), "counts": [[*key, value] for key, value in sorted(counts.items())]}


def pcm_seals(pcm):
    aligned = strip_digital_edges(pcm)
    return sorted({sha256(block).hexdigest() for block in (pcm, aligned, kw.trimmed(aligned)) if block})


def frozen_guard(args, scorer):
    frozen = json.loads(args.guard.read_text())
    if frozen["schema"] != SCHEMA or frozen["seal"] != scorer.seal:
        raise ValueError("Reference guard provenance changed")
    limit, margin = frozen["limit"], frozen["margin"]
    if limit is not None and not (isfinite(limit) and limit >= 0 and isfinite(margin) and margin >= 0):
        raise ValueError("Invalid D6 decision limits")
    development_pcm = {checksum for row in frozen["development"] for checksum in row["pcm_sha256"]}
    return limit, margin, development_pcm


def summarize(rows):
    report = {}
    for mode in ("raw", "retained"):
        extra = [row for row in rows if row[mode]["compact"] != row[mode]["combined"]]
        report[mode] = {
            "expected_d6": sum(row["expected"] == 6 for row in rows),
            "expected_abstention": sum(row["expected"] == 0 for row in rows),
            "compact_correct_d6": sum(row["expected"] == row[mode]["compact"] == 6 for row in rows),
            "added_right": [row["id"] for row in extra if row["expected"] == 6],
            "added_wrong": [row["id"] for row in extra if row["expected"] != 6],
        }
    return report


def replay(args, scorer):
    limit, margin, development_pcm = frozen_guard(args, scorer)
    panel = list(recordings(args.root))
    if len(panel) != 112:
        raise ValueError("Historical comparison panel must contain 112 rows")
    cases = []
    for identifier, split, category, expected, path, digest in panel:
        pcm = pcm_from(path, digest)
        row = {"id": identifier, "split": split, "category": category, "expected": expected}
        for mode, data in (("raw", pcm), ("retained", invoke(args.capture, pcm, "--pcm-with-hangover"))):
            eligible = eligible_pcm(data)
            overlap = eligible and not development_pcm.isdisjoint(pcm_seals(data))
            if overlap and split.startswith("public"):
                raise ValueError(f"Public case {identifier} overlaps development PCM")
            compact = int(invoke(args.reference, data)) if eligible else 0
            score = scorer.run(data) if eligible and not compact and not overlap else None
            decision = 6 if limit is not None and accepted(score, limit, margin) else 0
            row[mode] = {"compact": compact, "combined": compact or decision,
                         "score": score, "overlap": overlap, "input_bytes": len(data), "eligible": eligible}
        cases.append(row)
    public = [row for row in cases if row["split"].startswith("public")]
    return {"guard_sha256": file_digest(args.guard), "summary": {"historical": summarize(cases),
            "public": summarize(public)}, "cases": cases}


def chm_replay(args, scorer):
    import duckdb
    from phoneme_chm150_lab import CORPUS_SHA256, SPEAKER_SPLIT_SHA256, decode, eligible_rows

    limit, margin, development_pcm = frozen_guard(args, scorer)
    if file_digest(args.corpus) != CORPUS_SHA256 or file_digest(args.speaker_split) != SPEAKER_SPLIT_SHA256:
        raise ValueError("Unexpected separate CHM150 corpus or speaker split")
    assignment = json.loads(args.speaker_split.read_text())["speaker_roles"]
    reserved = {speaker for speaker, role in assignment.items() if role == "heldout"}
    count = compact = omitted = 0
    added = []
    speakers = set()
    with duckdb.connect(config={"autoload_known_extensions": False, "autoinstall_known_extensions": False}) as db:
        for _, speaker, _, blob in eligible_rows(db, args.corpus, reserved):
            data = decode(args.ffmpeg, blob)
            if not eligible_pcm(data):
                omitted += 1
                continue
            if not development_pcm.isdisjoint(pcm_seals(data)):
                raise ValueError("Reserved CHM150 waveform intersects D6 threshold calibration")
            count += 1
            speakers.add(speaker)
            if int(invoke(args.reference, data)):
                compact += 1
            elif limit is not None:
                score = scorer.run(data)
                if accepted(score, limit, margin):
                    added.append({"pcm_sha256": sha256(data).hexdigest(), "score": score})
    return {"guard_sha256": file_digest(args.guard), "utterances": count, "speakers": len(speakers),
            "compact_activations": compact, "added_d6": len(added), "added": added, "omitted": omitted,
            "ffmpeg_sha256": file_digest(args.ffmpeg)}


def existing(value):
    return Path(value).resolve(strict=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    for name in ("root", "teacher", "student", "espeak", "features"):
        common.add_argument("--" + name, type=existing, required=True)
    common.add_argument("--guard", type=Path, required=True)
    stages = parser.add_subparsers(dest="stage", required=True)
    fitting = stages.add_parser("freeze", parents=[common])
    for name in ("human", "synthetic-index", "wav"):
        fitting.add_argument("--" + name, type=existing, required=True)
    evaluation = stages.add_parser("replay", parents=[common])
    for name in ("reference", "capture"):
        evaluation.add_argument("--" + name, type=existing, required=True)
    chm = stages.add_parser("replay-chm", parents=[common])
    for name in ("reference", "corpus", "speaker-split", "ffmpeg"):
        chm.add_argument("--" + name, type=existing, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    scorer = Scorer(args)
    prepared = {"freeze": prepare, "replay": replay, "replay-chm": chm_replay}[args.stage]
    print(json.dumps(prepared(args, scorer), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
