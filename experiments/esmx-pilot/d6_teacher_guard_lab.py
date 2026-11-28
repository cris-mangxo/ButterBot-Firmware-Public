"""Calibrate and replay a desktop D6-only reference with full-vocabulary CTC."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch

from capture_replay import pcm_from, recordings
import keyword_lab as kw
from phoneme_confirm_lab import local_ctc_alignment, local_ctc_cost, model_output
from phoneme_replay_lab import frozen_model, native_frames, strip_digital_edges
from phoneme_teacher_lab import WEIGHTS_SHA256, collapse, load_teacher, phonemes, prepared_pcm
from time_lab import file_digest, invoke


SCHEMA = "bb-d6-full-ctc-reference-1"
SCOPE_SCHEMA = "bb-d6-turn-scope-1"
SCOPE_CASES = Path(__file__).with_name("d6_turn_scope_cases.json")
SCOPE_BASE_SHA256 = "46878f2616f4c8ba18574e460d62190ab716a9f0e4c3d2948808c1de3f4deded"
SCOPE_VOICES = {
    "development": (("Grandpa (Español (México))", 178), ("Shelley (Español (México))", 190)),
    "test": (("Reed (Español (México))", 174), ("Grandma (Español (México))", 185)),
}
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


def exterior_phone_counts(greedy, start, end):
    # A nonblank run that crosses an interval edge is one CTC phone.
    while start > 0 and greedy[start] != 0 and greedy[start - 1] == greedy[start]:
        start -= 1
    while end < len(greedy) and greedy[end - 1] != 0 and greedy[end] == greedy[end - 1]:
        end += 1
    return len(collapse(greedy[:start].tolist())), len(collapse(greedy[end:].tolist()))


def observation(scores, templates, negated, track_scope=False):
    if track_scope:
        command, interval = min((local_ctc_alignment(scores, seq, track_span=True)
                                for seq in templates["order"]), key=lambda item: item[0])
    else:
        command = min(local_ctc_cost(scores, seq) for seq in templates["order"])
    costs = {"order": command}
    costs.update({kind: min(local_ctc_cost(scores, seq) for seq in options)
                  for kind, options in templates.items() if kind != "order"})
    if not all(isfinite(value) for value in costs.values()):
        return None
    competitor = min(value for key, value in costs.items() if key != "order")
    result = {"cost": command, "margin": competitor - command, "veto": negated, "costs": costs}
    if track_scope:
        start, end = interval
        greedy = scores.argmax(axis=1)
        before, after = exterior_phone_counts(greedy, start, end)
        result["scope"] = {"before": before, "after": after, "match": [start, end], "frames": len(greedy)}
    return result


def accepted(item, limit, margin):
    return bool(item and limit is not None and not item["veto"]
                and item["cost"] <= limit and item["margin"] >= margin)


def eligible_pcm(data):
    return 0 < len(data) <= 128_000 and len(data) % 2 == 0


def outside_phones(item):
    return item["scope"]["before"] + item["scope"]["after"]


def scope_cap(base_passes):
    controls = [outside_phones(row["score"]) for row in base_passes if not row["positive"]]
    if controls:
        first_control = min(controls)
        positives = [outside_phones(row["score"]) for row in base_passes
                     if row["positive"] and outside_phones(row["score"]) < first_control]
        if positives:
            # Balance clearance from the last retained order and first control.
            return (max(positives) + first_control - 1) // 2
    return None


def permitted(item, limit, margin, scope):
    if not accepted(item, limit, margin):
        return False
    if scope is None:
        return True
    ceiling = scope["outside_ceiling"]
    return ceiling is not None and outside_phones(item) <= ceiling


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
        self.track_scope = args.stage in ("freeze-scope", "replay-scope-test") or bool(getattr(args, "scope_guard", None))
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
        return observation(logits, self.templates, bool(denial.item() >= 0), self.track_scope)


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


def scope_cases(role, scorer):
    data = json.loads(SCOPE_CASES.read_text())[role]
    with TemporaryDirectory(prefix="bb-scope-esmx-") as directory:
        for kind, phrases in data.items():
            for number, phrase in enumerate(phrases):
                for variant, (voice, rate) in enumerate(SCOPE_VOICES[role]):
                    identifier = f"scope-{role}-{kind}-{number:02}-{variant}"
                    wav = kw.render(Path(directory), identifier, voice, phrase, rate)
                    pcm = pcm_from(wav)
                    if not eligible_pcm(pcm):
                        raise ValueError(f"Scope fixture is overlong or invalid: {identifier}")
                    yield {"id": identifier, "positive": kind == "orders", "score": scorer.run(pcm),
                           "pcm_sha256": pcm_seals(pcm), "input_bytes": len(pcm)}


def scope_seal(args):
    if file_digest(args.guard) != SCOPE_BASE_SHA256:
        raise ValueError("Turn scope requires the reviewed D6 reference guard")
    return {"base_guard_sha256": SCOPE_BASE_SHA256, "fixture_sha256": file_digest(SCOPE_CASES),
            "voice_roles": {role: [list(item) for item in voices] for role, voices in SCOPE_VOICES.items()}}


def scope_results(rows, limit, margin, scope):
    outcomes = Counter((row["positive"], accepted(row["score"], limit, margin),
                        permitted(row["score"], limit, margin, scope)) for row in rows)
    return [[*key, total] for key, total in sorted(outcomes.items())]


def freeze_scope(args, scorer):
    seal = scope_seal(args)
    limit, margin, previous = frozen_guard(args, scorer)
    data = list(scope_cases("development", scorer))
    if any(not previous.isdisjoint(row["pcm_sha256"]) for row in data):
        raise ValueError("Context development intersects the existing D6 limit fit")
    passed = [row for row in data if accepted(row["score"], limit, margin)]
    report = {"schema": SCOPE_SCHEMA, "seal": seal, "outside_ceiling": scope_cap(passed), "development": data}
    report["counts"] = scope_results(data, limit, margin, report)
    with args.scope_guard.open("x") as output:
        output.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return {"scope_guard_sha256": file_digest(args.scope_guard), "outside_ceiling": report["outside_ceiling"],
            "counts": report["counts"]}


def checked_scope(args, development_pcm, limit, margin):
    if not args.scope_guard:
        return None
    data = json.loads(args.scope_guard.read_text())
    if data["schema"] != SCOPE_SCHEMA or data["seal"] != scope_seal(args):
        raise ValueError("Scope guard provenance changed")
    cap = data["outside_ceiling"]
    if cap is not None and (type(cap) is not int or cap < 0):
        raise ValueError("Invalid exterior phone limit")
    passed = [row for row in data["development"] if accepted(row["score"], limit, margin)]
    if cap != scope_cap(passed):
        raise ValueError("Exterior phone limit differs from its development data")
    development_pcm.update(checksum for row in data["development"] for checksum in row["pcm_sha256"])
    return data


def replay_scope_test(args, scorer):
    limit, margin, adjustment = frozen_guard(args, scorer)
    scope = checked_scope(args, adjustment, limit, margin)
    data = list(scope_cases("test", scorer))
    if any(not adjustment.isdisjoint(row["pcm_sha256"]) for row in data):
        raise ValueError("Synthetic scope test overlaps development")
    return {"scope_guard_sha256": file_digest(args.scope_guard), "counts": scope_results(data, limit, margin, scope),
            "cases": data}


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
    scope = checked_scope(args, development_pcm, limit, margin)
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
            decision = 6 if permitted(score, limit, margin, scope) else 0
            row[mode] = {"compact": compact, "combined": compact or decision,
                         "score": score, "overlap": overlap, "input_bytes": len(data), "eligible": eligible}
        cases.append(row)
    public = [row for row in cases if row["split"].startswith("public")]
    scope_seal_value = {"scope_guard_sha256": file_digest(args.scope_guard)} if scope else {}
    return {"guard_sha256": file_digest(args.guard), **scope_seal_value, "summary": {"historical": summarize(cases),
            "public": summarize(public)}, "cases": cases}


def chm_replay(args, scorer):
    import duckdb
    from phoneme_chm150_lab import CORPUS_SHA256, SPEAKER_SPLIT_SHA256, decode, eligible_rows

    limit, margin, development_pcm = frozen_guard(args, scorer)
    scope = checked_scope(args, development_pcm, limit, margin)
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
                if permitted(score, limit, margin, scope):
                    added.append({"pcm_sha256": sha256(data).hexdigest(), "score": score})
    scope_seal_value = {"scope_guard_sha256": file_digest(args.scope_guard)} if scope else {}
    return {"guard_sha256": file_digest(args.guard), **scope_seal_value, "utterances": count, "speakers": len(speakers),
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
    for command in (evaluation, chm):
        command.add_argument("--scope-guard", type=existing)
    stages.add_parser("freeze-scope", parents=[common]).add_argument("--scope-guard", type=Path, required=True)
    stages.add_parser("replay-scope-test", parents=[common]).add_argument("--scope-guard", type=existing, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    scorer = Scorer(args)
    prepared = {"freeze": prepare, "replay": replay, "replay-chm": chm_replay,
                "freeze-scope": freeze_scope, "replay-scope-test": replay_scope_test}[args.stage]
    print(json.dumps(prepared(args, scorer), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
