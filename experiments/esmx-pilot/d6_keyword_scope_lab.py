"""Screen complete D6 candidates using one shared-density acoustic search."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import argparse
import gzip
import json
from pathlib import Path
import re
import threading

import duckdb
import numpy as np
from pocketsphinx import Decoder

from capture_replay import pcm_from, recordings
from d6_acoustic_provenance import clock_reference
from phoneme_chm150_lab import CORPUS_SHA256, SPEAKER_SPLIT_SHA256, decode as wav_decode, eligible_rows
from time_lab import file_digest, invoke

VERBS = ("tira", "tirar", "tírame", "tíreme", "lanza", "lanzar", "aviéntate", "avienta", "échame", "tire")
TARGET = ("un dado de seis", "el dado de seis")
POSITIVE = tuple(f"{verb} {number}" for verb in VERBS for number in TARGET) + (
    "tienes que tirar un dado de seis", "puedes tirar un dado de seis",
    "me puedes tirar un dado de seis", "tirar un dado de seis caras", "tíreme un dado de seis por favor",
    "tira un dado de seis caras", "tírame un dado de seis por favor", "lanza un dado de seis lados",
    "tire un dado de seis por favor",
)
NEGATIVE = ("mejor no", "no tires", "no lances", "no avientes", "no tire", "ni tires", "no lo tires",
            "no quiero que", "cancela la tirada", "olvida la tirada", "olvídalo", "déjalo",
            "por cada", "cada estadística", "cada atributo", "dos dados", "tres dados", "varios dados",
            "cuántos dados", "se tiran", "si necesitas", "si necesito", "si quieres",
            "un dado de ocho", "un dado de diez", "un dado de doce", "un dado de veinte", "un dado de cien")
LOCAL = threading.local()

PAIRED = (
    "no tires un dado de seis", "no tire un dado de seis", "no tires el dado de seis",
    "no lances un dado de seis", "no avientes un dado de seis", "no quiero que tires un dado de seis",
    "no quiero tirar un dado de seis", "no me tires un dado de seis", "no me tire un dado de seis",
    "no vamos a tirar un dado de seis", "no hay que tirar un dado de seis",
    "mejor no tires un dado de seis", "mejor no tire un dado de seis",
    "tira un dado de seis mejor no", "tirar un dado de seis mejor no",
    "tire un dado de seis mejor no", "tira el dado de seis mejor no",
    "cancela la tirada de un dado de seis", "olvida la tirada de un dado de seis",
    "tira un dado de seis olvídalo", "tira un dado de seis por cada",
    "tirar un dado de seis por cada", "si necesitas tirar un dado de seis",
    "no tire usted un dado de seis", "no tire usted el dado de seis",
    "no me tire usted un dado de seis", "no tire usted un dado de seis por favor",
    "no lance un dado de seis", "no lance usted un dado de seis",
    "tira un dado de seis cancélalo", "aviente un dado de seis no olvídalo",
)
AFTER = {"mejor no", "olvídalo", "déjalo", "por cada", "cada estadística", "cada atributo",
         "dos dados", "tres dados", "varios dados", "cuántos dados", "cancélalo", "no olvídalo",
         "mejor olvídalo", "no mejor no"}
TAILS = {"por favor", "porfa", "seis caras", "seis lados"}
NEGATIVE = tuple(dict.fromkeys((*NEGATIVE, *PAIRED, *sorted(AFTER))))
SYNTHESIS_SHA256 = "25b7b0ab3425394ff191b39ad0d4db7049b5db33ee21eac0c9efb2033846ee60"
SUPPLEMENT_SHA256 = "b520af9599dc798dd78571ca06bbd3aaf50c405946aceed77f966e6e5cb686a6"
WITNESSES = {"public-03", "public-37", "public-81", "public-88", "public-89", "test-0379"}

def build_dictionary(source, output):
    words = set(" ".join((*POSITIVE, *NEGATIVE, *TAILS)).split())
    with gzip.open(source, "rt", encoding="utf-8") as original:
        lines = {row.split()[0]: row for row in original if row.strip() and row.split()[0] in words}
    additions = {"porfa": "p o_7 r( f a", "tíreme": "t i_7 r( e m e", "avientes": "a b i e_7 n t e s", "cancélalo": "k a n s e_7 l a l o"}
    for word, phones in additions.items():
        lines.setdefault(word, f"{word} {phones}\n")
    if words - lines.keys():
        raise ValueError("Missing words: " + " ".join(sorted(words - lines.keys())))
    output.write_text("".join(lines[word] for word in sorted(words)), encoding="utf-8")

def build_search(path):
    with path.open("x", encoding="utf-8") as file:
        for group, threshold in ((POSITIVE, 60), (sorted(TAILS), 24), (NEGATIVE, 56)):
            for phrase in group:
                file.write(f"{phrase} /1e-{threshold}/\n")


def span(pcm):
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.int64)
    pad = np.pad(samples, (0, -len(samples) % 160)).reshape(-1, 160)
    frames = np.flatnonzero(np.mean(pad * pad, axis=1) >= (180 * 180))
    return [int(frames[0]), int(frames[-1])] if len(frames) else None

def spot(args, pcm):
    if not hasattr(LOCAL, "decoder"):
        dec = Decoder(hmm=str(args.model), dict=str(args.output / "keywords.dict"),
                      lm=None, samprate=16000, loglevel="FATAL")
        dec.add_kws("d6", str(args.output / "search-60.kws"))
        dec.activate_search("d6")
        LOCAL.decoder = dec
    dec = LOCAL.decoder
    dec.start_utt()
    dec.process_raw(pcm, False, True)
    dec.end_utt()
    if not dec.hyp():
        return []
    return [(word.word.strip(), word.start_frame, word.end_frame,
             dec.get_logmath().log(word.prob)) for word in dec.seg()]

def inside_window(first, last, start, end, exterior):
    return start - first <= exterior and last - end <= exterior

def decide(view, positive_floor, negative_floor, exterior):
    limits = view["speech_span"]
    if not limits or view["baseline"] not in {"no_match", "dice_negation_conflict"}:
        return False
    first, last = limits
    positives = [hit for hit in view["hits"] if hit[0] in POSITIVE and hit[3] >= positive_floor]
    tails = [hit for hit in view["hits"] if hit[0] in TAILS and hit[3] >= -2200]
    paired = [hit for hit in view["hits"] if hit[0] in PAIRED and hit[3] >= negative_floor]
    after = [hit for hit in view["hits"] if hit[0] in AFTER and hit[3] >= negative_floor]
    for _, start, end, _ in positives:
        full_end = max([end] + [tail_end for _, tail_start, tail_end, _ in tails
                                if end - 45 <= tail_start <= end + 16 and tail_end - end <= 82])
        if not inside_window(first, last, start, full_end, exterior):
            continue
        if any(inside_window(first, last, begin, stop, exterior) for _, begin, stop, _ in paired):
            continue
        if any(begin >= end - 6 for _, begin, _, _ in after):
            continue
        return True
    return False

def measure(rows, setting, policy, view):
    accepted = [row for row in rows if (observation := row["views"].get(view))
                and (policy == "conflict" or observation["baseline"] == "no_match")
                and decide(observation, *setting)]
    return {"tp": sum(row["expected"] == 6 for row in accepted),
            "fp": sum(row["expected"] != 6 for row in accepted),
            "accepted": [row["id"] for row in accepted]}

def groups(rows):
    selected = {}
    for policy in ("no_match", "conflict"):
        calibration = [r for r in rows if r["group"] in ("synthetic_dev", "chm_dev")]
        scores = []
        for setting in product((-1900, -2050, -2200, -2350, -2500), (-1850, -2050, -2250, -2450), (20, 32, 48)):
            m = measure(calibration, setting, policy, "retained")
            if not m["fp"]:
                scores.append((m["tp"], setting))
        if not scores:
            selected[policy] = {"result": "no zero-addition setting on development"}
            continue
        correct, setting = max(scores, key=lambda item: (item[0], item[1][0], -item[1][1], -item[1][2]))
        selected[policy] = {"setting": setting, "positive_development": correct, "groups": {}}
        for kind, count in sorted(Counter(row["group"] for row in rows).items()):
            group = [row for row in rows if row["group"] == kind]
            selected[policy]["groups"][kind] = {"size": count, **{
                view: measure(group, setting, policy, view) for view in ("raw", "retained")
            }}
    return selected

def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "dictionary", "historical", "capture", "host", "output", "roles", "corpus", "ffmpeg", "synthetic", "supplement",
                 "baseline_evidence"):
        cli.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = cli.parse_args()
    if file_digest(args.corpus) != CORPUS_SHA256 or file_digest(args.roles) != SPEAKER_SPLIT_SHA256:
        raise ValueError("Corpus or speaker split differs")
    reference = clock_reference(args.baseline_evidence, args.model, args.host, args.capture)
    if args.output.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError("Keep per-turn records outside the repository")
    args.output.mkdir(mode=0o700)
    build_dictionary(args.dictionary, args.output / "keywords.dict")
    build_search(args.output / "search-60.kws")
    history = [(uid, split, expected, pcm_from(wav, seal))
                for uid, split, _, expected, wav, seal in recordings(args.historical)]
    synthesis = args.synthetic / "synthesis.json"
    if file_digest(synthesis) != SYNTHESIS_SHA256:
        raise ValueError("Unexpected previous synthesis seal")
    voices = []
    for case in json.loads(synthesis.read_text()):
        split, name = case["split"], case["id"]
        if split in {"dev", "test"}:
            if re.fullmatch(r"(dev|test)-[0-9]{2}-[01]-[01]", name) is None:
                raise ValueError("Unexpected archived synthesis name")
            group = "synthetic_dev" if split == "dev" else "synthetic_new_test"
            voices.append((name, group, 6 if case["d6"] else 0,
                           pcm_from(args.synthetic / (name + ".wav"), case["sha256"])))
    if len(voices) != 40:
        raise ValueError("Expected 16 development and 24 test turn examples")
    if file_digest(args.supplement / "seal.json") != SUPPLEMENT_SHA256:
        raise ValueError("Unexpected new synthesis seal")
    supplement = json.loads((args.supplement / "seal.json").read_text())
    if len(supplement) != 24:
        raise ValueError("Expected 24 separate synthetic controls")
    for case in supplement:
        name = case["file"]
        if re.fullmatch(r"target-[0-9]{2}-[01]-[01]\.wav", name) is None:
            raise ValueError("Unexpected synthetic file name")
        voices.append((name.removesuffix(".wav"), "supplement", case["expected"],
                       pcm_from(args.supplement / name, case["sha256"])))
    roles = json.loads(args.roles.read_text())["speaker_roles"]
    speakers = {speaker for speaker, role in roles.items() if role != "train"}
    with duckdb.connect(config={"autoload_known_extensions": False, "autoinstall_known_extensions": False}) as conn:
        chm = [(uid, "chm_" + roles[speaker], 0, wav)
               for uid, speaker, _, wav in eligible_rows(conn, args.corpus, speakers)]
    def evaluate(entry):
        uid, group, expected, raw = entry
        if group.startswith("chm_"):
            raw = wav_decode(args.ffmpeg, raw)
        hold = invoke(args.capture, raw, "--pcm-with-hangover")
        result = {"id": uid, "group": group, "expected": expected, "views": {}}
        for name, pcm in (("raw", raw), ("retained", hold)):
            if not 0 < len(pcm) <= 128000:
                continue
            decision = json.loads(invoke(args.host, pcm, "--decision"))
            existing = decision["intent"] or int(invoke(args.host, pcm))
            eligible = existing == 0 and decision["disposition"] in {"no_match", "dice_negation_conflict"}
            result["views"][name] = {
                "baseline": decision["disposition"], "label": existing, "search_ran": eligible,
                "speech_span": span(pcm),
                "hits": spot(args, pcm) if eligible else [],
            }
        return result
    with ThreadPoolExecutor(max_workers=8) as runner:
        rows = list(runner.map(evaluate, (*history, *voices, *chm)))
    (args.output / "replay.local.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    results = groups(rows)
    for measurement in results.values():
        for name, group in measurement.get("groups", {}).items():
            if name.startswith("chm_"):
                for view in ("raw", "retained"):
                    del group[view]["accepted"]
    report = {"baseline": reference, "recipe_sha256": file_digest(Path(__file__)),
              "dictionary_sha256": file_digest(args.output / "keywords.dict"),
              "search_sha256": file_digest(args.output / "search-60.kws"),
              "calibration_seals": {"speaker_split": SPEAKER_SPLIT_SHA256, "synthesis": SYNTHESIS_SHA256},
              "supplement_seal_sha256": SUPPLEMENT_SHA256,
              "evaluation": results,
              "known_cases": {row["id"]: {view: {key: info[key] for key in ("baseline", "label", "search_ran")}
                                           for view, info in row["views"].items()}
                              for row in rows if row["id"] in WITNESSES}}
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
