"""Replay a narrow D6 grammar against existing dispositions and Mexican conversation."""

from concurrent.futures import ThreadPoolExecutor
import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import re
import threading

import duckdb
from pocketsphinx import Decoder

from capture_replay import pcm_from, recordings
from phoneme_chm150_lab import CORPUS_SHA256, SPEAKER_SPLIT_SHA256, decode, eligible_rows
from time_lab import file_digest, invoke


GRAMMAR = """
#JSGF V1.0 UTF-8;
grammar roll;
public <turn> = <positive> | <negative> | <background>;
<please> = [y | oye | oiga | a ver | por favor];
<roll> = tira | lanza | avienta | échame | tírame | tíreme | tire | aviéntate | tirar | lanzar | aventar | aviente | échate;
<d6> = un dado de seis [caras | lados] | un de seis | el dado de seis;
<request> = [me puedes | puedes | necesito que | quiero que | tienes que | vas a | me puede] <roll> <d6>;
<positive> = <please> <request> [por favor | porfa];
<negative> = [mejor | oye | a ver | y] no [quiero que | debes | me | hace falta que] (tires | lances | tire | <roll>) <d6> |
             <please> <request> [pero] (no | mejor no | cancela | olvídalo) |
             (cancela | olvida) [la tirada de] <d6> |
             <please> <roll> (un dado | dos dados | el dado) de (cuatro | ocho | diez | doce | veinte | cien)
             [caras | lados] | <please> <roll> (dos | tres) dados de seis [caras | lados];
<background> = (qué hora es | a qué hora es | cuéntame un chiste | dime un chiste | pásame la mantequilla |
                qué es eso | no sé | no quiero | dime un dato curioso | dime un dato | lanza un dado |
                tira un dado | con un dado de seis | elige un dado de seis | un dado de seis);
""".strip()

POSITIVE = re.compile(
    r"(?:(?:y|oye|oiga|a ver|por favor) )?"
    r"(?:(?:me puedes|puedes|necesito que|quiero que|tienes que|vas a|me puede) )?"
    r"(?:tira|lanza|avienta|échame|tírame|tíreme|tire|aviéntate|tirar|lanzar|aventar|aviente|échate) "
    r"(?:un dado de seis(?: caras| lados)?|un de seis|el dado de seis)"
    r"(?: por favor| porfa)?"
)
THREAD = threading.local()
PRONUNCIATION_ADDITIONS = {"tíreme": "t i_7 r( e m e", "porfa": "p o_7 r( f a"}


def prepare_dictionary(source, output):
    body = re.sub(r"<[^>]+>", " ", GRAMMAR.split("public", 1)[1])
    words = set(re.findall(r"[a-záéíóúñ]+", body))
    with gzip.open(source, "rt", encoding="utf-8") as original:
        lines = [row for row in original if row.strip()
                 and re.sub(r"\([0-9]+\)$", "", row.split()[0]) in words]
    missing = words - {row.split()[0] for row in lines}
    if missing - PRONUNCIATION_ADDITIONS.keys():
        raise ValueError("Source dictionary lacks grammar words: " + ", ".join(sorted(missing)))
    lines.extend(f"{word} {PRONUNCIATION_ADDITIONS[word]}\n" for word in sorted(missing))
    with output.open("x", encoding="utf-8") as target:
        target.writelines(lines)

def decide(args, pcm):
    if not hasattr(THREAD, "decoder"):
        d = Decoder(hmm=str(args.model), dict=str(args.dictionary), lm=None, samprate=16000, loglevel="FATAL")
        d.add_jsgf_string("d6", GRAMMAR)
        d.activate_search("d6")
        THREAD.decoder = d
    d = THREAD.decoder
    d.start_utt()
    d.process_raw(pcm, False, True)
    d.end_utt()
    hyp = d.hyp()
    if hyp is None:
        return {"path": "", "strict": False}
    return {"path": hyp.hypstr, "strict": bool(POSITIVE.fullmatch(hyp.hypstr)),
            "frames": d.n_frames(), "log_probability": d.get_logmath().log(hyp.best_score),
            "log_posterior": d.get_logmath().log(hyp.prob),
            "segments": [[seg.word, seg.start_frame, seg.end_frame] for seg in d.seg()]}

def row(args, key, expected, pcm, source):
    result = {"id": key, "expected": expected, "source": source, "views": {}}
    for view, data in (("raw", pcm), ("retained", invoke(args.capture, pcm, "--pcm-with-hangover"))):
        if not 0 < len(data) <= 128000:
            continue
        before = json.loads(invoke(args.host, data, "--decision"))
        selected = int(invoke(args.host, data))
        search = decide(args, data) if before["allows_fallback"] and selected == 0 else {}
        result["views"][view] = dict(before, selected=selected, search=search)
    result["would_add"] = len(result["views"]) == 2 and all(
        item["search"].get("strict", False) for item in result["views"].values())
    return result

def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "source_dictionary", "historical", "capture", "host", "roles", "corpus", "ffmpeg", "output"):
        cli.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = cli.parse_args()
    acoustic_clock = Path(os.environ["BB_ESMX_CLOCK_ACOUSTIC_DIR"])
    if not (acoustic_clock / "clock.dict").is_file():
        raise ValueError("Enable the existing acoustic clock profile for the baseline")
    if file_digest(args.roles) != SPEAKER_SPLIT_SHA256 or file_digest(args.corpus) != CORPUS_SHA256:
        raise ValueError("CHM150 dataset or speaker split differs from the sealed panel")
    args.output.mkdir(mode=0o700)
    args.dictionary = args.output / "d6-reduced.dict"
    prepare_dictionary(args.source_dictionary, args.dictionary)
    corpus_roles = json.loads(args.roles.read_text())["speaker_roles"]
    history = ((key, expected, pcm_from(wav, pin), split)
               for key, split, _, expected, wav, pin in recordings(args.historical))
    with duckdb.connect(config={"autoload_known_extensions": False, "autoinstall_known_extensions": False}) as con:
        speakers = {speaker for speaker, role in corpus_roles.items() if role != "train"}
        chm = list(eligible_rows(con, args.corpus, speakers))
    samples = list(history)
    for ident, speaker, _, wav in chm:
        samples.append((ident, 0, wav, "chm_" + corpus_roles[speaker]))
    def evaluate(item):
        key, expected, pcm, source = item
        if source.startswith("chm_"):
            pcm = decode(args.ffmpeg, pcm)
        return row(args, key, expected, pcm, source)
    with ThreadPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(evaluate, samples))
    report = {"host_sha256": file_digest(args.host), "capture_sha256": file_digest(args.capture),
              "dictionary_sha256": file_digest(args.dictionary), "recipe_sha256": file_digest(Path(__file__)),
              "acoustic_files_sha256": {name: file_digest(args.model / name) for name in
                                        ("mdef", "means", "variances", "mixture_weights", "transition_matrices",
                                         "feat.params", "noisedict")},
              "rows": rows}
    (args.output / "replay.local.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    summaries = {}
    for source, count in sorted(Counter(item["source"] for item in rows).items()):
        records = [item for item in rows if item["source"] == source]
        summaries[source] = {"clips": count}
        for view in ("raw", "retained", "both"):
            additions = [item for item in records if (
                item["would_add"] if view == "both" else item["views"].get(view, {}).get("search", {}).get("strict", False)
            )]
            summaries[source][view] = {"expected_d6": sum(item["expected"] == 6 for item in additions),
                                       "false_positives": sum(item["expected"] != 6 for item in additions)}
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
