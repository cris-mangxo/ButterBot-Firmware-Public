"""Compare a locally supplied CIEMPIESS model and compact acoustic reference on desktop."""
import argparse
from pathlib import Path
import gzip
import json
import re
import shutil
import struct

import numpy as np

from time_lab import digest, file_digest
from clock_elastic_lab import example_id


def read_s3(path, dimensions):
    buf = path.read_bytes()
    header, contents = buf.split(b"endhdr\n", 1)
    if not header.startswith(b"s3\nversion 1.0\n") or "chksum0 yes" not in header.decode("ascii"):
        raise ValueError("Unknown S3 acoustic representation")
    if struct.unpack_from("<I", contents)[0] != 0x11223344:
        raise ValueError("Unexpected S3 endian")
    geometry = struct.unpack_from(f"<{dimensions}I", contents, 4)
    size = geometry[-1]
    start = 4 + 4 * dimensions
    if len(contents) != start + 4 * size + 4:
        raise ValueError("Truncated model")
    return geometry[:-1], np.frombuffer(contents, dtype="<f4", count=size, offset=start)


GRAMMAR = """
#JSGF V1.0 UTF-8;
grammar robot;
public <utterance> = <clock> | <agenda> | <other>;
<clock> = [oye | disculpa | dime | me puedes decir] (qué hora es [ahorita | ahora] | qué horas son | qué hora tienes | dame la hora | dime la hora);
<agenda> = [oye | disculpa | sabes | dime] (a qué horas | a qué hora) [es | sale | llega | empieza | viene]
           [el avión | el vuelo | la cena | el examen | la reunión | el partido | la clase | la cita | el camión] |
           (qué hora es | qué horas son) (de la cita | en madrid | en valencia | de la reunión | mañana) |
           qué hora era [antes | ayer] | no me digas [qué hora es | la hora] | es hora de [comer | salir];
<other> = (tira | avienta | lanza) [un] dado de (seis | diez | veinte) [caras | lados] |
           (dime | échame | cuéntame) un chiste | (dame | pásame) [la] mantequilla |
           qué es eso | es temprano | no sé | [son | es] [la | las] (una | dos | tres | cuatro | cinco | seis | siete | ocho | nueve | diez | once | doce) [y media];
""".strip()


ACOUSTIC_FILES = {"mdef", "feat.params", "noisedict", "transition_matrices",
                  "means", "variances", "mixture_weights"}


def write_s3(path, dimensions, values):
    header = b"s3\nversion 1.0\nendhdr\n"
    values = np.asarray(values, dtype="<f4")
    geometry = struct.pack(f"<{2 + len(dimensions)}I", 0x11223344, *dimensions, values.size)
    path.write_bytes(header + geometry + values.tobytes())


def quantized(dimensions, values):
    feature_count = dimensions[-1]
    matrix = values.reshape(-1, feature_count)
    low = matrix.min(axis=0)
    scale = np.maximum(matrix.max(axis=0) - low, 1e-7) / 255
    quantized = np.clip(np.rint((matrix - low) / scale), 0, 255).astype("u1")
    return low, scale, quantized


def reduce(source, target, densities=None, quantize=False):
    rows = (source / "mdef").read_text().splitlines()
    counts = dict((label, int(number)) for row in rows[:8] for number, label in re.findall(r"^(\d+) (\w+)", row))
    base, full_states, ci_states = (counts[key] for key in ("n_base", "n_tied_state", "n_tied_ci_state"))
    unit = [row for row in rows[8:] if len(row.split()) >= 10 and row.split()[1:4] == ["-", "-", "-"]]
    if len(unit) != base or sorted(int(n) for row in unit for n in row.split()[6:9]) != list(range(ci_states)):
        raise ValueError("Model CI states are not the expected leading slice")
    target.mkdir(mode=0o700)
    (target / "mdef").write_text(f"0.3\n{base} n_base\n0 n_tri\n{base*4} n_state_map\n"
                                 f"{ci_states} n_tied_state\n{ci_states} n_tied_ci_state\n"
                                 f"{counts['n_tied_tmat']} n_tied_tmat\n" + "\n".join(unit) + "\n")
    for name in ("feat.params", "noisedict", "transition_matrices"):
        shutil.copyfile(source / name, target / name)
    (states, streams, components), mixtures = read_s3(source / "mixture_weights", 4)
    if (states, streams) != (full_states, 1):
        raise ValueError("Unexpected mixture geometry")
    mixtures = mixtures.reshape(full_states, components)[:ci_states]
    if densities is None:
        densities = components
    if not 1 <= densities <= components:
        raise ValueError("Invalid component count")
    keep = np.sort(np.argsort(-mixtures, axis=1, kind="stable")[:, :densities], axis=1)
    write_s3(target / "mixture_weights", (ci_states, 1, densities),
             np.take_along_axis(mixtures, keep, axis=1))
    evidence = {}
    for name in ("means", "variances"):
        (states, streams, n_density, veclen), values = read_s3(source / name, 5)
        if (states, streams, n_density) != (full_states, 1, components):
            raise ValueError("Unexpected Gaussian geometry")
        floats = values.reshape(full_states, components, veclen)[:ci_states]
        selected = np.take_along_axis(floats, keep[:, :, None], axis=1)
        dimensions = (ci_states, 1, densities, veclen)
        if quantize:
            low, scale, codes = quantized(dimensions, selected)
            restored = low + scale * codes
            packed = target / (name + ".uint8")
            packed.write_bytes(low.astype("<f4").tobytes() + scale.astype("<f4").tobytes() + codes.tobytes())
            evidence[name] = {"parameter_count": codes.size, "bytes": packed.stat().st_size,
                              "sha256": file_digest(packed),
                              "max_absolute_error": float(abs(restored-selected.reshape(restored.shape)).max())}
            selected = restored
        write_s3(target / name, dimensions, selected)
    return evidence


def acoustic_seal(directory):
    return {name: file_digest(directory / name) for name in sorted(ACOUSTIC_FILES)}


def replay(args):
    grammar = re.sub(r"<[^>]+>", " ", GRAMMAR.split("public")[1])
    words = set(re.findall(r"[a-záéíóúñ]+", grammar))
    dictionary = args.out / "subset.dict"
    with gzip.open(args.dictionary, "rt", encoding="utf-8") as source:
        lines = [s for s in source if s.strip().split() and re.sub(r"\([0-9]+\)$", "", s.split()[0]) in words]
    missing = words - {row.split()[0] for row in lines}
    if missing:
        raise ValueError(f"Missing pronunciation: {sorted(missing)}")
    dictionary.write_text("".join(lines))
    from pocketsphinx import Decoder

    decoder = Decoder(hmm=str(args.acoustic), dict=str(dictionary), lm=None, samprate=16000, loglevel="ERROR")
    decoder.add_jsgf_string("clock", GRAMMAR)
    decoder.activate_search("clock")
    panel = json.loads(args.panel.read_text())
    quality = json.loads(args.quality.read_text())["labels"]
    records = []
    for row in panel["results"]:
        identifier = example_id(row["id"], r"[CT][0-9]{2}-[0-9]{3}")
        report = {"id": identifier, "expected": row["expected"],
                  "confirmed": quality[identifier]["include_in_confirmed_denominator"]}
        for partition in ("raw", "retained"):
            pcm = (args.pcm / f"{identifier}.{partition}.pcm").read_bytes()
            if digest(pcm) != row[partition]["pcm_sha256"]:
                raise ValueError("Unsealed sample")
            decoder.start_utt()
            decoder.process_raw(pcm, False, True)
            decoder.end_utt()
            hyp = decoder.hyp()
            reported = hyp.hypstr if hyp else ""
            report[partition] = {"hyp": reported, "prob": hyp.prob if hyp else None,
                                 "score": hyp.best_score if hyp else None,
                                 "pilot": row[partition]["decision"]}
        records.append(report)
    print(json.dumps({"grammar_sha256": digest(GRAMMAR.encode()), "subset_sha256": file_digest(dictionary),
                      "replay_input_sha256": file_digest(args.panel), "quality_sha256": file_digest(args.quality),
                      "model_files": acoustic_seal(args.acoustic),
                      "records": records}, indent=2,
                     ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="stage", required=True)
    compress = modes.add_parser("reduce")
    compress.add_argument("source", type=Path)
    compress.add_argument("target", type=Path)
    compress.add_argument("--densities", type=int)
    compress.add_argument("--uint8", action="store_true",
                          help="Store packed arrays and emit dequantized S3 files for the desktop decoder")
    run = modes.add_parser("replay")
    for name in ("acoustic", "dictionary", "out", "panel", "quality", "pcm"):
        run.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "reduce":
        print(json.dumps(reduce(args.source, args.target, args.densities, args.uint8), indent=2))
    else:
        replay(args)
