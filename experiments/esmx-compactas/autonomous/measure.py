"""Measure local C++ D6 inference without keeping public audio or transcripts."""

import argparse
from collections import Counter
import json
from pathlib import Path, PurePosixPath
import re
import subprocess

from ald_round import ALD_SHA256, BASELINE_SHA256, canonical, digest
from common_voice import DATASETS
from d6_guard import LOCAL_CV_SHA256, guarded_score
from phrases import too_long
from phrases_human import group, read_tables, select, selected_wavs
from pilot import MAX_COST, MIN_GAP, read_pcm, score
from public_orders import FIXTURE as PUBLIC_FIXTURE, HELDOUT_FIXTURES, add_panel_options, specification, verified_cases
import whole_orders

BANK_SHA256 = "ca40780ad1ee096f211aa59d3d1d7e410bbeb3f2879c0c3f23e76633e589d7ca"
CONTRAST_BANK_SHA256 = "7d5a53c1c98f4c74d45ac6984e223c41dd8d79f5e61558a9b19cf2459a05c8d8"
CONTRAST_WORD_SHA256 = "bc4dab7f7dcacfebc53d641712429c1e5c0d409fe9fab71becb43d768ecc801d"
EXPLICIT_BANK_SHA256 = "53dc9e084b543190312e3a03fbe527b3a358fe8cb1ead3d154487ff079739f8f"
FIXTURE = Path(__file__).with_name("fixture.json")
VARIANTS = ("compact", "turn_cnn", "word_gate")


def trial(args, wav, category, expected, identifier=None):
    long = too_long(wav)
    variants = VARIANTS + (("word_pair",) if args.extra else ()) + (("word_adapt",) if args.adapted else ())
    if args.continuous:
        variants += ("word_full", "word_guarded")
    if args.rescue:
        variants += ("word_rescue",)
    if args.contrast:
        variants += ("word_contrast",)
    data = {"category": category, "expected": expected, "overlong": long,
            "baseline": 0, "decisions": dict.fromkeys(variants, 0)}
    if identifier is not None:
        data["id"] = identifier
    if long:
        return data
    data["baseline"] = score(args.program, args.baseline, wav)["label"]
    compact = guarded_score(args.program, args.baseline, args.prior, wav, mode="phrases")
    data["decisions"]["compact"] = compact["label"]
    commands = [("turn_cnn", "acoustic", [args.turn]), ("word_gate", "words", [args.words])]
    if args.extra:
        commands.append(("word_pair", "word-pair", [args.words, args.extra]))
    if args.adapted:
        commands.append(("word_adapt", "word-adapt", [args.words, args.extra, args.adapted]))
    if args.continuous:
        banks = [args.words, args.extra, args.adapted, args.continuous]
        commands.extend([("word_full", "word-full", banks), ("word_guarded", "word-guarded", banks)])
        if args.rescue:
            commands.append(("word_rescue", "word-rescue", banks))
        if args.contrast:
            commands.append(("word_contrast", "word-contrast", banks))
    for variant, mode, models in commands:
        command = [args.program, mode, args.baseline, str(MAX_COST), str(MIN_GAP), args.prior, *models]
        process = subprocess.run(command, input=read_pcm(wav), capture_output=True, timeout=10)
        if process.returncode == 3:
            if compact["label"]:
                raise ValueError("Una orden aceptada quedó sin firma")
            continue
        process.check_returncode()
        result = json.loads(process.stdout)
        if result["label"] == 6 and (result.get("scenario"), result.get("data")) != (7, 2):
            raise ValueError("D6 no conserva el escenario del fabricante")
        if (data["baseline"] or compact["label"]) and result["label"] != compact["label"]:
            raise ValueError("Se alteró una decisión anterior o un veto")
        data["decisions"][variant] = result["label"]
    if args.extra and data["decisions"]["word_gate"] == 6 and data["decisions"]["word_pair"] != 6:
        raise ValueError("El banco nuevo perdió un D6 ya reconocido")
    if args.adapted and data["decisions"]["word_adapt"] != 6 and data["decisions"]["word_pair"] == 6:
        raise ValueError("El banco adaptado perdió un D6 de los dos bancos anteriores")
    if args.continuous:
        decisions = data["decisions"]
        prior = decisions["word_adapt"]
        if prior and (decisions["word_full"], decisions["word_guarded"]) != (prior, prior):
            raise ValueError("El detector continuo alteró una orden ya reconocida")
        if decisions["word_guarded"] and decisions["word_guarded"] != decisions["word_full"]:
            raise ValueError("La confirmación añadió una activación sin propuesta de la CNN")
        if args.rescue and ((decisions["word_guarded"] and decisions["word_rescue"] != decisions["word_guarded"])
                            or (decisions["word_rescue"] and decisions["word_rescue"] != decisions["word_full"])):
            raise ValueError("El rescate perdió una decisión anterior o añadió una tirada sin propuesta")
        if args.contrast and decisions["word_rescue"] and decisions["word_contrast"] != decisions["word_rescue"]:
            raise ValueError("El contraste perdió una decisión de rescate")
    return data


def summarize(rows):
    output = {}
    for row in rows:
        variants = tuple(row["decisions"])
        current = output.setdefault(row["category"], {
            "trials": 0, "overlong": 0, "evaluated": 0, "correct": dict.fromkeys(variants, 0),
            "false_d6": dict.fromkeys(variants, 0),
        })
        current["trials"] += 1
        current["overlong"] += row["overlong"]
        if row["overlong"]:
            continue
        current["evaluated"] += 1
        for model, decision in row["decisions"].items():
            current["correct"][model] += decision == row["expected"]
            current["false_d6"][model] += decision == 6 and row["expected"] == 0
    return output


def manifest(folder, fixture):
    info = json.loads((folder / "manifest.json").read_text("utf-8"))
    if info["fixture_sha256"] != digest(fixture) or info["ald_sha256"] != ALD_SHA256:
        raise ValueError("El corpus Ald no coincide con la referencia")
    rows = info["samples"]
    names = {row["id"] for row in rows}
    if len(names) != len(rows) or any(not re.fullmatch(r"[a-z]+-[0-9]{3,4}(?:-[0-9])?", token)
                                     for token in names):
        raise ValueError("Identificador de caso inválido")
    return rows


def real_trials(args):
    dataset, bytes_, catalog = DATASETS["female"]
    sha = digest(args.archive)
    if args.archive.stat().st_size != bytes_ or sha not in {catalog, LOCAL_CV_SHA256}:
        raise ValueError("Archivo Common Voice desconocido")
    prefix, rows, counts = read_tables(args.archive)
    early, _, inventory = select(rows, fresh=False)
    later, _, _ = select(rows, fresh=True)
    if early.keys() & later.keys():
        raise ValueError("Los dos cortes comparten clips")
    paths = {**{name: f"previous_long_first/{category}" for name, category in early.items()},
             **{name: f"previous_short_first/{category}" for name, category in later.items()}}
    speakers = set()
    used_texts = set()
    for row in rows:
        if row["path"] in paths:
            speakers.add(row["client_id"])
            used_texts.add(canonical(row["sentence"]))
    if len(speakers) != len(paths):
        raise ValueError("Voces duplicadas en el banco anterior")

    new = Counter()
    for row in sorted(rows, key=lambda r: (len(canonical(r["sentence"]).split()), r["path"])):
        path, speaker = row["path"], row["client_id"]
        text = canonical(row["sentence"])
        words = set(text.split())
        category = group(words)
        if (new[category] >= 18 or not 3 <= len(words) <= 12 or not speaker or speaker in speakers
                or path in paths or text in used_texts or PurePosixPath(path).name != path
                or not path.endswith(".mp3")):
            continue
        paths[path] = f"new_holdout/{category}"
        new[category] += 1
        speakers.add(speaker)
        used_texts.add(text)
    samples = [trial(args, wav, category, 0)
               for category, wav in selected_wavs(args.archive, prefix, paths)]
    return samples, {
        "dataset_id": dataset, "archive_sha256": sha, "catalog_sha256": catalog, "provisional": sha != catalog,
        "table_counts": counts, "command_lexical_inventory": inventory,
        "distinct_contributor_ids": len(speakers), "new_contributor_ids": sum(new.values()),
    }


def group_alias(value):
    return canonical(value).replace(" ", "-")


def source_identity(name, source):
    return {group_alias(name), group_alias(source.get("speaker_group", name)),
            group_alias(source.get("publisher", ""))}


def reassigned_assets(groups):
    assets = set()
    for fixture in HELDOUT_FIXTURES:
        for name, source in specification(fixture)["sources"].items():
            if source_identity(name, source) & groups:
                assets.update(source[key] for key in ("page", "enclosure", "sha256") if source.get(key))
    return assets


def is_reassigned(case, sources, groups, assets):
    names = [case["source"]]
    if "negation_prefix" in case:
        names.append(case["negation_prefix"]["source"])
    return any(source_identity(name, sources[name]) & groups
               or any(sources[name].get(key) in assets for key in ("page", "enclosure", "sha256"))
               for name in names)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("test", "fresh", "asr-holdout", "replay", "human", "public-orders"))
    for name in ("program", "baseline", "prior", "turn", "words"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--extra", type=Path)
    parser.add_argument("--adapted", type=Path)
    parser.add_argument("--continuous", type=Path)
    parser.add_argument("--rescue", action="store_true")
    parser.add_argument("--contrast", action="store_true")
    add_panel_options(parser, "-public")
    parser.add_argument("--esmx-explicit-partition", action="store_true",
                        help="Exclude programs reassigned to the explicit Mexican Spanish training set")
    for name in ("corpus", "older-wide", "archive"):
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()
    if args.rescue and not args.continuous:
        parser.error("--rescue requiere --continuous")
    if args.contrast and (not args.rescue or not args.adapted or digest(args.continuous) != CONTRAST_BANK_SHA256
                          or digest(args.adapted) != CONTRAST_WORD_SHA256):
        parser.error("--contrast requiere --rescue y los bancos fijados de voz continua y palabras")
    if (args.public_fixture or args.esmx_explicit_partition) and args.operation != "public-orders":
        parser.error("Seleccionar un panel o una partición humana requiere public-orders")
    if args.adapted and not args.extra:
        parser.error("--adapted requiere --extra para conservar la referencia anterior")
    if args.continuous and not args.adapted:
        parser.error("--continuous requiere los bancos anteriores")
    if digest(args.baseline) != BASELINE_SHA256 or digest(args.prior) != BANK_SHA256:
        parser.error("El banco acústico anterior no coincide con la referencia")
    provenance = {key: digest(value) for key, value in (("compact", args.prior), ("turn_cnn", args.turn),
                                                       ("word_gate", args.words))}
    if args.extra:
        provenance["word_pair_extra"] = digest(args.extra)
    if args.adapted:
        provenance["word_adapt_extra"] = digest(args.adapted)
    if args.continuous:
        provenance["word_full_acoustic"] = digest(args.continuous)
    if (args.operation == "public-orders" and provenance.get("word_full_acoustic") == EXPLICIT_BANK_SHA256
            and not args.esmx_explicit_partition):
        parser.error("El banco fijado de órdenes explícitas requiere --esmx-explicit-partition")
    if args.operation == "human":
        rows, metadata = real_trials(args)
        report = metadata
    elif args.operation == "public-orders":
        if args.corpus is None:
            parser.error("--corpus debe apuntar a los cortes públicos verificados")
        fixture = args.public_fixture or PUBLIC_FIXTURE
        excluded = []
        groups = frozenset()
        assets = set()
        if args.esmx_explicit_partition:
            train = whole_orders.specification(whole_orders.EXPLICIT_ESMX_FIXTURE)
            groups = whole_orders.explicit_exclusions(train["sources"])
            assets = reassigned_assets(groups)
        sources = specification(fixture)["sources"]
        rows = []
        for case, wav in verified_cases(args.corpus, fixture):
            if is_reassigned(case, sources, groups, assets):
                excluded.append(case["id"])
                continue
            rows.append(trial(args, wav, case["category"], case["expected"], case["id"]))
        if any(row["overlong"] for row in rows):
            raise ValueError("Una orden pública supera el límite del turno")
        report = {"public_fixture_sha256": digest(fixture), "cases": rows}
        if args.esmx_explicit_partition:
            report["training_partition"] = {
                "fixture_sha256": digest(whole_orders.EXPLICIT_ESMX_FIXTURE),
                "excluded_groups": sorted(groups), "excluded_ids": excluded,
            }
    else:
        parent = FIXTURE.parent.parent
        fixture = {"test": FIXTURE, "fresh": FIXTURE.with_name("fresh_suite.json"),
                   "asr-holdout": parent / "transcript/holdout.json",
                   "replay": parent / "phrases_cases.json"}[args.operation]
        source = manifest(args.corpus, fixture)
        rows = []
        if args.operation == "replay":
            source = [row for row in source if row["split"] in {"test", "enrolled"}
                      and row["category"] in {"d6", "legacy", "known_deny"}]
            older = manifest(args.older_wide, parent / "d6_cases.json")
            original, = [item for item in older if item["id"] == "test-0379"]
            rows.append(trial(args, args.older_wide / "audio/test-0379.wav", "original_fact_veto",
                              0, original["id"]))
        for row in source:
            if args.operation in {"test", "fresh"} and row["split"] != "test":
                continue
            expected = (6 if row["label"] == 1 else 0) if args.operation in {"test", "fresh"} else row["label"]
            wav = args.corpus / "audio" / (row["id"] + ".wav")
            rows.append(trial(args, wav, row["category"], expected, row["id"]))
        report = {"fixture_sha256": digest(fixture), "cases": rows}
    print(json.dumps({"operation": args.operation, "bank_sha256": provenance,
                      "categories": summarize(rows), **report}, indent=2))


if __name__ == "__main__":
    main()
