"""Cut fixed continuous human utterances from local copies of public RSS audio."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import wave

from ald_round import digest
from pilot import read_pcm

FIXTURE = Path(__file__).with_name("public_orders.json")
EXPANDED_FIXTURE = FIXTURE.with_name("public_orders_expanded.json")
ORDERED_FIXTURE = FIXTURE.with_name("public_orders_ordered.json")
NATURAL_FIXTURE = FIXTURE.with_name("public_orders_natural.json")
ESMX_FIXTURE = FIXTURE.with_name("public_orders_esmx.json")
DIRECT_ESMX_FIXTURE = FIXTURE.with_name("public_orders_esmx_direct.json")
CONTINUOUS_ESMX_FIXTURE = FIXTURE.with_name("public_orders_esmx_continuous.json")
CANCELLATION_FIXTURE = FIXTURE.with_name("public_orders_esmx_cancellation.json")
TABLE_ESMX_FIXTURE = FIXTURE.with_name("public_orders_esmx_table.json")
MENTION_ESMX_FIXTURE = FIXTURE.with_name("public_orders_esmx_mention.json")
GAMES_ESMX_FIXTURE = FIXTURE.with_name("public_orders_esmx_board_games.json")
FULL_TURNS_ESMX_FIXTURE = FIXTURE.with_name("public_orders_esmx_full_turns.json")
HELDOUT_FIXTURES = (FIXTURE, EXPANDED_FIXTURE, ORDERED_FIXTURE, NATURAL_FIXTURE,
                    ESMX_FIXTURE, DIRECT_ESMX_FIXTURE, CONTINUOUS_ESMX_FIXTURE, CANCELLATION_FIXTURE,
                    TABLE_ESMX_FIXTURE, MENTION_ESMX_FIXTURE, GAMES_ESMX_FIXTURE, FULL_TURNS_ESMX_FIXTURE)
STRICT_ESMX_PROFILE = "esmx-orders-with-controls-v1"
ESMX_POSITIVES = {"explicit_single_d6_order", "contextual_single_d6_order"}
ESMX_CONTEXTUAL = {"contextual_single_d6_order", "contextual_single_d6_cancel",
                   "contextual_single_d6_turn_skip", "contextual_single_d6_reroll_exception",
                   "spontaneous_contextual_die_switch", "contextual_d6_omission_without_substitution"}
NEGATION_GAP_MS = 70


def valid_interval(value):
    return (isinstance(value, list) and len(value) == 2 and all(type(point) is int for point in value)
            and 0 <= value[0] < value[1])


def specification(fixture=None):
    fixture = fixture or FIXTURE
    data = json.loads(fixture.read_text("utf-8"))
    esmx = data.get("profile") == STRICT_ESMX_PROFILE
    fixture_path = fixture.resolve()
    if fixture_path in {ESMX_FIXTURE.resolve(), DIRECT_ESMX_FIXTURE.resolve(),
                        CONTINUOUS_ESMX_FIXTURE.resolve(), CANCELLATION_FIXTURE.resolve(),
                        TABLE_ESMX_FIXTURE.resolve(), MENTION_ESMX_FIXTURE.resolve(),
                        GAMES_ESMX_FIXTURE.resolve(), FULL_TURNS_ESMX_FIXTURE.resolve()} and not esmx:
        raise ValueError("Falta el perfil del panel es-MX")
    pcm_required = esmx or fixture_path == NATURAL_FIXTURE.resolve()
    declared = set()
    for name, source in data["sources"].items():
        if (not re.fullmatch(r"[a-z]+(?:-[a-z]+)*", name)
                or not re.fullmatch(r"[a-f0-9]{64}", source["sha256"])
                or source.get("format", "mp3") not in {"mp3", "wav", "m4a", "mp4", "webm"}):
            raise ValueError("Fuente pública inválida")
        if esmx and (not source.get("speaker_group") or source.get("locale") not in {"es-MX", "es-ES", "es"}
                     or not source.get("locale_evidence")):
            raise ValueError("Falta procedencia del habla pública")
    for case in data["cases"]:
        name = case["id"]
        start, end = case["start_ms"], case["end_ms"]
        if (not re.fullmatch(r"public-[0-9]{2}", name) or name in declared
                or case["source"] not in data["sources"] or case["expected"] not in (0, 6)
                or start < 0 or not 0 < end - start <= 4000):
            raise ValueError("Caso humano inválido")
        pcm_hash = case.get("pcm_sha256")
        if pcm_required and pcm_hash is None:
            raise ValueError("Falta la huella PCM del panel natural")
        if pcm_hash is not None and not re.fullmatch(r"[a-f0-9]{64}", pcm_hash):
            raise ValueError("Huella PCM inválida")
        if esmx:
            locale = data["sources"][case["source"]]["locale"]
            secondary_control = case.get("supplemental_control", False)
            if (not case.get("summary") or "negation_prefix" in case
                    or (case["expected"] == 6) != (case["category"] in ESMX_POSITIVES)
                    or type(secondary_control) is not bool
                    or secondary_control != (locale != "es-MX")
                    or (secondary_control and case["expected"] != 0)):
                raise ValueError("El panel es-MX requiere tomas continuas y controles externos marcados")
        if esmx and case["category"] in ESMX_CONTEXTUAL:
            context = case.get("context_ms")
            evidence = case.get("die_evidence_ms")
            if (not valid_interval(context) or not context[0] <= start < end <= context[1]
                    or not valid_interval(evidence)):
                raise ValueError("Falta el contexto de la toma o del tipo de dado")
            if (case["category"] == "contextual_d6_omission_without_substitution"
                    and not context[0] <= evidence[0] < evidence[1] <= context[1]):
                raise ValueError("La decisión de omitir queda fuera del contexto")
        if "negation_prefix" in case:
            prefix = case["negation_prefix"]
            first, last = prefix["start_ms"], prefix["end_ms"]
            if (case["category"] != "composed_single_d6_negation" or case["expected"] != 0
                    or prefix["source"] not in data["sources"] or prefix["source"] == case["source"]
                    or not 0 <= first < last or last - first > 500
                    or end - start + last - first + NEGATION_GAP_MS > 4000):
                raise ValueError("Composición de negación inválida")
        elif case["category"] == "composed_single_d6_negation":
            raise ValueError("Falta el prefijo humano de negación")
        declared.add(name)
    return data


def verify_unique_ids(fixtures=HELDOUT_FIXTURES):
    seen = set()
    for fixture in fixtures:
        ids = {case["id"] for case in specification(fixture)["cases"]}
        if seen & ids:
            raise ValueError("ID repetido entre paneles humanos")
        seen.update(ids)


def verify_pcm(wav, case):
    actual = hashlib.sha256(read_pcm(wav)).hexdigest()
    if "pcm_sha256" in case and actual != case["pcm_sha256"]:
        raise ValueError("El PCM del recorte no coincide con la referencia")


def cut(path, start, end, output=None):
    sink = output if output is not None else "pipe:1"
    format_flag = [] if output is not None else ["-f", "s16le"]
    return subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
         "-ss", f"{start / 1000:.3f}", "-i", path, "-t", f"{(end - start) / 1000:.3f}",
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", *format_flag, sink],
        check=True, capture_output=True, timeout=30,
    ).stdout

def extract(sources, out, fixture_path=None):
    fixture_path = fixture_path or FIXTURE
    verify_unique_ids()
    fixture = specification(fixture_path)
    paths = {key: sources / f'{key}.{source.get("format", "mp3")}'
             for key, source in fixture["sources"].items()}
    for key, path in paths.items():
        if digest(path) != fixture["sources"][key]["sha256"]:
            raise ValueError(f"La grabación publicada de {key} difiere de la copia fijada")
    out.mkdir(mode=0o700)
    rows = []
    for case in fixture["cases"]:
        wav = out / (case["id"] + ".wav")
        source = paths[case["source"]]
        start, end = case["start_ms"], case["end_ms"]
        if "negation_prefix" not in case:
            cut(source, start, end, wav)
        else:
            prefix = case["negation_prefix"]
            negation = cut(paths[prefix["source"]], prefix["start_ms"], prefix["end_ms"])
            positive = cut(source, start, end)
            with wave.open(str(wav), "wb") as audio:
                audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                audio.writeframes(negation + bytes(32 * NEGATION_GAP_MS) + positive)
        verify_pcm(wav, case)
        rows.append(dict(case, wav_sha256=digest(wav)))
    (out / "manifest.json").write_text(json.dumps({
        "public_fixture_sha256": digest(fixture_path), "cases": rows,
    }, indent=2) + "\n", encoding="utf-8")


def verified_cases(folder, fixture_path=None):
    fixture_path = fixture_path or FIXTURE
    verify_unique_ids()
    evidence = json.loads((folder / "manifest.json").read_text("utf-8"))
    if evidence["public_fixture_sha256"] != digest(fixture_path):
        raise ValueError("El corpus humano difiere de la anotación fijada")
    for case, reference in zip(evidence["cases"], specification(fixture_path)["cases"], strict=True):
        if {key: value for key, value in case.items() if key != "wav_sha256"} != reference:
            raise ValueError("Cambió la anotación previa al ensayo")
        wav = folder / (reference["id"] + ".wav")
        if digest(wav) != case["wav_sha256"]:
            raise ValueError("Cambió la toma pública")
        verify_pcm(wav, case)
        yield case, wav


def add_panel_options(parser, suffix=""):
    panel = parser.add_mutually_exclusive_group()
    for name, fixture in (("expanded", EXPANDED_FIXTURE), ("ordered", ORDERED_FIXTURE),
                          ("natural", NATURAL_FIXTURE), ("esmx", ESMX_FIXTURE),
                          ("esmx-direct", DIRECT_ESMX_FIXTURE), ("esmx-continuous", CONTINUOUS_ESMX_FIXTURE),
                          ("esmx-cancellation", CANCELLATION_FIXTURE), ("esmx-table", TABLE_ESMX_FIXTURE),
                          ("esmx-mention", MENTION_ESMX_FIXTURE), ("esmx-board-games", GAMES_ESMX_FIXTURE),
                          ("esmx-full-turns", FULL_TURNS_ESMX_FIXTURE)):
        panel.add_argument(f"--{name}{suffix}", action="store_const", const=fixture, dest="public_fixture")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    add_panel_options(parser)
    args = parser.parse_args()
    extract(args.sources, args.out, args.public_fixture)


if __name__ == "__main__":
    main()
