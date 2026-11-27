"""Prepare continuous public training turns without access to the held-out recordings."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess

from ald_round import canonical, digest
from pilot import read_pcm
from public_orders import DIRECT_ESMX_FIXTURE, HELDOUT_FIXTURES, ORDERED_FIXTURE, valid_interval, verify_pcm
from public_orders import specification as heldout_specification

FIXTURE = Path(__file__).with_name("whole_orders.json")
EXPANDED_FIXTURE = FIXTURE.with_name("whole_orders_expanded.json")
ESMX_FIXTURE = FIXTURE.with_name("whole_orders_esmx.json")
EXPLICIT_ESMX_FIXTURE = FIXTURE.with_name("whole_orders_esmx_explicit.json")
BACKUP_ESMX_FIXTURE = FIXTURE.with_name("whole_orders_esmx_backup_expanded.json")
RETRAIN_ESMX_FIXTURE = FIXTURE.with_name("whole_orders_esmx_retrain.json")
EXPLICIT_FIXTURES = (EXPLICIT_ESMX_FIXTURE, BACKUP_ESMX_FIXTURE)
TRAINING_FIXTURES = (FIXTURE, EXPANDED_FIXTURE, ESMX_FIXTURE, *EXPLICIT_FIXTURES, RETRAIN_ESMX_FIXTURE)
REASSIGNED_RECORDINGS = {"fungi", "juganor"}
# Hosts and guests shared with the archival episode.
EXPLICIT_GUEST_GROUPS = frozenset({"condado-veinte"})
ARCHIVED_LABELS = {"explicit_single_d6_order": "positive", "multiple_d6": "other_die"}
ESMX_PROFILE = "esmx-continuous-training-v1"
LABELS = {"outside": 0, "positive": 1, "other_die": 2, "negated": 3}
TEMPOS = {"train": (100, 92, 108), "dev": (100,)}


def says_literal_d6(text):
    return re.search(r"\bdado de seis\b", canonical(text)) is not None


def explicit_exclusions(sources):
    return EXPLICIT_GUEST_GROUPS | {sources[name]["speaker_group"] for name in REASSIGNED_RECORDINGS}


def specification(fixture=None):
    fixture = fixture or FIXTURE
    spec = json.loads(fixture.read_text("utf-8"))
    expanded_backup = fixture.resolve() == BACKUP_ESMX_FIXTURE.resolve()
    if expanded_backup:
        base = json.loads(EXPLICIT_ESMX_FIXTURE.read_text("utf-8"))
        if (spec.get("base_sha256") != digest(EXPLICIT_ESMX_FIXTURE)
                or spec["sources"].keys() & base["sources"].keys()):
            raise ValueError("Cambió la ficha explícita base o se repitió una fuente")
        spec["sources"] = base["sources"] | spec["sources"]
        spec["cases"] = base["cases"] + spec["cases"]
    esmx = spec.get("profile") == ESMX_PROFILE
    explicit = expanded_backup or fixture.resolve() == EXPLICIT_ESMX_FIXTURE.resolve()
    retrain = fixture.resolve() == RETRAIN_ESMX_FIXTURE.resolve()
    archives = {}
    if explicit:
        direct = heldout_specification(DIRECT_ESMX_FIXTURE)
        archives.update(dict.fromkeys(REASSIGNED_RECORDINGS, direct))
    if expanded_backup:
        archives["zelda"] = heldout_specification(ORDERED_FIXTURE)
    if (explicit or retrain or fixture.resolve() == ESMX_FIXTURE.resolve()) and not esmx:
        raise ValueError("Falta el perfil de entrenamiento es-MX")
    reserved = [(name, meta) for panel in HELDOUT_FIXTURES
                for name, meta in heldout_specification(panel)["sources"].items()]
    reserved_names = {name for name, _ in reserved}
    reserved_groups = {meta.get("speaker_group", name) for name, meta in reserved}
    asset_keys = ("page", "enclosure", "sha256") if expanded_backup else ("enclosure", "sha256")
    reserved_assets = {value[key] for _, value in reserved for key in asset_keys if value.get(key)}
    groups, publishers = {}, {}
    for source, meta in spec["sources"].items():
        split, group = meta["split"], meta["speaker_group"]
        overlaps = (source in reserved_names or group in reserved_groups
                    or any(meta.get(key) in reserved_assets for key in asset_keys))
        reassigned = False
        if source in archives and split == "train":
            recorded = archives[source]["sources"][source]
            reassigned = (all(meta.get(key) == recorded[key] for key in ("speaker_group", "page", "enclosure", "sha256"))
                          and (source == "zelda" or meta.get("locale") == recorded.get("locale"))
                          and meta.get("format", "mp3") == recorded.get("format", "mp3"))
        if (not re.fullmatch(r"[a-z][a-z0-9-]+", source) or split not in TEMPOS
                or not group or (overlaps and not reassigned)
                or meta.get("format", "mp3") not in {"mp3", "m4a", "webm"}
                or not re.fullmatch(r"[a-f0-9]{64}", meta["sha256"])):
            raise ValueError("La fuente no es válida o pertenece al panel reservado")
        if esmx:
            publisher = canonical(meta.get("publisher", ""))
            if not publisher or meta.get("locale") != "es-MX" or not meta.get("locale_evidence"):
                raise ValueError("Falta acreditar el origen es-MX del locutor")
            if publishers.setdefault(publisher, split) != split:
                raise ValueError("Un programa aparece en entrenamiento y calibración")
        if groups.setdefault(group, split) != split:
            raise ValueError("Un grupo de voces aparece en entrenamiento y calibración")
    sources = spec["sources"]
    if len({meta["sha256"] for meta in sources.values()}) != len(sources):
        raise ValueError("Una grabación aparece bajo dos fuentes")
    seen, time_ranges, positives = set(), {}, Counter()
    for case in spec["cases"]:
        name, split, category = case["id"], sources[case["source"]]["split"], case["category"]
        start, end = case["start_ms"], case["end_ms"]
        if (name in seen or not re.fullmatch(fr"{split}-[0-9]{{2}}", name)
                or category not in LABELS or not isinstance(start, int) or not isinstance(end, int)
                or start < 0 or not 300 <= end - start <= 4000):
            raise ValueError("Toma continua inválida")
        if esmx:
            context = case.get("context_ms")
            if (not re.fullmatch(r"[a-f0-9]{64}", case.get("pcm_sha256", ""))
                    or not case.get("summary")
                    or not valid_interval(context) or not context[0] <= start < end <= context[1]):
                raise ValueError("Falta contexto o huella PCM de la toma es-MX")
            if category == "positive":
                evidence = case.get("die_evidence_ms")
                if (case.get("die_basis") not in {"explicit", "contextual"}
                        or not valid_interval(evidence) or not context[0] <= evidence[0] < evidence[1] <= context[1]
                        or (case["die_basis"] == "explicit" and not start <= evidence[0] < evidence[1] <= end)):
                    raise ValueError("No se acredita que la orden pida un solo D6")
        if case["source"] in archives:
            original = next((row for row in archives[case["source"]]["cases"]
                             if row["id"] == case.get("archive_id")), None)
            if original is None or any(case[key] != original.get(key) for key in ("source", "start_ms", "end_ms")):
                raise ValueError("La toma reasignada difiere de su caso archivado")
            expected_label = ARCHIVED_LABELS.get(original["category"])
            if case["source"] == "zelda":
                expected_label = "positive" if original["id"] == "public-32" and original["expected"] == 6 else None
            elif case["pcm_sha256"] != original.get("pcm_sha256"):
                raise ValueError("La toma reasignada difiere de su caso archivado")
            if category != expected_label:
                raise ValueError("Cambió la etiqueta de la toma reasignada")
        if explicit and split == "train" and category == "positive" and (
                case.get("die_basis") != "explicit"
                or not says_literal_d6(case.get("spoken_form", ""))):
            raise ValueError("La orden de entrenamiento debe decir dado de seis")
        ranges = time_ranges.setdefault(case["source"], [])
        if any(start < earlier_end and earlier_start < end for earlier_start, earlier_end in ranges):
            raise ValueError("Los recortes de una grabación se superponen")
        ranges.append((start, end))
        seen.add(name)
        positives[split] += category == "positive"
    if not all(positives[split] for split in TEMPOS):
        raise ValueError("Falta una orden D6 en entrenamiento o calibración")
    return spec


def order_rows(spec):
    for case in spec["cases"]:
        split = spec["sources"][case["source"]]["split"]
        for tempo in TEMPOS[split]:
            if (case["end_ms"] - case["start_ms"]) * 100 > 4000 * tempo:
                continue
            yield dict(case, id=f'{case["id"]}-{tempo}', original_id=case["id"], split=split,
                       label=LABELS[case["category"]], tempo=tempo)


def make_audio(source, output, start, end):
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-n", "-ss", f"{start/1000:.3f}",
         "-i", source, "-t", f"{(end-start)/1000:.3f}", "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", output], check=True, capture_output=True, timeout=30,
    )


def build(root, out, fixture=None):
    fixture = fixture or FIXTURE
    spec = specification(fixture)
    paths = {name: root / f'{name}.{meta.get("format", "mp3")}' for name, meta in spec["sources"].items()}
    for name, path in paths.items():
        if digest(path) != spec["sources"][name]["sha256"]:
            raise ValueError(f"La grabación pública {name} no coincide con la copia fijada")
    out.mkdir(mode=0o700)
    audio = out / "audio"
    audio.mkdir(mode=0o700)
    rows = []
    for row in order_rows(spec):
        wav = audio / (row["id"] + ".wav")
        if row["tempo"] == 100:
            make_audio(paths[row["source"]], wav, row["start_ms"], row["end_ms"])
            verify_pcm(wav, row)
        else:
            original = audio / (row["original_id"] + "-100.wav")
            subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", "-n", "-i", original,
                 "-af", f'atempo={row["tempo"]/100:.2f}', "-c:a", "pcm_s16le", wav],
                check=True, capture_output=True, timeout=30,
            )
        read_pcm(wav)
        rows.append(dict(row, wav_sha256=digest(wav)))
    (out / "manifest.json").write_text(json.dumps({
        "fixture_sha256": digest(fixture), "samples": rows,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def verified_manifest(folder):
    path = folder / "manifest.json"
    manifest = json.loads(path.read_text("utf-8"))
    fixture = next((item for item in TRAINING_FIXTURES
                    if item.is_file() and manifest["fixture_sha256"] == digest(item)), None)
    if fixture is None:
        raise ValueError("El corpus continuo difiere de la anotación fijada")
    spec = specification(fixture)
    for row, expected in zip(manifest["samples"], order_rows(spec), strict=True):
        if {key: value for key, value in row.items() if key != "wav_sha256"} != expected:
            raise ValueError("Cambió una etiqueta, fuente o partición")
        if digest(folder / "audio" / (row["id"] + ".wav")) != row["wav_sha256"]:
            raise ValueError("Cambió una toma continua")
    provenance = {
        "fixture_sha256": digest(fixture), "manifest_sha256": digest(path),
        **({"public_excluded_groups": sorted(explicit_exclusions(spec["sources"]))}
           if fixture in EXPLICIT_FIXTURES else {}),
        "speaker_groups": {split: sorted({source["speaker_group"] for source in spec["sources"].values()
                                         if source["split"] == split}) for split in TEMPOS},
        **({"positive_by_die_basis": {
            split: dict(Counter(case["die_basis"] for case in spec["cases"] if case["category"] == "positive"
                                and spec["sources"][case["source"]]["split"] == split))
            for split in TEMPOS}} if spec.get("profile") == ESMX_PROFILE else {}),
        "original_counts": {split: dict(Counter(case["category"] for case in spec["cases"]
                                               if spec["sources"][case["source"]]["split"] == split))
                            for split in TEMPOS},
    }
    return manifest["samples"], provenance, spec["sources"]


def read_manifest(folder):
    rows, provenance, _ = verified_manifest(folder)
    return rows, provenance


def pooled_manifests(folders):
    """Combine verified speech corpora, counting any shared recording interval only once."""
    partitions, originals, variants, corpus, grouped = {}, {}, {}, [], []
    for folder in folders:
        rows, provenance, sources = verified_manifest(folder)
        for source in sources.values():
            for field in ("speaker_group", "sha256", "enclosure"):
                key = (field, source[field])
                if partitions.setdefault(key, source["split"]) != source["split"]:
                    raise ValueError("Una fuente o voz cruza entrenamiento y calibración en el corpus combinado")
        selected = []
        for row in rows:
            source = sources[row["source"]]["sha256"]
            start, end = row["start_ms"], row["end_ms"]
            label = row["label"]
            recordings = originals.setdefault(source, {})
            interval = (start, end)
            if interval not in recordings:
                if any(start < other_end and other_start < end for other_start, other_end in recordings):
                    raise ValueError("Se superponen recortes de una misma grabación en el corpus combinado")
                recordings[interval] = row
            elif recordings[interval]["label"] != label:
                raise ValueError("Hay etiquetas distintas para el mismo recorte en el corpus combinado")
            key = (source, interval, row["tempo"])
            if key not in variants:
                variants[key] = row["wav_sha256"]
                selected.append(row)
            elif variants[key] != row["wav_sha256"]:
                raise ValueError("Cambió el audio de una variante repetida en el corpus combinado")
        grouped.append((folder, selected))
        corpus.append(provenance)
    takes = [row for recording in originals.values() for row in recording.values()]
    excluded_groups = sorted({group for item in corpus for group in item.get("public_excluded_groups", ())})
    return grouped, {
        "corpora": corpus,
        "original_counts": {split: dict(Counter(row["category"] for row in takes if row["split"] == split))
                            for split in TEMPOS},
        "positive_by_die_basis": {
            split: dict(Counter(row.get("die_basis", "unannotated") for row in takes
                                if row["split"] == split and row["label"] == 1))
            for split in TEMPOS
        },
        "unique_tempo_variants": len(variants),
        **({"public_excluded_groups": excluded_groups} if excluded_groups else {}),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    panel = parser.add_mutually_exclusive_group()
    panel.add_argument("--expanded", action="store_const", const=EXPANDED_FIXTURE, dest="fixture")
    panel.add_argument("--esmx", action="store_const", const=ESMX_FIXTURE, dest="fixture")
    panel.add_argument("--esmx-explicit", action="store_const", const=EXPLICIT_ESMX_FIXTURE, dest="fixture")
    panel.add_argument("--esmx-backup-expanded", action="store_const", const=BACKUP_ESMX_FIXTURE, dest="fixture")
    panel.add_argument("--esmx-retrain", action="store_const", const=RETRAIN_ESMX_FIXTURE, dest="fixture")
    args = parser.parse_args()
    build(args.sources, args.out, args.fixture)
