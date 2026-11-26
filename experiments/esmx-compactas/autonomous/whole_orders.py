"""Prepare continuous public training turns without access to the held-out recordings."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess

from ald_round import digest
from pilot import read_pcm
from public_orders import EXPANDED_FIXTURE as EXPANDED_HELDOUT
from public_orders import specification as heldout_specification

FIXTURE = Path(__file__).with_name("whole_orders.json")
EXPANDED_FIXTURE = FIXTURE.with_name("whole_orders_expanded.json")
LABELS = {"outside": 0, "positive": 1, "other_die": 2, "negated": 3}
TEMPOS = {"train": (100, 92, 108), "dev": (100,)}


def specification(fixture=None):
    fixture = fixture or FIXTURE
    spec = json.loads(fixture.read_text("utf-8"))
    reserved = heldout_specification()["sources"] | heldout_specification(EXPANDED_HELDOUT)["sources"]
    reserved_groups = {meta.get("speaker_group", name) for name, meta in reserved.items()}
    reserved_assets = {value[key] for value in reserved.values() for key in ("enclosure", "sha256")}
    groups = {}
    for source, meta in spec["sources"].items():
        split, group = meta["split"], meta["speaker_group"]
        if (not re.fullmatch(r"[a-z][a-z0-9-]+", source) or split not in TEMPOS
                or not group or source in reserved or group in reserved_groups
                or meta.get("format", "mp3") not in {"mp3", "m4a"}
                or not re.fullmatch(r"[a-f0-9]{64}", meta["sha256"])
                or any(meta[key] in reserved_assets for key in ("enclosure", "sha256"))):
            raise ValueError("La fuente no es válida o pertenece al panel reservado")
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


def read_manifest(folder):
    path = folder / "manifest.json"
    manifest = json.loads(path.read_text("utf-8"))
    fixture = next((item for item in (FIXTURE, EXPANDED_FIXTURE)
                    if item.is_file() and manifest["fixture_sha256"] == digest(item)), None)
    if fixture is None:
        raise ValueError("El corpus continuo difiere de la anotación fijada")
    spec = specification(fixture)
    for row, expected in zip(manifest["samples"], order_rows(spec), strict=True):
        if {key: value for key, value in row.items() if key != "wav_sha256"} != expected:
            raise ValueError("Cambió una etiqueta, fuente o partición")
        if digest(folder / "audio" / (row["id"] + ".wav")) != row["wav_sha256"]:
            raise ValueError("Cambió una toma continua")
    return manifest["samples"], {
        "fixture_sha256": digest(fixture), "manifest_sha256": digest(path),
        "speaker_groups": {split: sorted({source["speaker_group"] for source in spec["sources"].values()
                                         if source["split"] == split}) for split in TEMPOS},
        "original_counts": {split: dict(Counter(case["category"] for case in spec["cases"]
                                               if spec["sources"][case["source"]]["split"] == split))
                            for split in TEMPOS},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expanded", action="store_true")
    args = parser.parse_args()
    build(args.sources, args.out, EXPANDED_FIXTURE if args.expanded else FIXTURE)
