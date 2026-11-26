"""Check MSWC train/dev identities against every table in the local Common Voice archive."""

import argparse
import csv
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path, PurePosixPath
import tarfile

from ald_round import digest
from common_voice import DATASETS
from d6_guard import LOCAL_CV_SHA256
from human_words import read_manifest
from phrases_human import MAX_TABLE_BYTES, TABLES


def common_voice_ids(archive):
    tables, all_ids, regression_ids = {}, set(), set()
    with tarfile.open(archive, "r|*") as tar:
        for member in tar:
            name = PurePosixPath(member.name).name
            if not member.isfile() or not name.endswith(".tsv"):
                continue
            if name in tables or not 0 < member.size <= MAX_TABLE_BYTES:
                raise ValueError("Tabla de Common Voice inválida para la auditoría")
            with tar.extractfile(member) as stream:
                table = csv.DictReader(StringIO(stream.read().decode("utf-8-sig")), delimiter="\t")
                has_ids = "client_id" in (table.fieldnames or ())
                ids = {sha256(row["client_id"].encode()).hexdigest() for row in table if row["client_id"]} if has_ids else set()
            tables[name] = {"speaker_ids": len(ids), "has_id_column": has_ids}
            all_ids.update(ids)
            if name in TABLES:
                regression_ids.update(ids)
    if not set(TABLES) <= tables.keys() or not regression_ids:
        raise ValueError("Faltan los identificadores de las tablas de regresión")
    return tables, all_ids, regression_ids


def audit(archive, corpora):
    checksum = digest(archive)
    if checksum not in {DATASETS["female"][2], LOCAL_CV_SHA256}:
        raise ValueError("El archivo Common Voice no coincide con la referencia")
    tables, all_ids, regression_ids = common_voice_ids(archive)
    report = {
        "archive_sha256": checksum, "tables": tables, "all_available_ids": len(all_ids),
        "regression_table_ids": len(regression_ids), "training_overlap": {},
    }
    for corpus in corpora:
        rows, evidence = read_manifest(corpus)
        if evidence["reference_sha256"] != checksum or corpus.name in report["training_overlap"]:
            raise ValueError("El manifiesto no coincide con la referencia de la auditoría")
        speakers = {split: {speaker for row in rows if row["split"] == split for speaker, _ in row["sources"]}
                    for split in ("train", "dev")}
        report["training_overlap"][corpus.name] = {
            "manifest_sha256": evidence["manifest_sha256"],
            "train_vs_dev": len(speakers["train"] & speakers["dev"]),
            **{split: {"vs_all_tables": len(ids & all_ids), "vs_regression_tables": len(ids & regression_ids)}
               for split, ids in speakers.items()},
        }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.reference, args.corpus), indent=2))
