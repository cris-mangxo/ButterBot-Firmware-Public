"""Count transcripts matching command patterns in a Common Voice tarball."""

import argparse
from collections import Counter
import csv
import io
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import unicodedata


MENTIONS = {
    "butter": r"\bmantequilla\b", "joke": r"\bchistes?\b", "time": r"\bhoras?\b", "d6": r"\bdados?\b",
}
COMMANDS = {
    "butter": r"\b(?:pasa|pasame|pasale|pasenme|pasenos|pasar|dame|dales|dale|acercame|alcanzame|trae|traeme) "
              r"(?:por favor )?(?:un poco de |la |una )?mantequilla\b",
    "joke": r"\b(?:cuenta|cuentame|cuentanos|dime|di|dinos|echa|echame|echate|avienta|avientate|aventate|avientame|"
            r"cuente|cuenteme) (?:por favor )?(?:un|el|otro|algun) chiste\b",
    "time": r"\b(?:que horas? (?:es|son)(?: ahora)?|(?:dime|dame|diga) la hora(?: actual)?|hora actual)\b",
    "d6": r"\b(?:dado|dados) de (?:seis|6)\b",
}
ROLES = {"train.tsv", "dev.tsv", "test.tsv"}


def normalize(text):
    letters = "".join(char for char in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(char))
    return " ".join(re.findall(r"\w+", letters))


def screen(archive_path):
    counts = {}
    with tarfile.open(archive_path, mode="r|gz") as archive:
        for member in archive:
            name = PurePosixPath(member.name).name
            if not member.isfile() or name not in ROLES:
                continue
            if name in counts or member.size > 50_000_000:
                raise ValueError("Unexpected metadata size or duplicate role")
            with archive.extractfile(member) as stream:
                table = io.StringIO(stream.read().decode("utf-8-sig"), newline="")
            rows = csv.DictReader(table, delimiter="\t")
            if "sentence" not in (rows.fieldnames or ()):
                raise ValueError("Expected Common Voice sentence column")
            totals = Counter()
            for row in rows:
                text = normalize(row["sentence"])
                totals["rows"] += 1
                for suffix, patterns in (("mention", MENTIONS), ("phrase", COMMANDS)):
                    for intent, pattern in patterns.items():
                        if re.search(pattern, text):
                            totals[f"{intent}_{suffix}"] += 1
            counts[name] = totals
    if set(counts) != ROLES:
        raise ValueError("Missing Common Voice train/dev/test metadata")
    return {"archive_bytes": archive_path.stat().st_size, "counts": counts,
            "phrase_matches": sum(count for totals in counts.values() for key, count in totals.items()
                                  if key.endswith("_phrase"))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    print(json.dumps(screen(parser.parse_args().archive), indent=2))
