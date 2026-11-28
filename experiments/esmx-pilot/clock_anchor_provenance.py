"""Bind the source roles of the local clock-transfer experiment without audio."""

import argparse
import json
from pathlib import Path
import re

from time_lab import digest, file_digest


def seal(value):
    return digest(value.encode("utf-8"))


def heroico(entry):
    path = entry["origin"]
    origin = re.fullmatch(r"LDC2006S37/data/speech/heroico/Recordings_Spanish/([0-9]+)/([0-9]+)\.wav", path)
    if origin is None or entry["prompt"] != origin[2]:
        raise ValueError("Unexpected Heroico reading archive entry")
    return {"speaker": seal("Heroico reading speaker " + origin[1]), "session": seal(path.rsplit("/", 1)[0]),
            "recording": seal(path), "source_wav_sha256": entry["sha256"]}


def require_disjoint(groups):
    fitted, heldout = groups["fitting"], groups["heldout"]
    if not fitted or not heldout:
        raise ValueError("Both speaker roles must be present")
    for axis in ("speaker", "session", "recording", "source_wav_sha256"):
        candidate = {row[axis] for row in fitted}
        reserved = {row[axis] for row in heldout}
        if any(not re.fullmatch(r"[0-9a-f]{64}", v) for v in candidate | reserved) or candidate & reserved:
            raise ValueError(f"Invalid or overlapping {axis} across fitting and held-out roles")
    if len(heldout) != len({row["recording"] for row in heldout}):
        raise ValueError("Duplicate held-out recording")


def curate(args):
    def read(path):
        return json.loads(path.read_text())

    donor = read(args.donor)
    agenda = read(args.agenda)
    donor_seal = read(args.heroico_calibration)
    if (donor["role"] != "local_calibration_only" or agenda["role"] != "local_calibration_only"
            or file_digest(args.heroico_calibration) != donor["role_assignment_manifest_sha256"]):
        raise ValueError("Missing source roles for fitting")
    if (donor_seal["speaker_disjoint_from_both_test_sets"] is not True
            or any(entry["role"] != "calibration_only" for entry in donor_seal["entries"])):
        raise ValueError("The Heroico donor seal must contain only calibration sources")
    selected, = [row for row in donor_seal["entries"] if row["sha256"] == donor["source_sha256"]]
    clocks, controls = read(args.heroico_clock), read(args.heroico_agenda)
    if not all(panel["heldout_only"] is True for panel in (clocks, controls)):
        raise ValueError("Heroico evaluation source is not sealed")
    entries = [*clocks["entries"], *controls["entries"]]
    if any(entry["role"] != "heldout" for entry in entries):
        raise ValueError("Heroico evaluation includes a non-reserved role")
    fitting = [heroico(selected)]
    agenda_sources = {row["sha256"]: row for row in read(args.agenda_source)["rows"]}
    for entry in agenda["rows"]:
        source = agenda_sources[entry["sha256"]]
        if "FBK-MT%2FSpeech-MASSIVE" not in source["source_url"]:
            raise ValueError("Unexpected calibration agenda corpus")
        fitting.append({"speaker": seal("Speech-MASSIVE speaker " + source["speaker_group"]),
                        "session": seal("Speech-MASSIVE collection " + source["source_sha256"]),
                        "recording": seal("Speech-MASSIVE source row " + source["id"]),
                        "source_wav_sha256": source["source_sha256"]})
    roles = {"fitting": fitting, "heldout": [heroico(row) for row in entries]}
    require_disjoint(roles)
    panel = read(args.panel)
    expected = {row["speaker"] + "-" + row["prompt"]: row["sha256"] for row in entries}
    observed = {row["id"]: row["source_sha256"] for row in panel["results"]}
    if expected != observed or len(expected) != len(panel["results"]):
        raise ValueError("Replay is not built from these held-out source WAVs")
    return {"schema": "bb-clock-source-roles-1", "roles": roles, "pins": {
        key: file_digest(getattr(args, key)) for key in ("donor", "agenda", "heroico_calibration",
            "heroico_clock", "heroico_agenda", "agenda_source", "panel", "quality")},
        "scope": "Source corpus keys and recording origins. Cross-corpus real-world speaker identity is unknown."}


def verify(path, **inputs):
    roles = json.loads(path.read_text())
    if roles["schema"] != "bb-clock-source-roles-1":
        raise ValueError("Unexpected audio role seal format")
    require_disjoint(roles["roles"])
    for name, artifact in inputs.items():
        if file_digest(artifact) != roles["pins"][name]:
            raise ValueError(f"Source role seal differs for {name}")
    return file_digest(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("donor", "agenda", "heroico-calibration", "heroico-clock", "heroico-agenda",
                 "agenda-source", "panel", "quality"):
        parser.add_argument("--" + name, type=Path, required=True)
    print(json.dumps(curate(parser.parse_args()), indent=2))
