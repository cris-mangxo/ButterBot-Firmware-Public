"""Verify the native clock fallback and original factory routine on the sealed panel."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import subprocess

from capture_replay import pcm_from, recordings
from time_lab import digest, file_digest, invoke


CLOCK = 5
EXPECTED_NEW = {"T01-388", "T02-388", "T06-388"}


def replay(args):
    index = json.loads(args.panel.read_text())
    if (index["host_sha256"] != file_digest(args.reference_host)
            or index["capture_sha256"] != file_digest(args.reference_capture)):
        raise ValueError("Baseline binary changed")
    args.output.mkdir(mode=0o700)
    rows = []
    for row in index["results"]:
        case = row["id"]
        if re.fullmatch(r"[CT][0-9]{2}-[0-9]{3}", case) is None:
            raise ValueError("Unexpected panel ID")
        episode = {"id": case, "views": {}}
        for view in ("raw", "retained"):
            info = row[view]
            pcm = (args.pcm / f"{case}.{view}.pcm").read_bytes()
            if digest(pcm) != info["pcm_sha256"]:
                raise ValueError(f"Sealed PCM changed: {case}/{view}")
            selected = int(invoke(args.host, pcm))
            detailed = json.loads(invoke(args.host, pcm, "--decision"))
            expected = CLOCK if case in EXPECTED_NEW else info["intent"]
            if selected != expected or detailed["disposition"] != info["decision"]:
                raise ValueError(f"Native recognition or previous disposition differs: {case}/{view}")
            witness = {"intent": selected, "upstream_disposition": detailed["disposition"], "pcm_sha256": digest(pcm)}
            if view == "raw":
                capture = json.loads(invoke(args.capture, pcm, "--nominal-hangover"))
                captured_pcm = invoke(args.capture, pcm, "--pcm-with-hangover")
                if capture["observed"] != selected or digest(captured_pcm) != row["retained"]["pcm_sha256"]:
                    raise ValueError(f"Capture decision or PCM differs: {case}")
                witness["capture"] = capture
            if case in EXPECTED_NEW:
                clip = args.output / f"{case}.{view}.aac"
                routine = subprocess.run([args.routine, "time"], input=pcm, check=True, capture_output=True,
                                         timeout=10, env=dict(os.environ, BB_ROUTINE_HOST_AAC_OUT=str(clip)))
                checked = json.loads(routine.stdout)
                if (checked["recognized_label"] != CLOCK or checked["real_ticks"] != 1
                        or not checked["stock_payload_verified"] or not checked["matching_aac_clips"]):
                    raise ValueError(f"Unexpected clock routine witness: {case}/{view}")
                subprocess.run([args.ffmpeg, "-v", "error", "-i", clip, "-f", "null", "-"],
                               check=True, capture_output=True, timeout=10)
                witness["factory_routine"] = dict(checked, aac_sha256=file_digest(clip))
            episode["views"][view] = witness
        rows.append(episode)
    if len(rows) != 18 or not EXPECTED_NEW <= {row["id"] for row in rows}:
        raise ValueError("Incomplete Heroico panel")
    return rows


def historical(args):
    def evaluate(row):
        case, _, _, expected, path, pin = row
        pcm = pcm_from(path, pin)
        retained = invoke(args.reference_capture, pcm, "--pcm-with-hangover")
        decisions = {}
        for view, sample in (("raw", pcm), ("retained", retained)):
            if not 0 < len(sample) <= 128_000:
                decisions[view] = None
                continue
            previous = int(invoke(args.reference_host, sample))
            now = int(invoke(args.host, sample))
            if previous != now:
                raise ValueError(f"Historical regression: {case}/{view}, expected baseline {previous}, received {now}")
            decisions[view] = now
        return {"id": case, "expected": expected, "intents": decisions}
    with ThreadPoolExecutor(max_workers=6) as tasks:
        return list(tasks.map(evaluate, recordings(args.historical)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("panel", "pcm", "host", "capture", "routine", "ffmpeg", "reference-host",
                 "reference-capture", "historical", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"panel_sha256": file_digest(args.panel),
                      "binary_sha256": {name: file_digest(getattr(args, name))
                                        for name in ("host", "capture", "routine", "reference_host", "reference_capture")},
                      "heroico": replay(args), "historical": historical(args)}, ensure_ascii=False, indent=2))
