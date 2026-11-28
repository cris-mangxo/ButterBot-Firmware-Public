#!/usr/bin/env python3
"""Replay a sealed, local es-MX panel through the host pilot and factory routines."""

import argparse
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import wave

from capture_replay import DISPATCH, pcm_from


LABEL = {0: "none", 1: "butter", 3: "joke", 5: "time", 6: "dice6"}
ARTIFACTS = ("recognizer", "capture", "routine", "ffmpeg")
MAX_SAMPLES = 64_000
MIN_SAMPLES = 4_160  # 24 log-mel windows, including 23 hops of 160 samples.


class PanelError(ValueError):
    """Safe, fixed diagnostic to show to the operator."""


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def require(condition, reason):
    if not condition:
        raise PanelError(reason)


def unique_keys(pairs):
    result = dict(pairs)
    require(len(result) == len(pairs), "duplicate JSON key")
    return result


def text(value):
    return isinstance(value, str) and bool(value) and value == value.strip()


def groups(row):
    source, speaker = (row.get(key) for key in ("source_group", "speaker_group"))
    require(text(source) and text(speaker), "missing group metadata")
    return source, speaker


def read_panel(args):
    manifest = args.manifest.resolve(strict=True)
    content = manifest.read_bytes()
    require(sha256(content).hexdigest() == args.manifest_sha256, "manifest digest mismatch")
    spec = json.loads(content, object_pairs_hook=unique_keys)
    require(isinstance(spec, dict), "expected a manifest object")
    require(type(spec["version"]) is int and spec["version"] == 1, "unsupported manifest version")
    require(re.fullmatch(r"[0-9a-f]{40}", spec["source_sha"]), "invalid source commit")
    require(set(spec["artifact_sha256"]) == set(ARTIFACTS), "incomplete executable inventory")
    programs = {name: getattr(args, name).resolve(strict=True) for name in ARTIFACTS}
    for name, path in programs.items():
        require(path.is_file() and os.access(path, os.X_OK) and digest(path) == spec["artifact_sha256"][name],
                "executable missing or digest mismatch: " + name)

    env = {"PATH": os.defpath, "LC_ALL": "C"}
    clock_files = spec["clock_files_sha256"]
    if args.clock_dir is None:
        require(clock_files == {}, "clock directory required")
    else:
        clock_dir = args.clock_dir.resolve(strict=True)
        entries = {p.name: p for p in clock_dir.iterdir()}
        require(set(entries) == set(clock_files) and entries, "clock inventory mismatch")
        for name, path in entries.items():
            require(not path.is_symlink() and path.is_file() and digest(path) == clock_files[name],
                    "clock input is not a pinned regular file")
        env["BB_ESMX_CLOCK_ACOUSTIC_DIR"] = str(clock_dir)

    rows = spec["episodes"]
    require(isinstance(rows, list) and rows and all(isinstance(row, dict) for row in rows),
            "invalid episode panel")
    adjustment = spec["adjustment"]
    require(isinstance(adjustment, list) and all(isinstance(row, dict) for row in adjustment),
            "invalid adjustment inventory")
    separate = adjustment + [row for row in rows if row.get("role") == "regression"]
    adjust_sources = set()
    adjust_speakers = set()
    for row in separate:
        source, speaker = groups(row)
        adjust_sources.add(source)
        adjust_speakers.add(speaker)
    source_ids = set()
    wave_hashes = set()
    panel = []
    for row in rows:
        role, expected = row["role"], row["expected"]
        require(role in ("heldout", "regression") and row["locale"] == "es-MX" and
                type(expected) is int and expected in LABEL, "invalid episode role, locale or expectation")
        require(text(row.get("rights")) and text(row.get("source_id")), "missing source or rights metadata")
        source, speaker = groups(row)
        if role == "heldout":
            require(source not in adjust_sources and speaker not in adjust_speakers,
                    "reserved group overlaps adjustment or regression")
        file = Path(row["wav"])
        require(file.name == row["wav"] and file.suffix == ".wav", "expected a local WAV filename")
        path = manifest.parent / file
        require(path.is_file() and not path.is_symlink(), "expected a regular WAV inside the panel directory")
        wave_hash = row["wav_sha256"]
        require(re.fullmatch(r"[0-9a-f]{64}", wave_hash)
                and row["source_id"] not in source_ids and wave_hash not in wave_hashes,
                "invalid pin or duplicate recording")
        source_ids.add(row["source_id"])
        wave_hashes.add(wave_hash)
        with wave.open(str(path)) as wav:
            sample_count = wav.getnframes()
        require(MIN_SAMPLES <= sample_count <= MAX_SAMPLES, "WAV sample count outside the pilot budget")
        try:
            pcm = pcm_from(path, wave_hash)
        except ValueError:
            raise PanelError("WAV format or digest mismatch") from None
        require(len(pcm) == 2 * sample_count, "truncated WAV")
        panel.append((role, expected, pcm))
    return spec, programs, env, panel


def invoke(program, arguments, data, env):
    return subprocess.run([program, *arguments], input=data, capture_output=True, env=env, timeout=45)


def episode_view(programs, env, expected, pcm, aac):
    recognition = invoke(programs["recognizer"], (), pcm, env)
    require(recognition.returncode == 0 and re.fullmatch(rb"[0-6]\n", recognition.stdout),
            "unexpected recognizer response")
    observed = int(recognition.stdout)
    reduced = {"observed": observed, "samples": len(pcm) // 2}
    if observed == 0:
        detailed = invoke(programs["recognizer"], ("--decision",), pcm, env)
        require(detailed.returncode == 0, "cannot validate the reason for abstention")
        decision = json.loads(detailed.stdout)
        require(isinstance(decision, dict) and type(decision.get("intent")) is int and decision["intent"] == 0
                and text(decision.get("disposition")), "inconsistent abstention disposition")
        if decision["disposition"] == "invalid_audio":
            return {**reduced, "status": "invalid_audio"}
    output = invoke(programs["routine"], (LABEL[expected],), pcm,
                    {**env, "BB_ROUTINE_HOST_AAC_OUT": str(aac)})
    require(output.returncode in (0, 1), "unexpected routine result")
    report = json.loads(output.stdout)
    require(isinstance(report, dict) and type(report.get("recognized_label")) is int, "invalid routine label")
    require(report["recognized_label"] == observed, "recognizer and routine label differ")
    if observed != expected:
        require(output.returncode == 1 and report.get("expected_label") == expected and
                report.get("real_ticks") == 0 and not aac.exists(), "routine ran on a mismatch")
        return {**reduced, "status": "intent_mismatch", "routine_ticks": 0}
    require(output.returncode == 0, "unexpected routine failure")
    if expected == 0:
        require(report.get("abstained") is True and report.get("real_ticks") == 0 and not aac.exists(),
                "unconfirmed abstention")
        return {**reduced, "status": "verified", "routine_ticks": 0}
    require(report.get("real_ticks") == 1 and report.get("stock_payload_verified") is True and
            report.get("matching_aac_clips") is True and aac.is_file() and aac.stat().st_size > 0,
            "unconfirmed factory payload or AAC")
    decoded = invoke(programs["ffmpeg"], ("-nostdin", "-v", "error", "-xerror",
                     "-protocol_whitelist", "file", "-f", "aac", "-i",
                     str(aac), "-map", "0:a:0", "-f", "s16le", "pipe:1"), b"", env)
    require(decoded.returncode == 0 and decoded.stdout and len(decoded.stdout) % 2 == 0,
            "AAC did not decode")
    return {**reduced, "status": "verified", "routine_ticks": 1, "aac_sha256": digest(aac)}


def retained_view(programs, env, expected, pcm, aac):
    capture = invoke(programs["capture"], ("--nominal-hangover",), pcm, env)
    require(capture.returncode == 0, "capture did not return its disposition")
    result = json.loads(capture.stdout)
    require(isinstance(result, dict), "invalid capture result")
    samples = len(pcm) // 2
    if (result.get("reason") != "classified" or result.get("input_samples") != samples
            or result.get("input_samples_sent") != samples or not result.get("voiced_frames")):
        return {"status": "capture_inconclusive"}
    observation = result["observed"]
    require(type(observation) is int and observation in range(7) and
            result["route"] == DISPATCH.get(observation, "none"), "capture route and intent differ")
    retained = invoke(programs["capture"], ("--pcm-with-hangover",), pcm, env)
    if retained.returncode == 3 and not retained.stdout:
        return {"status": "capture_inconclusive"}
    data = retained.stdout
    require(retained.returncode == 0 and MIN_SAMPLES * 2 <= len(data) <= MAX_SAMPLES * 2 and len(data) % 2 == 0,
            "invalid retained PCM")
    replay = episode_view(programs, env, expected, data, aac)
    require(replay["observed"] == observation, "retained replay and capture intent differ")
    return replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--clock-dir", type=Path)
    for program in ARTIFACTS:
        parser.add_argument("--" + program, type=Path, required=True)
    args = parser.parse_args()

    try:
        spec, programs, env, panel = read_panel(args)
        rows = []
        with tempfile.TemporaryDirectory() as scratch:
            for number, (role, expected, pcm) in enumerate(panel, 1):
                stem = Path(scratch) / str(number)
                rows.append({"episode": number, "role": role, "expected": expected,
                             "raw": episode_view(programs, env, expected, pcm, stem.with_suffix(".raw.aac")),
                             "retained": retained_view(programs, env, expected, pcm,
                                                       stem.with_suffix(".retained.aac"))})
    except (OSError, ValueError, KeyError, TypeError, EOFError, wave.Error, subprocess.TimeoutExpired) as error:
        # Do not publish paths, transcript data or text received from subprocesses.
        reason = str(error) if isinstance(error, PanelError) else type(error).__name__
        parser.exit(2, reason + "\n")
    receipt = {"source_sha_declared": spec["source_sha"], "manifest_sha256": args.manifest_sha256,
               "artifact_sha256": spec["artifact_sha256"], "label_injected": False, "episodes": rows}
    print(json.dumps(receipt, indent=2))
    return int(any(row[view]["status"] != "verified" for row in rows for view in ("raw", "retained")))


if __name__ == "__main__":
    raise SystemExit(main())
