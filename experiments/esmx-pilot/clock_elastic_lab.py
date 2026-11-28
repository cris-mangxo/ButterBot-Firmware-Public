"""Evaluate an independent human clock anchor with range-limited whole-turn matching."""

import argparse
from collections import Counter
from pathlib import Path
import json
import re

import numpy as np

from capture_replay import pcm_from, recordings
from clock_anchor_provenance import verify
from joke_contrast_lab import fit, score
from phoneme_reject_lab import load_negatives
from phoneme_replay_lab import native_frames, strip_digital_edges
from time_lab import digest, file_digest, invoke


FRAMES, CHANNELS, WARP = 40, 10, 6
PILOT = "adce9515712f67d1a03e174866c54d298702396f"
CLOCK = 5
DCT = np.cos(np.pi * (np.arange(20)[:, None] + .5) * np.arange(1, CHANNELS + 1) / 20)
DCT *= np.sqrt(2 / 20) / np.sqrt(np.arange(1, CHANNELS + 1))


def cepstra(mel, delta):
    transformed = mel.astype(np.float64) / 32 @ DCT
    transformed -= transformed.mean(axis=0)
    pos = np.linspace(0, len(mel) - 1, FRAMES)
    temporal = np.array([np.interp(pos, np.arange(len(mel)), band) for band in transformed.T]).T
    if delta:
        edge = np.pad(temporal, ((1, 1), (0, 0)), mode="edge")
        temporal = np.column_stack((temporal, edge[2:] - edge[:-2]))
    return temporal


def align(query, templates, warp=WARP):
    local = np.mean(np.abs(query[:, None, None, :] - templates[None, :, :, :]), axis=-1)
    prior = np.full((len(query) + 1, len(templates)), np.inf)
    prior[0] = 0
    for row in range(len(query)):
        current = np.full_like(prior, np.inf)
        for col in range(max(0, row - warp), min(row + warp + 1, len(query))):
            current[col + 1] = local[row, :, col] + np.minimum(
                prior[col], np.minimum(prior[col + 1], current[col]))
        prior = current
    return prior[-1] / len(query)


def matrix(frontend, pcm):
    return native_frames(frontend, strip_digital_edges(pcm))


def example_id(value, pattern):
    if not isinstance(value, str) or re.fullmatch(pattern, value) is None:
        raise ValueError("Unexpected recording ID")
    return value


def feature_matrix(mel, bank, delta):
    if mel is None or len(mel) < 12:
        return None
    query = cepstra(mel, delta)
    positive, negative = bank["positive"], bank["negative"]
    yes, no = align(query, positive).min(), align(query, negative).min()
    prefix_yes = align(query[:12], positive[:, :12, :], 3).min()
    prefix_no = align(query[:12], negative[:, :12, :], 3).min()
    tail_yes = align(query[-14:], positive[:, -14:, :], 3).min()
    tail_no = align(query[-14:], negative[:, -14:, :], 3).min()
    return np.array([-yes, no - yes, -prefix_yes, prefix_no - prefix_yes,
                     -tail_yes, tail_no - tail_yes, len(mel) / 100,
                     -abs(len(mel) - bank["donor_frames"]) / 100,
                     -np.mean(np.abs(query))])


def count(rows, rule):
    result = Counter("positive" if row[1] else "negative" for row in rows)
    for _, goal, value in rows:
        if value is not None and score(value, rule) > rule["threshold"]:
            result["accepted_positive" if goal else "accepted_negative"] += 1
    return dict(result)


def read_wav(manifest, entry):
    file = entry["wav"]
    if not re.fullmatch(r"[a-zA-Z0-9_.-]+\.wav", file):
        raise ValueError("Unexpected calibration filename")
    return pcm_from(manifest.parent / file, entry["sha256"])


def train(args):
    roles_sha = verify(args.roles, donor=args.donor, agenda=args.agenda)
    synthetic = json.loads(args.synthetic.read_text())
    addon = json.loads(args.donor.read_text())
    agenda = json.loads(args.agenda.read_text())
    if addon["role"] != "local_calibration_only" or agenda["role"] != "local_calibration_only":
        raise ValueError("Inputs are not authorized fitting voices")
    donor_pcm = read_wav(args.donor, addon)
    if digest(donor_pcm) != addon["pcm_sha256"]:
        raise ValueError("Unexpected donor PCM")
    donor = matrix(args.frontend, donor_pcm)
    agenda_pcm = [read_wav(args.agenda, row) for row in agenda["rows"]]
    items = []
    for row in synthetic:
        part = row["split"]
        if part not in {"train", "dev", "test"}:
            raise ValueError("Unexpected synthetic split")
        key = example_id(row["id"], rf"{part}-[0-9]{{2}}-[01]-[0-9]")
        pcm = pcm_from(args.synthetic.parent / (key + ".wav"), row["wav_sha256"])
        items.append((row["id"], part, row["time_now"], matrix(args.frontend, pcm), row["text"], digest(pcm)))
    manifest = args.negatives / "index.json"
    prior = load_negatives(args.negatives, args.speakers, file_digest(args.frontend))
    development = [*items, *(("CHM-" + part + "-" + str(i), part, False, m, "", None)
                            for part, group in prior.items() for i, m in enumerate(group)),
                   *(("agenda-" + str(i), "dev", False, matrix(args.frontend, wav), "", digest(wav))
                     for i, wav in enumerate(agenda_pcm))]
    pcm_seals = {digest(donor_pcm), *(row[5] for row in development if row[5]),
                 *(row["effective_pcm_sha256"] for row in json.loads(manifest.read_text())["records"].values())}
    result = {"schema": "bb-esmx-clock-elastic-1", "baseline_source": PILOT, "occupied_pcm": sorted(pcm_seals),
              "pins": {name: file_digest(path) for name, path in {
                  "script": Path(__file__), "frontend": args.frontend, "donor": args.donor,
                  "agenda": args.agenda, "synthetic": args.synthetic, "chm": manifest, "speakers": args.speakers}.items()},
              "arms": {}}
    result["pins"]["roles"] = roles_sha
    for topology in ("without_donor", "with_donor"):
        for delta in (False, True):
            pos = [(key, m) for key, part, goal, m, phrase, _ in items
                   if part == "train" and goal and phrase == "Qué hora es"]
            if topology == "with_donor":
                pos.append(("heroico_donor", donor))
            neg = [(key, m) for key, part, goal, m, _, _ in items if part == "train" and not goal]
            bank = {"positive": np.stack([cepstra(m, delta) for _, m in pos]),
                    "negative": np.stack([cepstra(m, delta) for _, m in neg]),
                    "donor_frames": len(donor) if topology == "with_donor" else round(np.mean([len(m) for _, m in pos]))}
            features = {part: [(key, goal, feature_matrix(m, bank, delta))
                               for key, split, goal, m, _, _ in development if split == part]
                        for part in ("train", "dev", "test")}
            # Exclude a training utterance's own template when estimating fitting features.
            for i, (key, goal, _) in enumerate(features["train"]):
                originals = pos if goal else neg
                duplicates = [k for k, (identifier, _) in enumerate(originals) if identifier == key]
                if duplicates:
                    mel = next(m for identifier, _, _, m, _, _ in items if identifier == key)
                    kind = "positive" if goal else "negative"
                    filtered_bank = dict(bank, **{kind: np.delete(bank[kind], duplicates, axis=0)})
                    features["train"][i] = (key, goal, feature_matrix(mel, filtered_bank, delta))
            rule = fit([(goal, value) for _, goal, value in features["train"]], tuple(range(9)))
            rule["threshold"] = max(score(x, rule) for part in ("train", "dev")
                                    for _, goal, x in features[part] if not goal and x is not None)
            key = topology + ("_delta" if delta else "_static")
            result["arms"][key] = {"rule": rule, "delta": delta,
                                   "templates": {k: v.tolist() if isinstance(v, np.ndarray) else v
                                                 for k, v in bank.items()},
                                   "counts": {part: count(rows, rule) for part, rows in features.items()},
                                   "reference_ids": {"positive": [n for n, _ in pos], "negative":[n for n, _ in neg]}}
    # Use the declared simulation-development gain and deterministic tie breaking.
    result["chosen"] = max(result["arms"], key=lambda k: (result["arms"][k]["counts"]["dev"].get("accepted_positive", 0),
                                                          k.startswith("with_donor"), not k.endswith("_delta")))
    return result


def predict(args, frozen, data, decision):
    mel = matrix(args.frontend, data)
    if mel is None:
        return {}
    result = {}
    for key, arm in frozen["arms"].items():
        values = feature_matrix(mel, arm["templates"], arm["delta"])
        if values is None:
            continue
        margin = score(values, arm["rule"]) - arm["rule"]["threshold"]
        result[key] = {"score_margin": margin, "proposal": margin > 0, "eligible": decision == "no_match" and margin > 0}
    return result


def replay(args):
    frozen = json.loads(args.frozen.read_text())
    if (frozen["pins"]["script"] != file_digest(Path(__file__))
            or frozen["pins"]["frontend"] != file_digest(args.frontend)):
        raise ValueError("Replay script or extractor does not match freeze")
    for arm in frozen["arms"].values():
        for kind in ("positive", "negative"):
            arm["templates"][kind] = np.array(arm["templates"][kind])
    if verify(args.roles, panel=args.panel, quality=args.quality) != frozen["pins"]["roles"]:
        raise ValueError("Replay role seal differs from fitting")
    occupied = set(frozen["occupied_pcm"])

    def outcome(data, expected_hash=None, saved=None):
        if expected_hash and digest(data) != expected_hash:
            raise ValueError("Held-out waveform does not match the sealed evaluation")
        if digest(data) in occupied or digest(strip_digital_edges(data)) in occupied:
            raise ValueError("Evaluation revealed fitting audio")
        if not data or len(data) > 128_000:
            return {"intention": 0, "decision": "outside_window", "arms": {}}
        if saved is None:
            intent = int(invoke(args.host, data))
            decision = json.loads(invoke(args.host, data, "--decision"))["disposition"]
        else:
            intent, decision = saved["intent"], saved["decision"]
        arms = predict(args, frozen, data, decision) if not intent else {}
        return {"intention": intent, "decision": decision, "arms": arms}

    heldout = json.loads(args.panel.read_text())
    if (heldout["host_sha256"] != file_digest(args.host) or heldout["capture_sha256"] != file_digest(args.capture)):
        raise ValueError("Host/capture mismatches the sealed baseline")
    quality = json.loads(args.quality.read_text())["labels"]
    rows = []
    for item in heldout["results"]:
        key = example_id(item["id"], r"[CT][0-9]{2}-[0-9]{3}")
        row = {"id": key, "confirmed": quality[key]["include_in_confirmed_denominator"],
               "expected": item["expected"]}
        for view in ("raw", "retained"):
            pcm = (args.pcm / f"{key}.{view}.pcm").read_bytes()
            row[view] = outcome(pcm, item[view]["pcm_sha256"], saved=item[view])
        rows.append(row)
    history = []
    duplicates = 0
    if args.history:
        for identifier, split, category, expected, path, digest_wav in recordings(args.history):
            pcm = pcm_from(path, digest_wav)
            if digest(pcm) in occupied:
                duplicates += 1
                continue
            retained = invoke(args.capture, pcm, "--pcm-with-hangover")
            history.append({"id": identifier, "split": split, "category": category, "expected": expected,
                            "raw": outcome(pcm), "retained": outcome(retained)})
    return {"freeze_sha256": file_digest(args.frozen), "panel_sha256": file_digest(args.panel),
            "quality_sha256": file_digest(args.quality), "host_sha256": heldout["host_sha256"],
            "capture_sha256": heldout["capture_sha256"], "chosen": frozen["chosen"],
            "heroico": rows, "history": history, "historical_duplicate_pcm_excluded": duplicates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fit", "replay"))
    for key in ("frontend", "donor", "agenda", "synthetic", "negatives", "speakers", "frozen",
                "host", "capture", "panel", "quality", "pcm", "history"):
        parser.add_argument("--" + key, type=Path)
    parser.add_argument("--roles", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(train(args) if args.stage == "fit" else replay(args), allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
