"""Host-only temporal keyword and full-turn verification for the four es-MX intents."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
import platform
import sys
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch import nn

from capture_replay import pcm_from, recordings
from time_lab import digest, file_digest, invoke, supplemental

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "esmx-compactas"))
from pilot import render


FIXTURE = Path(__file__).with_name("fallback_cases.json")
EXTRA = Path(__file__).with_name("keyword_lab_cases.json")
VOICE_SPLITS = {
    "train": (("Paulina", 176), ("Eddy (Español (México))", 159), ("Sandy (Español (México))", 190)),
    "dev": (("Grandpa (Español (México))", 169), ("Shelley (Español (México))", 185)),
    "test": (("Flo (Español (México))", 163), ("Rocko (Español (México))", 191)),
}
INTENTS = {"reject": 0, "butter": 1, "joke": 3, "time": 5, "d6": 6}
LABELS = tuple(INTENTS.values())
REASSIGNED = {
    "Avienta un dado de seis.", "Tira un dado de seis caras.", "Lanza un dado de seis lados.",
}
NEGATION = {"no", "cancela", "olvida", "olvídalo", "ni"}
KEYWORDS = (
    {"mantequilla"},
    {"chiste", "chistes"},
    {"hora", "horas"},
    {"dado", "dados", "tirada", "seis"},
)
CHANNELS, BANDS, MODEL_FRAMES = 24, 20, 128
SEED = 20261127
BANK_KIND = "BB-KEYWORD-LAB-1"
PUB_CONTEXT = ("hora-ajuste-spanishpod-a0056-esmx.wav",
               "0319e9b81c27930015d381c7f6a6c5aa31bc862509c46815814ba681936d3751")


@dataclass(frozen=True)
class Row:
    key: str
    split: str
    intent: int
    phrase: str
    wav: Path
    sha: str | None = None
    voice: tuple | None = None

    def words(self):
        return set(self.phrase.casefold().replace("¿", "").replace("¡", "")
                   .replace(",", " ").replace(".", " ").replace("?", " ").replace("!", " ").split())

    def keywords(self):
        words = self.words()
        return [int(bool(words & terms)) for terms in KEYWORDS]

    def denied(self):
        return int(bool(self.words() & NEGATION))


class Separable(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.spatial = nn.Conv1d(CHANNELS, CHANNELS, 7, dilation=dilation,
                                 padding=3 * dilation, groups=CHANNELS)
        self.mix = nn.Conv1d(CHANNELS, CHANNELS, 1)

    def forward(self, sequence):
        return torch.relu(self.mix(self.spatial(sequence)) + sequence)


class Keywords(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv1d(BANDS, CHANNELS, 5, stride=2, padding=2)
        self.local = Separable(1)
        self.context = Separable(3)
        self.keyword = nn.Conv1d(CHANNELS, len(KEYWORDS), 1)
        self.intent = nn.Linear(5 * CHANNELS, len(LABELS))
        self.negation = nn.Linear(5 * CHANNELS, 1)

    def forward(self, sequence, lengths):
        length = (lengths + 1) // 2
        index = torch.arange(MODEL_FRAMES // 2, device=sequence.device)
        valid = index[None, :] < length[:, None]
        local = self.local(torch.relu(self.stem(sequence)) * valid[:, None]) * valid[:, None]
        local_logits = self.keyword(local).masked_fill(~valid[:, None], -100).amax(dim=-1)
        entire_turn = self.context(local) * valid[:, None]
        mean = entire_turn.sum(dim=-1) / length[:, None]
        # Pool four relative positions so a "no" at the end is distinct from
        # the same word inside an otherwise harmless acoustic context.
        pos = index[None, :] * 4 // length[:, None]
        quarters = [entire_turn.masked_fill(~(valid & (pos == part))[:, None], 0).amax(dim=-1)
                    for part in range(4)]
        pooled = torch.cat([mean, *quarters], dim=1)
        return self.intent(pooled), local_logits, self.negation(pooled).squeeze(-1)


def rows(args, splits):
    original = json.loads(FIXTURE.read_text())
    addition = json.loads(EXTRA.read_text())
    seen = set()
    for split in splits:
        voices = VOICE_SPLITS[split]
        for category, phrases in original[split].items():
            for index, phrase in enumerate(phrases):
                canonical = phrase.lower().strip("¿?¡!., ")
                if canonical in seen:
                    raise ValueError(f"Repeated synthetic text before split: {phrase}")
                seen.add(canonical)
                label = INTENTS["d6" if phrase in REASSIGNED else category]
                suffixes = [str(i) for i in range(len(voices))]
                if split == "train":
                    suffixes.append("ald")
                for suffix in suffixes:
                    key = f"{split}-{category}-{index:02}-{suffix}"
                    yield Row(key, split, label, phrase, args.wav / f"{key}.wav")
        for category, phrases in addition[split].items():
            for index, phrase in enumerate(phrases):
                canonical = phrase.lower().strip("¿?¡!., ")
                if canonical in seen:
                    raise ValueError(f"Repeated synthetic text before split: {phrase}")
                seen.add(canonical)
                for number, voice in enumerate(voices):
                    key = f"{split}-joint-{category}-{index:02}-{number}"
                    yield Row(key, split, INTENTS[category], phrase, Path(f"{key}.wav"), voice=voice)
    if "train" in splits and not args.skip_adjustment:
        name, sha = PUB_CONTEXT
        yield Row("spanishpod_adjustment", "train", INTENTS["time"], "¿Qué horas son?",
                  args.root / "primer-candidato-voces" / name, sha)


def trimmed(pcm):
    data = np.frombuffer(pcm, dtype="<i2").astype(np.int64)
    if not 400 <= len(data) <= 64_000:
        return b""
    windows = np.pad(data, (0, (-len(data)) % 320)).reshape(-1, 320)
    energy = (windows * windows).mean(axis=1)
    gate = max(90 * 90, float(energy.max()) / 512)
    active = np.flatnonzero(energy >= gate)
    if not len(active):
        return b""
    # Keep one 20 ms edge frame on each side and retain all interior pauses.
    begin = max(0, (int(active[0]) - 1) * 320)
    end = min(len(data), (int(active[-1]) + 2) * 320)
    return pcm[2 * begin:2 * end]


def encode(program, pcm):
    pcm = trimmed(pcm)
    if len(pcm) < 800:
        return None
    history = invoke(program, pcm, "word-frames")
    if len(history) % BANDS:
        raise ValueError("C++ feature frontend returned truncated frame")
    sequence = np.frombuffer(history, dtype=np.int8).reshape(-1, BANDS)
    length = len(sequence)
    if length < 8:
        return None
    # Preserve order and local elapsed time on short turns. Average-pool only
    # long turns so the input cap never drops a late cancellation.
    factors = (np.arange(length) * MODEL_FRAMES) // length if length > MODEL_FRAMES else np.arange(length)
    out = np.zeros((BANDS, MODEL_FRAMES), dtype=np.float32)
    if length <= MODEL_FRAMES:
        out[:, :length] = sequence.T
    else:
        sums = np.zeros((MODEL_FRAMES, BANDS), dtype=np.float32)
        np.add.at(sums, factors, sequence)
        out[:] = (sums / np.bincount(factors, minlength=MODEL_FRAMES)[:, None]).T
        length = MODEL_FRAMES
    return out / 32, length


def make_features(args, cases):
    def one(row):
        path = row.wav
        if row.voice:
            with TemporaryDirectory(prefix="bb-keyword-voice-") as product:
                voice, speed = row.voice
                path = render(Path(product), row.key, voice, row.phrase, speed)
                pcm = pcm_from(path)
        else:
            pcm = pcm_from(path, row.sha)
        return row, digest(pcm), encode(args.features, pcm)

    with ThreadPoolExecutor(max_workers=6) as workers:
        return list(workers.map(one, cases))


def batch(cases):
    good = [(row, data) for row, _, data in cases if data is not None]
    refs = [row for row, _ in good]
    x = torch.from_numpy(np.stack([data[0] for _, data in good]))
    n = torch.tensor([data[1] for _, data in good])
    labels = torch.tensor([LABELS.index(row.intent) for row in refs])
    words = torch.tensor([row.keywords() for row in refs], dtype=torch.float32)
    denied = torch.tensor([row.denied() for row in refs], dtype=torch.float32)
    return refs, x, n, labels, words, denied


def train(cases):
    torch.manual_seed(SEED)
    model = Keywords()
    _, x, n, labels, words, denied = batch(cases)
    optim = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.012)
    weights = torch.bincount(labels, minlength=len(LABELS)).float().pow(-0.5)
    cross = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.04)
    bce = nn.BCEWithLogitsLoss()
    for epoch in range(80):
        for subset in torch.randperm(len(x)).split(40):
            noise = torch.randn_like(x[subset]) * 0.05
            noise = noise * (torch.arange(MODEL_FRAMES)[None, None] < n[subset, None, None])
            pred, local, deny = model(x[subset] + noise, n[subset])
            loss = cross(pred, labels[subset]) + 0.6 * bce(local, words[subset]) + 0.35 * bce(deny, denied[subset])
            optim.zero_grad()
            loss.backward()
            optim.step()
        if epoch == 54:
            optim.param_groups[0]["lr"] = 0.0014
    return model.eval()


def quantize(model, path):
    packed = {}
    for name, value in model.state_dict().items():
        source = value.detach().numpy()
        scale = np.float32(max(1e-8, np.abs(source).max() / 127))
        packed[name] = np.clip(np.rint(source / scale), -127, 127).astype(np.int8)
        packed[name + ":scale"] = np.asarray(scale)
    with path.open("xb") as output:
        np.savez(output, **packed)
    return sum(value.nbytes for value in packed.values())


def read_model(path):
    model = Keywords()
    with np.load(path, allow_pickle=False) as values:
        model.load_state_dict({name: torch.from_numpy(
            values[name].astype(np.float32) * values[name + ":scale"]
        ) for name in model.state_dict()})
    return model.eval()


def scores(model, x, n):
    with torch.inference_mode():
        logits, presence, denied = model(x, n)
        margins = torch.stack([logits[:, i] - torch.cat([logits[:, :i], logits[:, i + 1:]], dim=1).amax(dim=1)
                               for i in range(1, len(LABELS))], dim=1)
        # All three verifier heads must support the same intent. This score
        # becomes a fixed per-command threshold only after development replay.
        return torch.minimum(torch.minimum(margins, presence + 1), 0.6 - denied[:, None]).numpy()


def calibration(model, cases):
    _, x, n, labels, _, _ = batch(cases)
    strength = scores(model, x, n)
    truth = labels.numpy()
    thresholds = {}
    for i in range(strength.shape[1]):
        false_alarm = strength[truth != i + 1, i].max()
        thresholds[LABELS[i + 1]] = max(0.1, float(false_alarm) + 0.04)
    return thresholds


def decode(values, limits):
    if values is None:
        return 0
    best = int(np.argmax(values))
    return LABELS[best + 1] if values[best] >= limits[LABELS[best + 1]] else 0


def recipe():
    pilot_dir = Path(__file__).resolve().parent
    files = (Path(__file__), FIXTURE, EXTRA, pilot_dir / "time_lab.py",
             pilot_dir / "capture_replay.py", pilot_dir / "replay.py",
             pilot_dir.parent / "esmx-compactas" / "pilot.py")
    return {path.name: file_digest(path) for path in files}


def environment():
    return {"python": platform.python_version(), "numpy": np.__version__,
            "torch": str(torch.__version__), "macos": platform.mac_ver()[0],
            "synthetic_voices": {split: list(map(list, voices)) for split, voices in VOICE_SPLITS.items()}}


def freeze(args):
    prepared = make_features(args, rows(args, ("train", "dev")))
    models = {}
    evidence = {}
    for split in ("train", "dev"):
        entries = [(row, pcmhash, value) for row, pcmhash, value in prepared if row.split == split]
        current = {pcmhash for _, pcmhash, _ in entries}
        if any(key in evidence for key in current):
            raise ValueError("Exact PCM overlap between train/dev")
        evidence.update({pcmhash: row.key for row, pcmhash, _ in entries})
        models[split] = entries
    model = train(models["train"])
    weights = args.output / "keyword-host.npz"
    weight_memory = quantize(model, weights)
    float_from_int8 = read_model(weights)
    limits = calibration(float_from_int8, models["dev"])
    refs, x, n, _, _, _ = batch(models["dev"])
    measured = [(row.intent, decode(values, limits)) for row, values in zip(refs, scores(float_from_int8, x, n))]
    report = {
        "kind": BANK_KIND, "recipe": recipe(), "environment": environment(),
        "features_sha256": file_digest(args.features),
        "weight_sha256": file_digest(weights), "weight_bytes_int8_plus_scales": weight_memory,
        "host_container_bytes": weights.stat().st_size, "seed": SEED,
        "calibration": limits, "development_correct_per_label": {
            label: [sum(expected == label == observed for expected, observed in measured),
                    sum(expected == label for expected, _ in measured)] for label in LABELS},
        "train_counts": dict(Counter(row.intent for row, _, data in models["train"] if data is not None)),
        "discarded_feature_failures": [row.key for row, _, vector in prepared if vector is None],
        "pcm_hash_to_adjustment_id": evidence,
    }
    with (args.output / "frozen.json").open("x") as target:
        json.dump(report, target, indent=2)
    print(json.dumps({key: val for key, val in report.items() if key != "pcm_hash_to_adjustment_id"}, indent=2))


def raw_scores(model, vector):
    if vector is None:
        return None
    x, n = vector
    return scores(model, torch.from_numpy(x[None, ...]), torch.tensor([n]))[0]


def replay(args):
    config = json.loads((args.output / "frozen.json").read_text())
    weights = args.output / "keyword-host.npz"
    if (config["kind"] != BANK_KIND or config["recipe"] != recipe() or config["environment"] != environment()
            or config["features_sha256"] != file_digest(args.features)
            or config["weight_sha256"] != file_digest(weights)):
        raise ValueError("Frozen bank or recipe hash mismatch")
    model = read_model(weights)
    limits = {int(k): value for k, value in config["calibration"].items()}
    panel = list(recordings(args.root))
    if len(panel) != 112:
        raise ValueError(f"Expected historical 112-case panel, got {len(panel)}")
    if args.coerll_supplemental:
        panel.append(supplemental(args.root, args.coerll_supplemental))
    result = []

    def evaluate_pcm(pcm):
        baseline = int(invoke(args.reference, pcm)) if pcm and len(pcm) <= 128_000 else 0
        candidate = decode(raw_scores(model, encode(args.features, pcm)), limits)
        return {"baseline": baseline, "candidate": candidate, "combined": baseline or candidate}

    for key, split, category, expected, path, seal in panel:
        pcm = pcm_from(path, seal)
        seen = digest(pcm) in config["pcm_hash_to_adjustment_id"]
        if seen and split not in ("synthetic_panel", "synthetic_aliases"):
            raise ValueError(f"Training PCM overlapped a nonhistorical panel: {key}")
        raw = evaluate_pcm(pcm)
        capture = evaluate_pcm(invoke(args.capture, pcm, "--pcm-with-hangover"))
        result.append({"id": key, "split": split, "category": category, "expected": expected,
                       "pcm_sha256": digest(pcm),
                       "seen_pcm": seen, "raw": raw, "retained_capture": capture})

    # Read or render synthetic test only after validating the frozen recipe/model.
    for row, pcm_hash, features in make_features(args, rows(args, ("test",))):
        if pcm_hash in config["pcm_hash_to_adjustment_id"]:
            raise ValueError(f"Frozen synthetic test overlapped training/development: {row.key}")
        verdict = decode(raw_scores(model, features), limits)
        result.append({"id": row.key, "split": "synthetic_test", "expected": row.intent,
                       "pcm_sha256": pcm_hash,
                       "negation": bool(row.denied()), "candidate": verdict})

    synthetic = [row for row in result if row["split"] == "synthetic_test"]
    modes = ("raw", "retained_capture")
    panels = {
        "historical_112": result[:112],
        "public_historical_includes_development": [row for row in result if row["split"].startswith("public")],
        "human_time_held_out_from_fit": [row for row in result if row["split"] == "public_reserved_time"],
        "human_d6_held_out_from_fit": [row for row in result if row["split"] == "public_reserved_d6"],
        "human_supplemental": [row for row in result if row["split"] == "supplemental_human_time"],
    }
    summary = {set_name: {mode: {
        "new_wrong_dispatch": [row["id"] for row in cohort if row[mode]["baseline"] == 0
                               and row[mode]["combined"] not in (0, row["expected"])],
        "detected_by_label": {
            label: [sum(row[mode][arm] == label for row in cohort if row["expected"] == label)
                    for arm in ("baseline", "candidate", "combined")]
            for label in LABELS[1:]},
        "denominators": dict(Counter(row["expected"] for row in cohort)),
    } for mode in modes} for set_name, cohort in panels.items()}
    print(json.dumps({
        "weight_sha256": config["weight_sha256"], "reference_sha256": file_digest(args.reference),
        "capture_sha256": file_digest(args.capture), "summary": summary,
        "synthetic_test": {label: [sum(row["expected"] == label == row["candidate"] for row in synthetic),
                                   sum(row["expected"] == label for row in synthetic)] for label in LABELS},
        "synthetic_negation_activated": [row["id"] for row in synthetic if row["negation"] and row["candidate"]],
        "rows": result,
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "replay"))
    for name in ("root", "wav", "features", "output"):
        parser.add_argument("--" + name, type=lambda path: Path(path).resolve(strict=True), required=True)
    for name in ("capture", "reference", "coerll-supplemental"):
        parser.add_argument("--" + name, type=lambda path: Path(path).resolve(strict=True))
    parser.add_argument("--skip-adjustment", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(3)
    if args.phase == "freeze":
        freeze(args)
    elif args.reference and args.capture:
        replay(args)
    else:
        parser.error("replay requires --reference and --capture")


if __name__ == "__main__":
    main()
