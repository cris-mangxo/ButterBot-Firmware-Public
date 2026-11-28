"""Train a small contrastive whole-turn classifier with one independent human anchor."""

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from capture_replay import pcm_from, recordings
from clock_anchor_provenance import verify
from clock_elastic_lab import PILOT, example_id, matrix, read_wav
from keyword_lab import VOICE_SPLITS, render
from phoneme_reject_lab import load_negatives
from phoneme_replay_lab import strip_digital_edges
from time_lab import digest, file_digest, invoke


SEED = 2026112801
FRAMES, BANDS, WIDTH = 64, 20, 24
STAGES = { "train": [
    ("¿Qué hora es?", 1), ("¿Qué hora es ahora?", 1), ("¿Qué horas son?", 1),
    ("¿Me dices la hora?", 1), ("¿Qué hora tienes?", 1), ("¿Qué hora será ahorita?", 1),
    ("¿Qué hora marca?", 1), ("¿Me da la hora?", 1), ("¿Cuál es la hora actual?", 1),
    ("¿A qué hora es?", 0), ("¿A qué horas es?", 0), ("¿A qué hora?", 0),
    ("¿Qué hora era?", 0), ("¿Qué hora va a ser?", 0), ("¿A qué hora llegamos?", 0),
    ("¿Qué hora era entonces?", 0), ("¿Qué hora es mañana?", 0),
    ("¿Qué hora es en Monterrey?", 0), ("¿Qué hora es de la reunión?", 0),
    ("¿Qué hora es cuando llega el vuelo?", 0), ("¿Qué hora tiene tu vuelo?", 0),
    ("No me digas qué hora es", 0), ("Mejor no me digas la hora", 0),
    ("¿A qué hora es la cena?", 0), ("Quiero saber a qué hora es", 0),
    ("¿A qué hora sale el vuelo?", 0), ("¿A qué hora es la clase?", 0),
    ("¿Qué oro es?", 0), ("¿Qué olor es?", 0), ("¿Qué es eso?", 0),
    ("Ahorita es temprano", 0), ("Es hora de comer", 0), ("Es la una", 0),
    ("Échame un chiste", 0), ("Pásame la mantequilla", 0), ("Tira un dado de seis", 0),
], "dev": [
    ("¿Oye qué hora es?", 1), ("¿Qué hora es ahorita?", 1), ("Dime qué horas son", 1),
    ("¿Me puedes dar la hora?", 1), ("¿Sabes qué hora es?", 1),
    ("¿Oye a qué hora es?", 0), ("¿Qué hora es en Ciudad de México?", 0),
    ("¿A qué hora sale el avión?", 0), ("¿Qué hora era ayer?", 0),
    ("Me dices la hora de tu cita", 0), ("Ya no me digas la hora", 0),
    ("¿Qué hora es la de la salida?", 0), ("Qué buena hora es", 0),
], "test": [
    ("¿Y qué hora es?", 1), ("¿Qué hora será ahora?", 1), ("¿Qué hora tenemos?", 1),
    ("¿Podrías decirme qué hora es?", 1), ("¿Me pasas la hora?", 1),
    ("¿Y a qué hora es?", 0), ("¿A qué hora comenzamos?", 0),
    ("¿Qué hora es en Tijuana?", 0), ("¿Qué hora era anoche?", 0),
    ("¿Para qué hora es?", 0), ("Ya no, no me des la hora", 0),
    ("¿Qué hora es la del inicio?", 0), ("Pon una alarma a las seis", 0),
]}


class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv1d(BANDS, WIDTH, 5, stride=2, padding=2)
        self.local = nn.Conv1d(WIDTH, WIDTH, 5, groups=WIDTH, padding=2)
        self.mix = nn.Conv1d(WIDTH, WIDTH, 1)
        self.context = nn.Conv1d(WIDTH, WIDTH, 5, dilation=2, groups=WIDTH, padding=4)
        self.out = nn.Linear(WIDTH * 8 + 1, 1)

    def forward(self, x, length):
        z = F.relu(self.stem(x))
        z = F.relu(self.mix(self.local(z)) + z)
        z = F.relu(self.context(z) + z)
        return self.out(torch.cat((z.reshape((-1, WIDTH, 8, 4)).mean(-1).flatten(1), length[:, None]), 1))[:, 0]


def tensor(mel, rng=None):
    value = mel.astype(np.float32) / 32
    length = len(value)
    value -= value.mean(axis=0, keepdims=True)
    if rng is not None:
        length *= rng.uniform(0.82, 1.20)
        time = rng.uniform(0.85, 1.15, 4)
        time /= time.mean()
        x = np.interp(np.linspace(0, 4, FRAMES), np.arange(5), np.r_[0, np.cumsum(time)])
        value *= rng.uniform(0.75, 1.20, (1, BANDS))
        value += rng.normal(0, rng.uniform(0, .075), value.shape)
        if rng.random() < .30:
            lo = int(rng.integers(BANDS - 1))
            value[:, lo:lo + 2] = 0
    else:
        x = np.linspace(0, 4, FRAMES)
    resampled = np.array([np.interp(x * (len(value)-1)/4, np.arange(len(value)), band)
                          for band in value.T], dtype=np.float32)
    return resampled, np.float32(np.log(length / 70))


def pack(rows, rng=None):
    encoded = [tensor(mel, rng) for mel in rows]
    values, lengths = zip(*encoded)
    return torch.from_numpy(np.stack(values)), torch.from_numpy(np.array(lengths))


@torch.inference_mode()
def predict(model, rows):
    return model(*pack(rows)).numpy()


def synthesize(args):
    args.output.mkdir(mode=0o700)
    tasks = [(split, i, goal, text, j, voice, speed) for split, rows in STAGES.items()
             for i, (text, goal) in enumerate(rows) for j, (voice, speed) in enumerate(VOICE_SPLITS[split])]

    def generate(row):
        split, i, goal, text, j, voice, speed = row
        identifier = f"{split}-{i:02}-{j}"
        audio = render(args.output, identifier, voice, text, speed)
        return dict(id=identifier, split=split, goal=goal, voice=voice, sha256=file_digest(audio))

    with ThreadPoolExecutor(max_workers=6) as executor:
        manifest = list(executor.map(generate, tasks))
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def fit(args):
    roles_sha = verify(args.roles, donor=args.donor, agenda=args.agenda)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    donor_meta = json.loads(args.donor.read_text())
    agenda_meta = json.loads(args.agenda.read_text())
    if donor_meta["role"] != "local_calibration_only" or agenda_meta["role"] != "local_calibration_only":
        raise ValueError("Human fitting inputs require a calibration-only declaration")
    donor_pcm = read_wav(args.donor, donor_meta)
    if digest(donor_pcm) != donor_meta["pcm_sha256"]:
        raise ValueError("Wrong donor PCM")
    don = matrix(args.frontend, donor_pcm)
    occupied = {digest(donor_pcm)}
    synth_index = args.synthetic / "manifest.json"
    synth = json.loads(synth_index.read_text())
    synthetic = {key: [[], []] for key in STAGES}
    for row in synth:
        identifier = example_id(row["id"], r"(train|dev|test)-[0-9]{2}-[0-9]")
        split = row["split"]
        if split not in STAGES or not identifier.startswith(split + "-") or row["goal"] not in (0, 1):
            raise ValueError("Unexpected synthetic label or split")
        pcm = pcm_from(args.synthetic / (identifier + ".wav"), row["sha256"])
        mel = matrix(args.frontend, pcm)
        if mel is None or len(mel) < 12:
            raise ValueError("Invalid synthetic utterance")
        occupied.add(digest(pcm))
        synthetic[row["split"]][row["goal"]].append(mel)
    control_index = args.negatives / "index.json"
    human_controls = load_negatives(args.negatives, args.speakers, file_digest(args.frontend))
    occupied.update(row["effective_pcm_sha256"] for row in json.loads(control_index.read_text())["records"].values())
    agenda = []
    for row in agenda_meta["rows"]:
        pcm = read_wav(args.agenda, row)
        occupied.add(digest(pcm))
        agenda.append(matrix(args.frontend, pcm))
    threshold_pool = synthetic["dev"][0] + human_controls["dev"] + agenda
    train_positives = synthetic["train"][1]
    train_synthetic_neg = synthetic["train"][0]
    human_train = human_controls["train"]
    report = {"model_floats": sum(p.numel() for p in Detector().parameters()), "baseline": PILOT,
              "occupied_pcm": sorted(occupied), "arms": {}, "pins": {
                  "source": file_digest(Path(__file__)), "frontend": file_digest(args.frontend),
                  "donor": file_digest(args.donor), "agenda": file_digest(args.agenda),
                  "synthesis": file_digest(synth_index), "chm": file_digest(control_index),
                  "speakers": file_digest(args.speakers), "roles": roles_sha}}
    args.output.mkdir(mode=0o700)
    for mode in ("without_donor", "with_donor"):
        torch.manual_seed(SEED)
        rng = np.random.default_rng(SEED)
        model = Detector()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.004)
        for _ in range(640):
            pos = [train_positives[int(j)] for j in rng.integers(len(train_positives), size=14)]
            pos += [don] * 2 if mode == "with_donor" else [train_positives[int(j)] for j in rng.integers(len(train_positives), size=2)]
            neg = [train_synthetic_neg[int(j)] for j in rng.integers(len(train_synthetic_neg), size=10)]
            neg += [human_train[int(j)] for j in rng.integers(len(human_train), size=6)]
            out = model(*pack(pos + neg, rng))
            loss = F.binary_cross_entropy_with_logits(out, torch.tensor([1.] * 16 + [0.] * 16))
            if not loss.isfinite():
                raise RuntimeError("Nonfinite training loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        weights = args.output / (mode + ".npz")
        np.savez_compressed(weights, **{name: w.detach().numpy() for name, w in model.state_dict().items()})
        pools = dict(train=synthetic["train"], dev=[threshold_pool, synthetic["dev"][1]], test=synthetic["test"])
        threshold = float(predict(model, threshold_pool).max())
        counts = {split: {"negative": len(neg), "positive": len(pos),
                          "false": int((predict(model, neg) > threshold).sum()),
                          "correct": int((predict(model, pos) > threshold).sum())}
                  for split, (neg, pos) in pools.items()}
        report["arms"][mode] = {"threshold": threshold, "counts": counts, "donor_score": float(predict(model, [don])[0]),
                                 "weights_sha256": file_digest(weights)}
    report["selected"] = max(report["arms"], key=lambda k: (report["arms"][k]["counts"]["dev"]["correct"], k == "with_donor"))
    path = args.output / "freeze.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def replay(args):
    frozen = args.frozen / "freeze.json"
    manifest = json.loads(frozen.read_text())
    if manifest["pins"]["source"] != file_digest(Path(__file__)) or manifest["pins"]["frontend"] != file_digest(args.frontend):
        raise ValueError("Replay inputs differ from the fitting seal")
    models = {}
    for key, arm in manifest["arms"].items():
        path = args.frozen / (key + ".npz")
        if file_digest(path) != arm["weights_sha256"]:
            raise ValueError("Weights changed after the fitting seal")
        model = Detector().eval()
        with np.load(path, allow_pickle=False) as weights:
            model.load_state_dict({name: torch.from_numpy(weights[name]) for name in weights.files})
        models[key] = model, arm["threshold"]
    if verify(args.roles, panel=args.panel, quality=args.quality) != manifest["pins"]["roles"]:
        raise ValueError("Replay role seal differs from fitting")
    occupied = set(manifest["occupied_pcm"])
    quality = json.loads(args.quality.read_text())["labels"]
    panel = json.loads(args.panel.read_text())
    if panel["host_sha256"] != file_digest(args.host) or panel["capture_sha256"] != file_digest(args.capture):
        raise ValueError("Host or capture does not match the held-out seal")
    def outcome(pcm, baseline=None):
        if digest(pcm) in occupied or digest(strip_digital_edges(pcm)) in occupied:
            raise ValueError("Evaluation revealed fitting audio")
        if not pcm or len(pcm) > 128_000:
            return {"baseline": "outside_window", "intent": 0}
        if baseline is None:
            baseline = json.loads(invoke(args.host, pcm, "--decision"))
            baseline["intent"] = int(invoke(args.host, pcm))
        mel = matrix(args.frontend, pcm)
        disposition = baseline.get("decision", baseline.get("disposition"))
        gate = disposition == "no_match" and baseline["intent"] == 0
        view = {"baseline": disposition, "intent": baseline["intent"]}
        for name, (model, cut) in models.items():
            margin = None if mel is None or len(mel) < 12 else float(predict(model, [mel])[0] - cut)
            view[name] = {"margin": margin, "eligible_accept": gate and margin is not None and margin > 0}
        return view

    rows = []
    for row in panel["results"]:
        identifier = example_id(row["id"], r"[CT][0-9]{2}-[0-9]{3}")
        decision = {"id": identifier, "expected": row["expected"],
                    "confirmed": quality[identifier]["include_in_confirmed_denominator"]}
        for mode in ("raw", "retained"):
            wav = (args.pcm / f"{identifier}.{mode}.pcm").read_bytes()
            if digest(wav) != row[mode]["pcm_sha256"]:
                raise ValueError("Held-out PCM changed")
            decision[mode] = outcome(wav, row[mode])
        rows.append(decision)
    history = []
    duplicates = 0
    if args.history:
        for identifier, split, category, expected, path, pin in recordings(args.history):
            pcm = pcm_from(path, pin)
            if digest(pcm) in occupied:
                duplicates += 1
                continue
            retained = invoke(args.capture, pcm, "--pcm-with-hangover")
            history.append({"id": identifier, "split": split, "category": category, "expected": expected,
                            "raw": outcome(pcm), "retained": outcome(retained)})
    print(json.dumps({"frozen_sha256": file_digest(frozen), "selected": manifest["selected"], "rows": rows,
                      "panel_sha256": file_digest(args.panel), "quality_sha256": file_digest(args.quality),
                      "host_sha256": panel["host_sha256"], "capture_sha256": panel["capture_sha256"],
                      "history": history, "historical_duplicate_pcm_excluded": duplicates},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("generate", "fit", "replay"))
    for name in ("output", "synthetic", "frontend", "donor", "agenda", "negatives", "speakers",
                 "frozen", "quality", "panel", "pcm", "host", "capture", "history", "roles"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    if args.stage != "generate" and args.roles is None:
        parser.error("fit and replay require --roles")
    {"generate": synthesize, "fit": fit, "replay": replay}[args.stage](args)
