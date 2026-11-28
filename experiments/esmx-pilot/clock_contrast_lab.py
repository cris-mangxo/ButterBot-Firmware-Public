"""Evaluate a frozen Hora-only contrast filter without modifying the firmware."""

from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json

import numpy as np
import torch

from capture_replay import pcm_from
from keyword_lab import LABELS, trimmed
from phoneme_confirm_lab import (human_anchors, local_ctc_alignment, model_output,
                                 natural_adjustment_hashes, templates)
from phoneme_teacher_lab import phonemes
from phoneme_reject_lab import load_negatives
from phoneme_replay_lab import evaluate, frozen_model, native_frames
from phoneme_student_lab import load_cache
from time_lab import file_digest, invoke

CLOCK = 5
CLOCK_INDEX = LABELS.index(CLOCK)
FEATURES = ['ctc_deficit', 'span_fraction', 'edge_early', 'edge_late', 'turn_margin',
            'negation_clearance', 'sqrt_frames', 'greedy_phone_edit']
METHOD = 'bb-clock-contrast-temporal-3'
NEGATIVE_PHRASES = (
    'a qué hora es la cita', 'qué hora es la reunión', 'a qué hora llega el camión',
    'qué hora era', 'no me digas qué hora es', 'te digo la hora en un rato',
    'nos vemos a la misma hora', 'tira un dado de seis',
)
FEATURES += ['ctc_full_cost', 'ctc_contrast_margin', 'best_negative_full_cost']
FEATURES += [f'posterior-quarter-{quarter}-phone-{phone}' for quarter in range(4) for phone in range(32)]




def edit_distance(a, b):
    distances = list(range(len(b) + 1))
    for i, source in enumerate(a, 1):
        current = [i]
        for j, target in enumerate(b, 1):
            current.append(min(distances[j] + 1, current[-1] + 1,
                               distances[j-1] + (source != target)))
        distances = current
    return distances[-1]


@torch.inference_mode()
def clock_features(model, matrix, template_groups):
    if matrix is None:
        return None
    phones, intents, denied = model_output(model, matrix)
    if denied.item() >= 0:
        return None
    paths, negatives = template_groups
    logit = phones[0].numpy()
    count = len(logit)
    candidates = []
    for path in paths:
        cost, interval = local_ctc_alignment(logit, path, track_span=True)
        if interval is not None:
            candidates.append((cost, interval, path))
    if not candidates:
        return None
    cost, (start, end), path = min(candidates)
    emitted = logit.argmax(axis=1)
    selected = emitted[np.r_[True, emitted[1:] != emitted[:-1]]]
    selected = selected[selected != 0]
    rank = intents[0].numpy()
    other = np.delete(rank, CLOCK_INDEX).max()
    group = paths + negatives
    spans = torch.tensor([len(tokens) for tokens in group])
    labels = torch.tensor([phone for tokens in group for phone in tokens])
    as_logprobs = phones.log_softmax(-1).transpose(0,1).expand(-1, len(group), -1)
    likelihood = torch.nn.functional.ctc_loss(as_logprobs, labels, torch.full((len(group),), count),
                                              spans, reduction='none').numpy() / count
    valid, invalid = min(likelihood[:len(paths)]), min(likelihood[len(paths):])
    if not np.isfinite([valid, invalid]).all():
        return None
    posterior = phones.softmax(-1)[0].numpy()
    quarters = np.arange(count) * 4 // count
    temporal = np.concatenate([posterior[quarters == quarter].mean(axis=0) for quarter in range(4)])
    scores = np.array([cost, (end-start)/count, start/count, (count-end)/count,
                     rank[CLOCK_INDEX] - other, -denied.item(),
                     np.sqrt(count), edit_distance(selected, path)/len(path), valid, invalid-valid, invalid])
    return np.concatenate([scores, temporal])


def row_specs(model, commands, negatives, paths):
    rows = []
    for row, matrix, *_ in commands:
        positive = row['intent'] == CLOCK and not row['denied']
        rows.append((positive, clock_features(model, matrix, paths)))
    rows.extend((False, clock_features(model, matrix, paths)) for matrix in negatives)
    return rows


def fit_model(rows):
    complete = [(goal, feats) for goal, feats in rows if feats is not None]
    y = np.asarray([int(goal) for goal, _ in complete])
    x = np.stack([feats for _, feats in complete])
    if set(y) != {0, 1}:
        raise ValueError('Need both positive and negative fitting evidence')
    center = x.mean(axis=0)
    spread = x.std(axis=0).clip(min=1e-7)
    x = np.column_stack([np.ones(len(x)), (x-center)/spread])
    weights = 0.5 / np.bincount(y)[y]
    penalty = np.diag([0.] + [0.10] * 11 + [0.7] * (x.shape[1]-12))
    beta = np.zeros(x.shape[1])
    for _ in range(20):
        p = 1 / (1 + np.exp(-np.clip(x @ beta, -30, 30)))
        grad = x.T @ (weights * (p-y)) + penalty @ beta
        curvature = x.T @ ((weights * p * (1-p))[:, None] * x) + penalty
        beta -= np.linalg.solve(curvature, grad)
    return {'beta': beta.tolist(), 'center': center.tolist(), 'spread': spread.tolist()}


def score(features, recipe):
    if features is None:
        return None
    return float(recipe['beta'][0] + ((features-recipe['center'])/recipe['spread']) @ recipe['beta'][1:])


def summarize(rows, parameters):
    output = Counter()
    for expected, features in rows:
        value = score(features, parameters)
        passed = value is not None and value > parameters['threshold']
        output[f'{int(expected)}:{int(passed)}'] += 1
    return dict(output)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'replay'))
    for key in ('config','base','cache','negatives','speaker_split','features','human_anchors'):
        parser.add_argument('--'+key.replace('_','-'), type=Path, required=True)
    for key in ('corpus','reference','capture','root','coerll_supplemental','ffmpeg','espeak'):
        parser.add_argument('--'+key.replace('_','-'), type=Path)
    parser.add_argument('--respect-pilot-disposition', action='store_true',
                        help='Evaluate fallback only for abstentions eligible under the pilot policy')
    args = parser.parse_args()
    if args.stage == 'freeze' and not args.espeak:
        parser.error('freeze requires --espeak')
    if args.stage == 'replay' and not (args.root and args.capture and args.reference):
        parser.error('replay requires --root, --capture and --reference')
    if bool(args.corpus) != bool(args.ffmpeg):
        parser.error('--corpus and --ffmpeg must be supplied together')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model, meta = frozen_model(args.base)
    cache_sha = file_digest(args.cache / 'index.json')
    negative_sha = file_digest(args.negatives / 'index.json')
    extractor_sha = file_digest(args.features)
    synthetic = json.loads((args.cache / 'index.json').read_text())
    if meta['teacher_cache_sha256'] != cache_sha or synthetic['frontend_sha256'] != extractor_sha:
        raise ValueError('Synthetic bank and feature tool seals do not match')
    provenance = dict(bank_sha256=meta['weights_sha256'], synthetic_cache_sha256=cache_sha,
                      negative_cache_sha256=negative_sha, features_sha256=extractor_sha,
                      split_sha256=file_digest(args.speaker_split), recipe_sha256=file_digest(Path(__file__)))
    if args.stage == 'freeze':
        vocab, speech = load_cache(args.cache)
        paths = templates(args.espeak, vocab, clock_context=True)[CLOCK]
        lookup = {phone: i for i, phone in enumerate(vocab)}
        negatives = [tuple(lookup.get(phone, len(vocab)-1)
                           for phone in phonemes(str(args.espeak), phrase + '.')) for phrase in NEGATIVE_PHRASES]
        paths = paths, negatives
        negatives = load_negatives(args.negatives, args.speaker_split, extractor_sha)
        splits = {key: row_specs(model, speech[key], negatives[key], paths) for key in ('train','dev')}
        anchor = []
        for intent, _, filename, digest in human_anchors(args.human_anchors):
            if intent == CLOCK:
                path = (args.human_anchors / filename).resolve(strict=True)
                if path.parent != args.human_anchors.resolve(strict=True):
                    raise ValueError('Public anchor escaped its declared directory')
                pcm = pcm_from(path, digest)
                features = clock_features(model, native_frames(args.features, pcm), paths)
                anchor.append({'pcm_sha256': sha256(trimmed(pcm)).hexdigest(), 'features': features})
        if len(anchor) != 1:
            raise ValueError('Expected one declared clock calibration anchor')
        # Only the fixed synthetic and CHM training split fit coefficients.
        fit = fit_model(splits['train'])
        negatives = [score(ft, fit) for target, ft in splits['train'] + splits['dev'] if not target and ft is not None]
        fit['threshold'] = max(negatives)
        config = {'schema':METHOD,'provenance':provenance,'paths':paths,'fit':fit,
                  'features':FEATURES, 'anchor_pcm_sha256':anchor[0]['pcm_sha256'],
                  'scored_negative_threshold_examples':len(negatives),
                  'train':summarize(splits['train'], fit),'dev_calibration':summarize(splits['dev'], fit),
                  'anchor_score':score(anchor[0]['features'], fit)}
        with args.config.open('x') as handle:
            json.dump(config, handle, indent=2)
        print(json.dumps({k:config[k] for k in ('train','dev_calibration','anchor_score','fit')},indent=2))
        return

    recipe = json.loads(args.config.read_text())
    if recipe['schema'] != METHOD or recipe['provenance'] != provenance:
        raise ValueError('Recipe is outside the frozen replay inputs')
    params, paths = recipe['fit'], tuple(recipe['paths'])
    hashes = natural_adjustment_hashes(args.negatives) | {recipe['anchor_pcm_sha256']}
    synthetic_hashes = {row['pcm_sha256'] for row in synthetic['records'].values()}
    def decision(selected, frontend, pcm):
        if args.respect_pilot_disposition:
            pilot = json.loads(invoke(args.reference, pcm, '--decision'))
            if not pilot['allows_fallback']:
                return 0
        feats = clock_features(selected, native_frames(frontend, pcm), paths)
        value = score(feats, params)
        return CLOCK if value is not None and value > params['threshold'] else 0
    selection = list(evaluate(args, model, synthetic_hashes, hashes, predict=decision))
    compact = {mode:Counter() for mode in ('raw','retained')}
    for row in selection:
        for mode in compact:
            result = row[mode]
            if not result['matches_adjustment_pcm']:
                compact[mode][f"{row['expected']}:{result['baseline']}:{result['combined']}"] += 1
    planned = {key: 0 for key in ('total','added','baseline')}
    speakers_tested = set()
    if args.corpus:
        import duckdb
        from phoneme_chm150_lab import (CORPUS_SHA256, SPEAKER_SPLIT_SHA256, decode,
                                        eligible_rows)
        if file_digest(args.corpus) != CORPUS_SHA256 or file_digest(args.speaker_split) != SPEAKER_SPLIT_SHA256:
            raise ValueError('CHM corpus or roles do not match')
        roles = json.loads(args.speaker_split.read_text())['speaker_roles']
        speakers = {who for who, role in roles.items() if role == 'heldout'}
        with duckdb.connect(config={'autoload_known_extensions':False,
                                    'autoinstall_known_extensions':False}) as connection:
            for _, who, _, blob in eligible_rows(connection, args.corpus, speakers):
                pcm = decode(args.ffmpeg, blob)
                if 0 < len(pcm) <= 128_000 and len(pcm) % 2 == 0:
                    if sha256(trimmed(pcm)).hexdigest() in hashes:
                        raise ValueError('Reserved CHM waveform overlaps tuning')
                    planned['total'] += 1
                    speakers_tested.add(who)
                    if int(invoke(args.reference, pcm)):
                        planned['baseline'] += 1
                    elif decision(model, args.features, pcm):
                        planned['added'] += 1
    summary = {
        'config_sha256': file_digest(args.config),
        'baseline_sha256': file_digest(args.reference),
        'capture_sha256': file_digest(args.capture),
        'respected_pilot_disposition': args.respect_pilot_disposition,
        'historical': compact,
        'excluded_fitting_overlap': [
            row['id'] for row in selection if any(row[mode]['matches_adjustment_pcm'] for mode in compact)
        ],
        'non_human_summary_included': True,
        'chm': dict(planned, speakers=len(speakers_tested)) if args.corpus else None,
    }
    focus = {'mty-003-02-15','mty-042-04-15','coerll-int02c-hora-esmx','public-03','public-37','public-81','public-88','public-89','veto-fact'}
    summary['episodes'] = [row for row in selection if row['id'] in focus]
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
