"""Offline word-order experiment on a fixed compact Joke recognizer."""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json

import numpy as np
import torch

from capture_replay import pcm_from, recordings
from keyword_lab import LABELS, trimmed
from phoneme_confirm_lab import local_ctc_alignment, model_output, natural_adjustment_hashes
from phoneme_reject_lab import load_negatives
from phoneme_replay_lab import frozen_model, native_frames, overlaps_adjustment
from phoneme_student_lab import load_cache
from phoneme_teacher_lab import phonemes
from time_lab import file_digest, invoke, supplemental

JOKE = 3
WORD = ('chiste', 'chistes')
REQUEST = ('cuéntame', 'cuenta', 'échame', 'échate', 'aviéntate', 'contar', 'contarme',
           'dime', 'di', 'decirme', 'escuchar', 'quiero oír', 'comparte')
DENIAL = ('no me cuentes', 'no cuentes', 'no me digas', 'no quiero', 'no me interesa',
          'mejor no', 'no, cancela', 'olvida')
POSITIVE = ('cuéntame un chiste', 'dime un chiste', 'cuéntame un buen chiste', 'un chiste por favor',
            'échate un chiste', 'puedes contarme un chiste', 'un chiste divertido')
CONTROL = ('me contó un chiste', 'no me cuentes un chiste', 'un chiste de ayer',
           'ya escuché ese chiste', 'el chiste es que', 'qué es un chiste')
ARMS = (('isolated', (0, 1, 2)), ('order', (0, 1, 2, 3, 4, 5, 6)), ('contrast', tuple(range(9))))


def variant_paths(espeak, vocab):
    idx = {item: n for n, item in enumerate(vocab)}
    return {name: [tuple(idx.get(phone, len(vocab) - 1)
                         for phone in phonemes(str(espeak), phrase + '.')) for phrase in phrases]
            for name, phrases in [('word', WORD), ('request', REQUEST), ('denial', DENIAL),
                                  ('positive', POSITIVE), ('negative', CONTROL)]}


def cheapest(scores, paths):
    candidate = (local_ctc_alignment(scores, path, track_span=True) for path in paths)
    return min(((cost, interval) for cost, interval in candidate if interval is not None),
               default=(8.0, (0, 0)))


@torch.inference_mode()
def features(model, frames, paths):
    if frames is None or len(frames) < 16:
        return None
    phonetic, intent, neg = model_output(model, frames)
    negation = neg.item()
    if negation >= 0:
        return None
    scores = phonetic[0].numpy()
    ranks = intent[0].numpy()
    joke_index = LABELS.index(JOKE)
    margin = ranks[joke_index] - max(np.delete(ranks, joke_index))
    noun, (start, end) = cheapest(scores, paths['word'])
    prefix = scores[:start]
    verb, (_, verb_end) = cheapest(prefix, paths['request'])
    denial, _ = cheapest(prefix, paths['denial'])
    positive, _ = cheapest(scores, paths['positive'])
    negative, _ = cheapest(scores, paths['negative'])
    return np.array([margin, -negation, noun, verb, denial - verb,
                     (start-verb_end)/len(scores), (len(scores)-end)/len(scores),
                     positive, negative-positive], dtype=np.float64)


def fit(rows, columns):
    known = [(truth, value[list(columns)]) for truth, value in rows if value is not None]
    x = np.stack([value for _, value in known])
    y = np.asarray([int(truth) for truth, _ in known])
    if len(set(y)) != 2:
        raise ValueError("Need positive and negative fitting rows")
    mu = x.mean(axis=0)
    sigma = np.maximum(x.std(axis=0), 1e-7)
    x = np.column_stack([np.ones(len(x)), (x-mu)/sigma])
    weight = 0.5/np.bincount(y)[y]
    penalty = np.diag([0.] + [.12] * len(columns))
    coeff = np.zeros(x.shape[1])
    for _ in range(20):
        prob = 1 / (1 + np.exp(-np.clip(x @ coeff, -30, 30)))
        gradient = x.T @ (weight*(prob-y)) + penalty @ coeff
        curvature = x.T @ ((weight*prob*(1-prob))[:, None]*x) + penalty
        coeff -= np.linalg.solve(curvature, gradient)
    raw = np.zeros(9)
    raw[list(columns)] = coeff[1:]/sigma
    return {'coefficient':raw.tolist(), 'intercept':float(coeff[0] - (mu/sigma) @ coeff[1:])}


def score(value, rule):
    if value is None:
        return None
    return float(value @ rule['coefficient'] + rule['intercept'])


def passed(value, rule):
    measured = score(value, rule)
    return measured is not None and measured > rule['threshold']


def matrix(rows, rule):
    return dict(Counter(f'{int(expected)}:{int(passed(value,rule))}' for expected,value in rows))


def borrowed_rows(config):
    source=json.loads(config.read_text())
    if source['role'] != 'local_calibration_only':
        raise ValueError('Expected a local donor manifest')
    seen = set()
    root = config.resolve(strict=True).parent
    result=[]
    for row in source['rows']:
        file = (root / row['wav']).resolve(strict=True)
        if file.parent != root or row['speaker_group'] in seen:
            raise ValueError('Unpinned or repeated donor')
        seen.add(row['speaker_group'])
        pcm = pcm_from(file, row['sha256'])
        if 0 < len(pcm) <= 128_000:
            result.append((row,pcm))
    return result


def conversation(args, model, paths, rule):
    import duckdb
    from phoneme_chm150_lab import CORPUS_SHA256, SPEAKER_SPLIT_SHA256, decode, eligible_rows

    if (file_digest(args.corpus) != CORPUS_SHA256
            or file_digest(args.speakers) != SPEAKER_SPLIT_SHA256):
        raise ValueError('Conversation corpus or speaker split provenance changed')
    roles = json.loads(args.speakers.read_text())['speaker_roles']
    heldout = {who for who, role in roles.items() if role == 'heldout'}
    fitting = natural_adjustment_hashes(args.negatives)
    measured = dict(evaluated=0, base_nonzero=0, eligible_under_pilot_policy=0, added=0, independent_accept=0)
    tested = set()
    config = {'autoload_known_extensions': False, 'autoinstall_known_extensions': False}
    with duckdb.connect(config=config) as connection:
        for _, who, _, wav in eligible_rows(connection, args.corpus, heldout):
            pcm = decode(args.ffmpeg, wav)
            if not 0 < len(pcm) <= 128_000 or len(pcm) % 2:
                continue
            if sha256(trimmed(pcm)).hexdigest() in fitting:
                raise ValueError('Conversation intersects the tuning audio')
            tested.add(who)
            measured['evaluated'] += 1
            base = json.loads(invoke(args.pilot, pcm, '--decision'))
            measured['base_nonzero'] += bool(base['intent'])
            measured['eligible_under_pilot_policy'] += base['allows_fallback']
            accepted = passed(features(model, native_frames(args.frontend, pcm), paths), rule)
            measured['independent_accept'] += accepted
            measured['added'] += bool(accepted and base['allows_fallback'])
    return dict(config_sha=file_digest(args.config), speakers=len(tested), **measured)


def main():
    arg = ArgumentParser(description=__doc__)
    arg.add_argument('stage', choices=('freeze', 'replay', 'conversation'))
    for key in ('config','checkpoint','cache','negatives','speakers','frontend','borrowed'):
        arg.add_argument('--'+key,type=Path,required=True)
    for key in ('espeak','capture','pilot','root','reserve','corpus','ffmpeg','coerll'):
        arg.add_argument('--'+key,type=Path)
    arg.add_argument('--transfer-positive', action='store_true',
                     help='Fit with balanced positive weight from two independent donors')
    args = arg.parse_args()
    if args.stage == 'freeze' and args.espeak is None:
        arg.error('freeze requires --espeak')
    if args.stage == 'replay' and not all((args.pilot,args.root,args.capture,args.reserve,args.coerll)):
        arg.error('replay requires --pilot, --root, --capture, --reserve and --coerll')
    if args.stage == 'conversation' and not all((args.pilot,args.corpus,args.ffmpeg)):
        arg.error('conversation requires --pilot, --corpus and --ffmpeg')
    if args.stage != 'freeze' and args.transfer_positive:
        arg.error('--transfer-positive applies only while freezing the recipe')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model,meta=frozen_model(args.checkpoint)
    syn=json.loads((args.cache/'index.json').read_text())
    if (meta['teacher_cache_sha256'] != file_digest(args.cache / 'index.json')
            or syn['frontend_sha256'] != file_digest(args.frontend)):
        raise ValueError("Model, source cache or frontend differs from the fit seal")
    seal={key:file_digest(path) for key,path in [('source',Path(__file__)),('cache',args.cache/'index.json'),
          ('negatives',args.negatives/'index.json'),('speakers',args.speakers),('borrowed',args.borrowed)]}
    seal['weights']=meta['weights_sha256']
    if args.stage=='freeze':
        vocab,speech=load_cache(args.cache)
        paths=variant_paths(args.espeak,vocab)
        other=load_negatives(args.negatives,args.speakers,syn['frontend_sha256'])
        clips=borrowed_rows(args.borrowed)
        borrowed=[(row['compatible_as_joke'],features(model,native_frames(args.frontend,pcm),paths)) for row,pcm in clips]
        groups={}
        for split in ('train','dev'):
            groups[split]=[(row['intent']==JOKE and not row['denied'], features(model,frames,paths))
                           for row,frames,*_ in speech[split]]
            groups[split].extend((False,features(model,entry,paths)) for entry in other[split])
        fitting = groups['train']
        if args.transfer_positive:
            human_positive = [(label, value) for label, value in borrowed if label and value is not None]
            if len(human_positive) != 2:
                raise ValueError('Expected two full, independent donor positives')
            positives = sum(label for label, _ in fitting)
            fitting = fitting + human_positive * (positives // len(human_positive))
        result={}
        for name,columns in ARMS:
            rule=fit(fitting,columns)
            rule['threshold']=max(score(value,rule) for truth,value in groups['train']+groups['dev']+borrowed
                                  if not truth and value is not None)
            result[name]={'rule':rule,'train':matrix(groups['train'],rule),
                          'dev':matrix(groups['dev'],rule),'extra_human_calibration':matrix(borrowed,rule)}
        winner=max(result,key=lambda k:(result[k]['dev'].get('1:1',0),
                   0 if args.transfer_positive else result[k]['extra_human_calibration'].get('1:1',0)))
        report={'seals':seal,'paths':paths,'selected':winner,'development':result,
                'transfer_positive_fitting':args.transfer_positive,
                'calibration_ids':[row['id'] for row,_ in clips]}
        with args.config.open('x') as saved:
            json.dump(report,saved,indent=2)
        print(json.dumps({k:{g:v for g,v in a.items() if g!='rule'} for k,a in result.items()},indent=2))
        return

    spec=json.loads(args.config.read_text())
    if spec['seals'] != seal:
        raise ValueError("The frozen contrast recipe does not match its inputs")
    paths=spec['paths']
    rule=spec['development'][spec['selected']]['rule']
    if args.stage == 'conversation':
        print(json.dumps(conversation(args, model, paths, rule), indent=2))
        return
    reserves=json.loads(args.reserve.read_text())['rows']
    synthetic={row['pcm_sha256'] for row in syn['records'].values()}
    negatives=natural_adjustment_hashes(args.negatives)
    calibration=json.loads(args.borrowed.read_text())['rows']
    speaker_cal={row['speaker_group'] for row in calibration}
    cal_pcm={sha256(trimmed(pcm_from(args.borrowed.parent/row['wav'],row['sha256']))).hexdigest()
             for row in calibration}
    if speaker_cal.intersection(row['speaker_group'] for row in reserves):
        raise ValueError("The donor and reserved speaker groups overlap")
    def observed(pcm):
        base=json.loads(invoke(args.pilot,pcm,'--decision'))
        v=features(model,native_frames(args.frontend,pcm),paths)
        accept=passed(v,rule)
        combined=base['intent'] or (JOKE if accept and base['allows_fallback'] else 0)
        return {'base':base,'score':score(v,rule),'independent_accept':accept,'combined':combined}
    panel={mode:Counter() for mode in ('raw','retained')}
    exclusion=[]
    for key,_,_,intent,path,digest in list(recordings(args.root))+[supplemental(args.root,args.coerll)]:
        pcm=pcm_from(path,digest)
        batch={'raw':pcm,'retained':invoke(args.capture,pcm,'--pcm-with-hangover')}
        if any(overlaps_adjustment(s,synthetic,negatives|cal_pcm) for s in batch.values()):
            exclusion.append(key)
            continue
        for mode,s in batch.items():
            outcome=observed(s)
            panel[mode][f"{intent}:{outcome['base']['intent']}:{outcome['combined']}"]+=1
    report={'config_sha':file_digest(args.config),'selected':spec['selected'],'historical':panel,
            'overlap_exclusions':exclusion,'reserve':[]}
    for row in reserves:
        pcm=pcm_from(args.reserve.parent/row['sealed_wav'],row['sealed_wav_sha256'])
        clip=invoke(args.capture,pcm,'--pcm-with-hangover')
        if any(overlaps_adjustment(s, synthetic, negatives | cal_pcm) for s in (pcm, clip)):
            raise ValueError("A reserved waveform overlaps fitting or calibration")
        report['reserve'].append({'id':row['id'],'raw':observed(pcm),'retained':observed(clip)})
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    main()
