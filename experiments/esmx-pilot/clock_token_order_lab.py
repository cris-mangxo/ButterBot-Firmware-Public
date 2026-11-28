"""Investigate whole-turn evidence for Hora without overriding pilot vetoes."""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
from pathlib import Path
import itertools
import json

import numpy as np
import torch

from capture_replay import pcm_from,recordings
from joke_contrast_lab import cheapest, fit, score, passed, matrix
from keyword_lab import LABELS,trimmed
from phoneme_confirm_lab import human_anchors, model_output, natural_adjustment_hashes
from phoneme_reject_lab import load_negatives
from phoneme_replay_lab import frozen_model,native_frames,overlaps_adjustment
from phoneme_student_lab import load_cache
from phoneme_teacher_lab import phonemes
from time_lab import file_digest,invoke

HOUR = 5
PHRASES = {
    "word": ("hora", "horas"),
    "positive": ("qué hora es", "qué horas son", "dime la hora", "dime qué hora es",
                 "la hora actual", "me puedes decir la hora", "me das la hora", "cuál es la hora"),
    "context": ("a qué hora es", "a qué hora llega", "qué hora es la cita", "qué hora es en",
                "dime a qué hora", "la hora de la cita", "te digo la hora", "nos vemos a la misma hora",
                "qué hora era"),
    "denial": ("no me digas", "no me des", "no quiero", "no necesito", "mejor no", "cancela", "olvida"),
    "dice": ("dado de seis", "tira un dado", "un dado", "no tires", "no lances"),
}
ARMS = {"lexical": (0,1,2,3,4), "context": (0,1,2,3,4,5,6,7), "dice_contrast": tuple(range(9))}


def templates(executable,vocabulary):
    labels={phone:index for index,phone in enumerate(vocabulary)}
    return {key:[tuple(labels.get(phoneme,len(vocabulary)-1) for phoneme in phonemes(str(executable),text+'.'))
                 for text in values] for key,values in PHRASES.items()}


@torch.inference_mode()
def features(model,frames,paths):
    if frames is None or len(frames)<16:
        return None
    phones,intents,negation=model_output(model,frames)
    if negation.item()>=0:
        return None
    scores=phones[0].numpy()
    ranks=intents[0].numpy()
    index=LABELS.index(HOUR)
    margin=ranks[index]-max(np.delete(ranks,index))
    noun,_=cheapest(scores,paths['word'])
    positive,(start,end)=cheapest(scores,paths['positive'])
    context,_=cheapest(scores,paths['context'])
    denial,_=cheapest(scores[:start],paths['denial'])
    dice,_=cheapest(scores,paths['dice'])
    return np.asarray([margin,-negation.item(),noun,positive,context-positive,
                       (end-start)/len(scores),(len(scores)-end)/len(scores),
                       denial,dice-positive],dtype=np.float64)


def collection(model,paths,way,records,negative):
    rows=[(row['intent']==HOUR and not row['denied'],features(model,m,paths)) for row,m,*_ in records[way]]
    rows.extend((False,features(model,m,paths)) for m in negative[way])
    return rows


def policy(base,allow_conflicts):
    if base['allows_fallback']:
        return True
    return allow_conflicts and base['intent']==0 and base['disposition']=='dice_negation_conflict'


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('freeze','replay','conversation'))
    for key in ('config','checkpoint','cache','negatives','speakers','frontend','calibration','anchors'):
        parser.add_argument('--'+key,type=Path,required=True)
    for key in ('espeak','captured','pilot','root','extra-reserve','corpus','ffmpeg'):
        parser.add_argument('--'+key,type=Path)
    args = parser.parse_args()
    if args.stage == 'freeze' and not args.espeak:
        parser.error('freeze requires --espeak')
    if args.stage == 'replay' and not all((args.captured, args.pilot, args.root)):
        parser.error('replay requires --captured, --pilot and --root')
    if args.stage == 'conversation' and not all((args.pilot, args.corpus, args.ffmpeg)):
        parser.error('conversation requires --pilot, --corpus and --ffmpeg')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model,report=frozen_model(args.checkpoint)
    synthetic=json.loads((args.cache/'index.json').read_text())
    if report['teacher_cache_sha256']!=file_digest(args.cache/'index.json') or synthetic['frontend_sha256']!=file_digest(args.frontend):
        raise ValueError('Clock experiment input differs from phoneme cache')
    seals={'code':file_digest(Path(__file__)), 'weights':report['weights_sha256'],
           'cache':file_digest(args.cache/'index.json'),'calibration':file_digest(args.calibration),
           'negative_cache':file_digest(args.negatives/'index.json'),'speakers':file_digest(args.speakers),
           'frontend':synthetic['frontend_sha256']}
    calibration=json.loads(args.calibration.read_text())
    if calibration['role'] != 'local_calibration_only':
        raise ValueError('Unrecognized donor role')
    files = []
    donor_root = args.calibration.resolve(strict=True).parent
    for row in calibration['rows']:
        recording = (donor_root / row['wav']).resolve(strict=True)
        if recording.parent != donor_root:
            raise ValueError('Donor WAV escaped its calibration directory')
        pcm = pcm_from(recording, row['sha256'])
        if not 0 < len(pcm) <= 128_000:
            raise ValueError('Clock control must stay within the complete-turn limit')
        files.append((row, pcm))
    if args.stage=='freeze':
        vocab,records=load_cache(args.cache)
        variants=templates(args.espeak,vocab)
        negatives=load_negatives(args.negatives,args.speakers,synthetic['frontend_sha256'])
        training=collection(model,variants,'train',records,negatives)
        dev=collection(model,variants,'dev',records,negatives)
        anchor,=[pcm_from(args.anchors/filename,digest) for intent,_,filename,digest in human_anchors(args.anchors) if intent==HOUR]
        anchor_value=features(model,native_frames(args.frontend,anchor),variants)
        explicit=[(False,features(model,native_frames(args.frontend,pcm),variants)) for _,pcm in files]
        fitted={}
        for transfer in (False,True):
            fit_rows=training.copy()
            if transfer:
                if anchor_value is None:
                    raise ValueError('Clock adjustment anchor has no admissible features')
                fit_rows += [(True,anchor_value)]*8
            for name,columns in ARMS.items():
                rule=fit(fit_rows,columns)
                rule['threshold']=max(score(value,rule) for truth,value in itertools.chain(training,dev,explicit)
                                      if not truth and value is not None)
                arm=f'{name}-'+('anchor' if transfer else 'synthetic')
                fitted[arm]={'rule':rule,'train':matrix(training,rule),'dev':matrix(dev,rule),
                             'donor_negatives':matrix(explicit,rule),'anchor_check':passed(anchor_value,rule)}
        # Resolve ties to the arm with fewer features and without human transfer.
        selected=max(fitted,key=lambda arm:fitted[arm]['dev'].get('1:1',0))
        with args.config.open('x') as out:
            json.dump({'selected':selected,'seals':seals,'paths':variants,'arms':fitted,
                      'reference_anchor_pcm_sha256':sha256(trimmed(anchor)).hexdigest()},out,indent=2)
        print(json.dumps({'selected':selected,'arms':{k:{x:y for x,y in v.items() if x!='rule'} for k,v in fitted.items()}},indent=2))
        return
    spec=json.loads(args.config.read_text())
    if spec['seals']!=seals:
        raise ValueError('Clock replay is outside the frozen recipe')
    model_params=spec['arms'][spec['selected']]['rule']
    paths=spec['paths']
    occupied=natural_adjustment_hashes(args.negatives)|{spec['reference_anchor_pcm_sha256']}
    occupied.update(sha256(trimmed(pcm)).hexdigest() for _,pcm in files)
    adjusted={row['pcm_sha256'] for row in synthetic['records'].values()}
    def observe(pcm):
        base=json.loads(invoke(args.pilot,pcm,'--decision'))
        v=features(model,native_frames(args.frontend,pcm),paths)
        accepted=passed(v,model_params)
        prior = int(invoke(args.pilot, pcm))
        result = {'base':base,'prior':prior,'accepted':accepted,'score':score(v,model_params)}
        for name,permit in [('strict',False),('conflict_lab',True)]:
            result[name] = prior or (HOUR if accepted and policy(base, permit) else 0)
        return result

    if args.stage=='conversation':
        import duckdb
        from phoneme_chm150_lab import CORPUS_SHA256,SPEAKER_SPLIT_SHA256,decode,eligible_rows
        if file_digest(args.corpus) != CORPUS_SHA256 or file_digest(args.speakers) != SPEAKER_SPLIT_SHA256:
            raise ValueError('Unexpected CHM corpus or speaker partition')
        role=json.loads(args.speakers.read_text())['speaker_roles']
        reserved={s for s,r in role.items() if r=='heldout'}
        observed=Counter()
        speakers=set()
        with duckdb.connect(config={'autoload_known_extensions':False,'autoinstall_known_extensions':False}) as conn:
            for _,who,neg,wav in eligible_rows(conn,args.corpus,reserved):
                pcm=decode(args.ffmpeg,wav)
                if not 0<len(pcm)<=128_000 or len(pcm)%2:
                    continue
                if overlaps_adjustment(pcm, adjusted, occupied):
                    raise ValueError('CHM replay overlaps fitting audio')
                speakers.add(who)
                got=observe(pcm)
                observed['items']+=1
                observed['prior_activated']+=bool(got['prior'])
                observed['isolated_activated']+=got['accepted']
                for key in ('strict','conflict_lab'):
                    observed[key+'_added']+=got[key]==HOUR and got['prior']==0
        print(json.dumps(dict(speakers=len(speakers),recipe_sha=file_digest(args.config),
                              pilot_sha=file_digest(args.pilot),**observed),indent=2))
        return

    panel=list(recordings(args.root))
    if args.extra_reserve:
        extra=json.loads(args.extra_reserve.read_text())
        if extra['speaker_group'] in {row['speaker_group'] for row in calibration['rows']}:
            raise ValueError('Held-out clock speaker overlaps calibration')
        panel.append((extra['id'],'public_reserved_time','time',HOUR,args.extra_reserve.parent/extra['sealed_wav'],extra['wav_sha256']))
    counts={key:Counter() for key in ('raw','retained')}
    excluded=[]
    key_cases=['mty-003-02-15','mty-042-04-15','2425','public-03','public-37','public-81','public-88','public-89','veto-fact','veto-d6']
    details={}
    for name,split,_,target,filename,digest in panel:
        pcm=pcm_from(filename,digest)
        arms={'raw':pcm,'retained':invoke(args.captured,pcm,'--pcm-with-hangover')}
        if any(overlaps_adjustment(s,adjusted,occupied) for s in arms.values()):
            if split.startswith('public'):
                raise ValueError('Public replay overlaps adjustment')
            excluded.append(name)
            continue
        for key,clip in arms.items():
            data=observe(clip)
            counts[key][f"{target}:{data['prior']}:{data['strict']}:{data['conflict_lab']}"]+=1
            if name in key_cases:
                details.setdefault(name,{'target':target})[key]=data
    print(json.dumps({'recipe_sha':file_digest(args.config),'historical':counts,'excluded':excluded,
                      'examples':details,'pilot_sha':file_digest(args.pilot),
                      'capture_sha':file_digest(args.captured)},indent=2))


if __name__=='__main__':
    main()
