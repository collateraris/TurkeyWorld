"""Explicit saved-Adam recovery revision and held-out melee negatives."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from .rssm_spawn_curriculum import source_recovery_labels, file_sha


RECOVERY_PROTOCOL = 'rssm_one_enemy_combat_own_branches_balanced_spawn_v3'


def bind_recovery_revision(state, revision, *, sha256, path, base_plan_sha256,
                           data_sha256, negative_sha256, source_checkpoint_sha256):
    """Change only training metadata at an exact complete optimizer boundary."""
    from .rssm_combat_branches import normalize_combat_branch_protocol
    if (revision.get('version') != 3 or revision.get('base_plan_sha256') != base_plan_sha256
            or revision.get('data_manifest_sha256') != data_sha256
            or revision.get('negative_manifest_sha256') != negative_sha256
            or revision.get('source_checkpoint_sha256') != source_checkpoint_sha256
            or revision.get('numeric_state_feedback') != 'detached'
            or revision.get('source_future_feedback') is not False
            or revision.get('sampler') != {'batch':16,'spawn':4,'attack_negative':4,'ordinary':8}
            or revision.get('expanded_spawn_validation') != 'all_long_val_events'
            or revision.get('validation_every') != 250):
        raise ValueError('Recovery revision identity/protocol changed')
    branches = normalize_combat_branch_protocol(revision.get('combat_branch_protocol'))
    if branches is None or branches['offsets'] != [1,4,8,16]:
        raise ValueError('Recovery requires own-memory combat branches1/4/8/16')
    training = state['rssm_training']; feedback = training.get('feedback_revision')
    if not feedback or feedback.get('numeric_state_feedback') != 'detached':
        raise ValueError('Recovery must preserve the measured numeric-feedback fix')
    previous = training.get('combat_recovery_revision')
    if previous:
        if previous['sha256'] != sha256 or previous['last_optimizer_step'] != state['global_training_step']:
            raise ValueError('Recovery saved revision/counters changed')
    else:
        if revision['start_optimizer_step'] != state['global_training_step']:
            raise ValueError('Recovery needs its exact preserved Adam boundary')
        state.setdefault('objective_history', []).append({'loss_protocol':state['loss_protocol'],
            'last_step':state['global_training_step'],'reason':'user_requested_own_branches_balanced_spawn_melee_negatives',
            'last_validation_report':training.get('last_validation_report'),
            'last_validation_passed':training.get('last_validation_passed',False)})
        previous = {'sha256':sha256,'plan':str(path),'start_optimizer_step':revision['start_optimizer_step'],
            'last_optimizer_step':revision['start_optimizer_step'],'updates':0,'combat_updates':0,
            'sampler_state':None,'baseline_complete':False}
        training['combat_recovery_revision'] = previous
        training.update(quality_streak=0,last_validation_passed=False)
    state['loss_protocol'] = RECOVERY_PROTOCOL
    state['training_protocol'].update(combat_branch_protocol=branches,
        own_memory_combat_branches=True,combat_branch_active_environments=['combat'],
        auxiliary_combat_quotas=revision['sampler'],expanded_spawn_validation='all_long_val_events',
        close_wall_melee_negative_validation=True,source_future_feedback=False)
    return previous, branches


def evaluate_melee_negatives(model,data,output_dir,*,context=1,batch_size=16):
    """Every eligible recorded negative and every positive, kept as two cohorts."""
    import torch
    from .behavior_heads import _Metrics,_measure
    from .staged_validation import _isolated_validation
    from .training_pipeline import atomic_json
    if data.split not in ('val','test') or model.config.navigation_only:
        raise ValueError('Melee validation requires held-out one-enemy Source tapes')
    rows={'negative':[],'positive':[]}; source_ids=[]
    for episode,info in enumerate(data.episodes):
        with np.load(info['path'],allow_pickle=False) as saved:
            labels=source_recovery_labels(saved['states'],saved['attack_flags'],float(saved['dt']))
        source_ids.append({'episode':Path(info['path']).name,'sha256':file_sha(info['path'])})
        for name,label in [('negative','attack_negative'),('positive','attack_positive')]:
            rows[name].extend((episode,int(tick)-context+1) for tick in np.flatnonzero(labels[label]) if tick>=context-1)
    if not rows['negative'] or not rows['positive']:
        raise ValueError('Melee panel must contain both genuine eligible negatives and later attacks')
    device=next(model.parameters()).device; reports={}
    with _isolated_validation(model,data,173):
        for name,pairs in rows.items():
            meter=_Metrics()
            for first in range(0,len(pairs),batch_size):
                selected=np.asarray(pairs[first:first+batch_size],np.int64)
                batch={k:v.to(device) if torch.is_tensor(v) else v for k,v in
                    data.batch(len(selected),context=context,horizon=1,indices=selected).items()}
                features=model.encode_map(batch['map'])
                memory=model.initialize_from_image(batch['context_frames'],batch['states'][:,:context],features)
                result=model.transition(batch['states'][:,context-1],batch['actions'][:,0],features,memory)
                _measure(result,batch,context,meter,model.config.simulation_hz,model.config.endless,
                    game_training_v5=model.config.game_training_v5)
            reports[name]=meter.report()
    negative=reports['negative']['binary'].get('attack_accepted',{})
    positive=reports['positive']['binary'].get('attack_accepted',{})
    count=negative.get('fp',0)+negative.get('tn',0)
    rate=negative.get('fp',0)/count if count else None
    specification={'kind':'all_source_close_wall_melee_v1','context':context,'sources':source_ids,'rows':rows}
    report={'kind':specification['kind'],'source_held_out':True,'source_episode_count':len(source_ids),
        'negative_transitions':len(rows['negative']),'positive_transitions':len(rows['positive']),
        'negative_false_positive_rate':rate,'positive_recall':positive.get('recall'),
        'cohorts':reports,'future_rgb_inputs':0,'runtime_engine_calls':0,'source_future_feedback':False,
        'panel_fingerprint':hashlib.sha256(json.dumps(specification,sort_keys=True).encode()).hexdigest(),
        'cohort_scope':'All eligible Source melee negatives and all later positives; not a population estimate'}
    path=Path(output_dir);path.mkdir(parents=True,exist_ok=True)
    atomic_json(path/'melee_negative_validation.json',report);return report


def recovery_quality_gate(base,spawn,melee):
    """Retain every previous navigation/trajectory guard; expand Source checks."""
    result=copy.deepcopy(base);failures=result['failure_reasons'];summary=result['summary']
    # Old nine-event short-panel results stay in the report. Its spawn gate is
    # replaced by the exhaustive long-VAL gate rather than counted twice.
    failures[:]=[name for name in failures if name not in
        ('spawn_decoded_top1_tile_accuracy','spawn_decoded_xy_error_tiles')]
    overall=spawn['overall']['metrics']
    for name,is_min,bound in [('spawn_decoded_top1_tile_accuracy',True,.8),('spawn_decoded_xy_error_tiles',False,.5)]:
        item=overall.get(name,{});value=item.get('mean');summary['expanded_'+name]=value
        if item.get('denominator',0)<128 or value is None or (value<bound if is_min else value>bound):
            failures.append('expanded_'+name)
    if spawn['missing_anchor_classes'] or spawn['source_episode_count']<64:
        failures.append('expanded_spawn_anchor_coverage')
    summary['melee_negative_false_positive_rate']=melee['negative_false_positive_rate']
    summary['melee_positive_recall']=melee['positive_recall']
    if melee['negative_transitions']<64 or melee['negative_false_positive_rate'] is None or melee['negative_false_positive_rate']>.1:
        failures.append('melee_negative_false_positive_rate')
    if melee['positive_transitions']<8 or melee['positive_recall'] is None or melee['positive_recall']<.8:
        failures.append('melee_positive_recall')
    result['passed']=not failures
    result['thresholds'].update(expanded_spawn_events_min=128,expanded_spawn_source_episodes_min=64,
        expanded_spawn_all_eight_anchors_required=True,melee_negative_min=64,
        melee_false_positive_rate_max=.1,melee_positive_recall_min=.8)
    return result
