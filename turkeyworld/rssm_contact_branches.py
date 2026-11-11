"""Training-only own-RSSM contact branches with offline Source labels.

A branch starts at a detached *actual* prior state and packed hidden memory.
Source receives a private copy for labels; it never corrects the neural branch.
"""
from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
import math

import numpy as np
import torch
from torch.nn import functional as F

from .contact_objectives import (normalize_contact_protocol, CONTACT_PROTOCOL_V2,
    prepare_contact_targets, differentiable_contact_terms, swept_player_wall_loss)
from .engine import Game, STATE_DIM
from .gradient_safety import finite_gradient_norm
from .motion_recovery_training import pass_fraction_targets
from .runtime_checks import circle_is_clear, segment_is_clear


CONTACT_BRANCH_DEFAULTS = {
    'version': 1, 'max_branches': 4, 'horizon': 16, 'max_source_probes': 64,
    'weight': .25, 'xy_weight': 1., 'wall_weight': 1., 'anchor_bce_weight': 1.,
    'anchor_margin_weight': 1., 'physical_error_fraction': .2,
    'canonical_atol': 2e-7, 'offsets': [1, 4, 8, 16]}
AXIS_COHORTS = ('both_free', 'x_free_y_blocked', 'x_blocked_y_free', 'both_blocked')


def normalize_contact_branch_protocol(value):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value)-set(CONTACT_BRANCH_DEFAULTS):
        raise ValueError('Unknown own-contact-branch protocol settings')
    result = {**CONTACT_BRANCH_DEFAULTS, **value}
    result['offsets'] = list(result['offsets'])
    for key, maximum in (('max_branches', 4), ('horizon', 32), ('max_source_probes', 64)):
        item = result[key]
        if not isinstance(item, int) or isinstance(item, bool) or not 1 <= item <= maximum:
            raise ValueError('Invalid bounded own-contact-branch setting: '+key)
    if result['version'] != 1 or isinstance(result['version'], bool):
        raise ValueError('Own-contact branches require protocol version1')
    for key in ('weight','xy_weight','wall_weight','anchor_bce_weight','anchor_margin_weight',
                'physical_error_fraction','canonical_atol'):
        item = result[key]
        if not isinstance(item, (int,float)) or isinstance(item,bool) or not math.isfinite(item) or item < 0:
            raise ValueError('Branch settings must be finite and nonnegative: '+key)
    if (not 0 < result['physical_error_fraction'] < 1 or not 0 < result['canonical_atol'] <= 2e-7
            or result['max_branches'] > result['max_source_probes']):
        raise ValueError('Invalid physical/canonical tolerance or Source probe budget')
    if (not result['offsets'] or any(not isinstance(x,int) or isinstance(x,bool) or x < 1 or x > 32
            for x in result['offsets']) or result['offsets'] != sorted(set(result['offsets']))):
        raise ValueError('Branch offsets must be sorted unique positive integers up to32')
    result['offsets'] = [x for x in result['offsets'] if x <= result['horizon']]
    if not result['offsets']:
        raise ValueError('At least one reporting offset must fit the branch horizon')
    return result


def capture_contact_candidate(step, state, memory, actions, grid, prediction):
    """Copy actual own-prefix values, never references to a retained graph."""
    if (state.ndim != 2 or state.shape[-1] != STATE_DIM or memory.ndim != 2
            or memory.shape[0] != len(state) or actions.ndim != 3
            or actions.shape[0] != len(state) or actions.shape[-1] != 4
            or grid.shape != (len(state),16,16) or prediction.ndim != 3
            or prediction.shape[0] != len(state) or prediction.shape[-1] != STATE_DIM):
        raise ValueError('Own contact candidates require actual[B,state/memory/actions/map/prediction]')
    return {'step': int(step), **{name: value.detach().clone() for name,value in (
        ('state',state),('memory',memory),('actions',actions),('map',grid),('prediction',prediction))},
        'memory_provenance': 'actualownprefix', 'new_observed_rgb_frames': 0}


def _flatten(candidates):
    rows, first_wall = [], {}
    for item in sorted(candidates, key=lambda value:value['step']):
        if item.get('memory_provenance') != 'actualownprefix' or item.get('new_observed_rgb_frames') != 0:
            raise ValueError('Branches require actual own-prefix memory and zero new observed RGB')
        states = item['state'].detach().float().cpu().numpy()
        prediction = item['prediction'].detach().float().cpu().numpy()
        maps = item['map'].detach().cpu().numpy()
        for actor in range(len(states)):
            old = states[actor,:2]*16
            for offset, predicted in enumerate(prediction[actor]):
                point = predicted[:2]*16
                if (not circle_is_clear(maps[actor],point,.2)
                        or not segment_is_clear(maps[actor],old,point,.2)):
                    first_wall.setdefault(actor, item['step']+offset+1)
                    break
                old = point
            rows.append({'item':item,'actor':actor,'step':item['step'],
                'state_numpy':states[actor].copy(),'map_numpy':maps[actor].copy(),
                'prediction_numpy':prediction[actor].copy()})
    for row in rows:
        first = first_wall.get(row['actor'])
        row['first_wall_step'] = first
        row['first_wall_chunk'] = first is not None and row['step'] < first <= row['step']+len(row['prediction_numpy'])
    # Every actor first supplies the exact pre-wall chunk, then early prefixes.
    priority = sorted(rows,key=lambda row:(not row['first_wall_chunk'],row['step'],row['actor']))
    return [row for row in priority if row['first_wall_step'] is None or row['step'] < row['first_wall_step']], first_wall


def _source_labels(row, protocol, dt):
    """Private Source replay. No labels are returned to main or branch inputs."""
    state, grid = row['state_numpy'],row['map_numpy']
    if (not np.isfinite(state).all() or state[4] <= 0
            or not circle_is_clear(grid,state[:2]*16,.2)):
        raise ValueError('invalid_anchor_geometry')
    game = Game(seed=0,endless=True,enemies_enabled=False)
    game.map = np.asarray(grid,dtype=np.uint8).copy()
    game.map_height,game.map_width = game.map.shape
    # Strict training adapter rejects invalid geometry; do not project/repair.
    game.set_state_vector(state)
    canonical = game.state_vector()
    if not np.allclose(canonical,state,rtol=0,atol=protocol['canonical_atol']):
        raise ValueError('noncanonical_anchor_state')
    actions = row['item']['actions'][row['actor'],:protocol['horizon']].detach().float().cpu().numpy()
    labels = [canonical.copy()]
    for action in actions:
        game.step(action,dt=dt)
        labels.append(game.state_vector().copy())
    labels = np.stack(labels)
    old,target = torch.from_numpy(labels[:-1]),torch.from_numpy(labels[1:])
    fraction,active,command = pass_fraction_targets(old,target,torch.from_numpy(actions),dt=dt)
    fraction,active,command = (x.numpy() for x in (fraction,active,command))
    # A complete geometry scan finds the first wall across the full main path.
    # The Source-dependent physical error is earliest only among bounded probes.
    predicted = row['prediction_numpy'][:len(actions)]
    own_previous = np.concatenate([state[None],predicted[:-1]])
    compare = min(len(predicted),len(actions))
    free,blocked = active & (fraction >= .95),active & (fraction <= .05)
    actual = predicted[:compare,:2]-own_previous[:compare,:2]
    threshold = protocol['physical_error_fraction']*np.abs(command[:compare])
    # Match the actual validation definitions. Ordinary .7 progress on a free
    # axis is a velocity residual, not a false stop/contact failure.
    false_stop = free[:compare] & (np.abs(actual) < threshold)
    missed_block = blocked[:compare] & (np.abs(actual) > threshold)
    contact_bad = np.flatnonzero(missed_block.any(-1))
    free_bad = np.flatnonzero(false_stop.any(-1))
    bad = np.flatnonzero((false_stop | missed_block).any(-1))
    label_offset = int(contact_bad[0]) if len(contact_bad) else int(free_bad[0]) if len(free_bad) else 0
    velocity_residual = np.abs(actual-(labels[1:compare+1,:2]-labels[:compare,:2])) > threshold
    # Prefer the failed transition's physical outcome; otherwise retain a real
    # contact later in this short tape, then its actual first-step outcome.
    if not len(contact_bad):
        contacts = np.flatnonzero(blocked.any(-1))
        if len(contacts): label_offset = int(contacts[0])
    f,b,a = free[label_offset],blocked[label_offset],active[label_offset]
    if a.all():
        cohort = ('both_free' if f.all() else 'both_blocked' if b.all() else
                  'x_free_y_blocked' if f[0] and b[1] else
                  'x_blocked_y_free' if b[0] and f[1] else 'partial')
    else:
        cohort = 'single_axis' if a.any() else 'idle'
    return {**row,'labels':labels,'cohort':cohort,'cohort_label_offset':label_offset+1,
        'first_physical_error':row['step']+int(bad[0])+1 if len(bad) else None,
        'first_false_stop':row['step']+int(free_bad[0])+1 if len(free_bad) else None,
        'first_missed_block':row['step']+int(contact_bad[0])+1 if len(contact_bad) else None,
        'free_velocity_residual_axes':int((velocity_residual & free[:compare] & ~false_stop).sum()),
        'partial_velocity_residual_axes':int((velocity_residual & active[:compare]
            & ~free[:compare] & ~blocked[:compare]).sum()),
        'canonical_max_abs_difference':float(np.max(np.abs(canonical-state)))}


def select_contact_branches(candidates, *, contact_branch_protocol, dt=1/30):
    """Bounded Source probes, first-wall/physical-error priorities and coverage."""
    protocol = normalize_contact_branch_protocol(contact_branch_protocol)
    rows, first_wall = _flatten(candidates)
    skips, probed, probes = Counter(),[],0
    full = [row for row in rows if row['item']['actions'].shape[1] >= protocol['horizon']]
    terminal = [row for row in rows if row['item']['actions'].shape[1] < protocol['horizon']]
    # One terminal two-step anchor must not truncate three unrelated H16
    # branches. Reserve a tiny fallback budget, but first use full-length data.
    reserve = min(len(terminal), protocol['max_branches'], max(0, protocol['max_source_probes']-1))
    full_budget = protocol['max_source_probes']-reserve if full else 0
    for row in full:
        if probes >= full_budget: break
        probes += 1
        try: probed.append(_source_labels(row,protocol,dt))
        except ValueError as error:
            skips[str(error)] += 1
    fallback = not probed and bool(terminal)
    if fallback:
        for row in terminal:
            if probes >= protocol['max_source_probes']: break
            probes += 1
            try: probed.append(_source_labels(row,protocol,dt))
            except ValueError as error: skips[str(error)] += 1
    earliest = {}
    for row in probed:
        error = row['first_missed_block']
        if error is not None: earliest[row['actor']] = min(earliest.get(row['actor'],error),error)
    # Free false stops remain diagnostic/contrast anchors. They must never
    # remove the later pre-wall chunk or all real sliding-contact candidates.
    eligible = [row for row in probed if row['first_wall_chunk']
        or row['actor'] not in earliest or row['step'] < earliest[row['actor']]]
    def priority(row):
        error = min(x for x in (row['first_missed_block'],row['first_wall_step'],10**9) if x is not None)
        contact = row['cohort'] in AXIS_COHORTS[1:]
        return (not row['first_wall_chunk'],not contact,error==10**9,error,row['step'],row['actor'])
    eligible.sort(key=priority)
    selected,used = [],set()
    for cohort in (*AXIS_COHORTS[1:],AXIS_COHORTS[0],'single_axis','partial','idle'):
        same = [row for row in eligible if row['cohort']==cohort]
        if same and len(selected)<protocol['max_branches']:
            row = next((r for r in same if r['actor'] not in used),same[0])
            selected.append(row); used.add(row['actor'])
    for row in eligible:
        if len(selected)>=protocol['max_branches']: break
        if not any(row is item for item in selected):
            selected.append(row)
    return selected,{'source_probes':probes,'legal_source_probes':len(probed),
        'terminal_candidates_deferred':len(terminal),'short_horizon_fallback':fallback,
        'skip_counts':dict(skips),'complete_first_wall_actor_count':len(first_wall),
        'source_probe_cohort_counts':dict(Counter(row['cohort'] for row in probed)),
        'selected_cohort_counts':dict(Counter(row['cohort'] for row in selected)),
        'false_stop_probe_count':sum(row['first_false_stop'] is not None for row in probed),
        'missed_block_probe_count':sum(row['first_missed_block'] is not None for row in probed),
        'free_velocity_residual_axes':sum(row['free_velocity_residual_axes'] for row in probed),
        'partial_velocity_residual_axes':sum(row['partial_velocity_residual_axes'] for row in probed),
        'axis_outcome_coverage_complete':all(any(r['cohort']==name for r in selected) for name in AXIS_COHORTS),
        'contact_error_detection_scope':'complete_wall_scan_bounded_Source_probes',
        'selected_anchors':[{'actor':r['actor'],'step':r['step'],'first_wall_step':r['first_wall_step'],
            'first_physical_error':r['first_physical_error'],'cohort':r['cohort'],
            'first_false_stop':r['first_false_stop'],'first_missed_block':r['first_missed_block'],
            'cohort_label_offset':r['cohort_label_offset']} for r in selected]}


def contact_branches_backward(model,candidates,*,contact_branch_protocol,contact_protocol=None,
                              amp=None,check_gradients=True):
    """Fresh short prior graphs, same detached h/z, and no optimizer/renderer."""
    protocol = normalize_contact_branch_protocol(contact_branch_protocol)
    if protocol is None: raise ValueError('Contact branch backward requires explicit opt-in')
    config = getattr(model,'config',None)
    if not getattr(config,'navigation_only',False):
        raise ValueError('Own-contact Source branches require the navigation-only profile')
    dt = 1/getattr(config,'simulation_hz',30)
    chosen,selection = select_contact_branches(candidates,contact_branch_protocol=protocol,dt=dt)
    metrics = {'loss':0.,'branches':len(chosen),'protocol_version':1,**selection,
        'prior_initial_memory_preserved':True,'initial_state_provenance':'actualownprefix',
        'memory_provenance':'actualownprefix','new_observed_rgb_frames':0,
        'source_future_inputs':0,'state_resets':0,'memory_resets':0,'renderer_calls':0,'optimizer_steps':0}
    if not chosen: return metrics
    horizon = min(len(row['labels'])-1 for row in chosen)
    state = torch.stack([row['item']['state'][row['actor']] for row in chosen]).detach()
    memory = torch.stack([row['item']['memory'][row['actor']] for row in chosen]).detach()
    actions = torch.stack([row['item']['actions'][row['actor'],:horizon] for row in chosen]).detach()
    grid = torch.stack([row['item']['map'][row['actor']] for row in chosen]).detach()
    target = torch.tensor(np.stack([row['labels'][:horizon+1] for row in chosen]),device=state.device,dtype=torch.float32)
    counts = Counter(row['cohort'] for row in chosen)
    group_weight = state.new_tensor([1/(len(counts)*counts[row['cohort']]) for row in chosen])
    contact = normalize_contact_protocol(contact_protocol or CONTACT_PROTOCOL_V2)
    autocast = nullcontext if amp is None else amp
    with torch.no_grad(),autocast(): features = model.encode_map(grid).detach()
    initial,expected_memory_shape = state,memory.shape
    outputs,first_result = [],None
    with autocast():
        for offset in range(horizon):
            result = model.transition(state,actions[:,offset],features,memory)
            if (result['state'].shape!=initial.shape or result['memory'].shape!=expected_memory_shape
                    or not bool(torch.isfinite(result['state']).all() and torch.isfinite(result['memory']).all())):
                raise FloatingPointError('Nonfinite own-contact branch state or memory; no repairs')
            if first_result is None: first_result=result
            state,memory = result['state'],result['memory']
            outputs.append(state)
        prediction = torch.stack(outputs,1)
        scale = 16/(2.4*dt)
        xy_per_branch = F.smooth_l1_loss((prediction[:,:,:2].float()-target[:,1:,:2])*scale,
            torch.zeros_like(prediction[:,:,:2]),reduction='none').mean((1,2))
        xy = (xy_per_branch*group_weight).sum()
        own_previous = torch.cat([initial[:,None,:2],prediction[:,:-1,:2]],1)
        wall = swept_player_wall_loss(own_previous,prediction[:,:,:2],target[:,:-1,:2],
            target[:,1:,:2],grid,group_weight[:,None].expand(-1,horizon)/horizon,contact_protocol=contact)
        anchor_labels = prepare_contact_targets(initial,target[:,1],actions[:,0],contact_protocol=contact,dt=dt)
        anchor = differentiable_contact_terms(first_result,initial,target[:,1],actions[:,0],grid,
            contact_protocol=contact,supervision=anchor_labels,index=0)
        objective = (protocol['xy_weight']*xy+protocol['wall_weight']*wall+
            protocol['anchor_bce_weight']*anchor['pass_bce']+
            protocol['anchor_margin_weight']*anchor['endpoint_margin'])
        weighted = protocol['weight']*objective
    if not bool(torch.isfinite(weighted)): raise FloatingPointError('Nonfinite own-contact branch loss')
    weighted.backward()
    if check_gradients: finite_gradient_norm(model.named_parameters(),context='Own-contact branch backward')
    metrics.update(loss=float(weighted.detach()),unweighted_loss=float(objective.detach()),
        xy_loss=float(xy.detach()),wall_loss=float(wall.detach()),
        anchor_bce=float(anchor['pass_bce'].detach()),anchor_margin=float(anchor['endpoint_margin'].detach()),
        horizon=horizon,forward_transitions=len(chosen)*horizon,backward_calls=1,
        canonical_max_abs_difference=max(row['canonical_max_abs_difference'] for row in chosen))
    for offset in protocol['offsets']:
        if offset<=horizon:
            error=((prediction[:,offset-1,:2].detach().float()-target[:,offset,:2])*16).norm(dim=-1)
            metrics[f'xy_offset{offset}_tiles_mean']=float(error.mean())
    return metrics

