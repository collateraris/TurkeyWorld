"""Bounded training-only Source targets for actual own-state/h-z combat prefixes."""
from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
import math

import numpy as np
import torch
from torch.nn import functional as F

from .engine import Game, STATE_DIM
from .gradient_safety import finite_gradient_norm
from .rssm_contact_branches import capture_contact_candidate
from .runtime_checks import circle_is_clear, segment_is_clear
from .rssm_combat_objectives import prepare_combat_supervision, combat_supervised_terms
from .train import state_loss
from .trajectory_objectives import trajectory_state_losses


COMBAT_BRANCH_DEFAULTS = {'version': 1, 'max_branches': 4, 'horizon': 16,
    'max_source_probes': 64, 'weight': .25, 'xy_weight': 1., 'state_weight': 1.,
    'wall_weight': .05, 'anchor_combat_weight': 1., 'canonical_atol': 2e-7,
    'npc_error_tiles': .25, 'offsets': [1, 4, 8, 16], 'numeric_state_feedback': 'detached'}


def normalize_combat_branch_protocol(value):
    if value is None: return None
    if not isinstance(value, dict) or set(value)-set(COMBAT_BRANCH_DEFAULTS):
        raise ValueError('Unknown own-combat branch settings')
    result = {**COMBAT_BRANCH_DEFAULTS, **value}; result['offsets'] = list(result['offsets'])
    if result['version'] != 1 or isinstance(result['version'], bool): raise ValueError('Combat branch version must be1')
    for key, limit in (('max_branches', 4), ('horizon', 16), ('max_source_probes', 64)):
        if not isinstance(result[key], int) or isinstance(result[key], bool) or not 1 <= result[key] <= limit:
            raise ValueError('Invalid combat branch bound: '+key)
    for key in ('weight', 'xy_weight', 'state_weight', 'wall_weight', 'anchor_combat_weight',
                'canonical_atol', 'npc_error_tiles'):
        if not isinstance(result[key], (int,float)) or isinstance(result[key],bool) or not math.isfinite(result[key]) or result[key] < 0:
            raise ValueError('Invalid finite combat branch weight/tolerance: '+key)
    if not 0 < result['canonical_atol'] <= 2e-7 or result['numeric_state_feedback'] != 'detached':
        raise ValueError('Combat branches require strict canonical anchors and detached numeric feedback')
    if (not result['offsets'] or result['offsets'] != sorted(set(result['offsets']))
            or any(not isinstance(x,int) or isinstance(x,bool) or x < 1 or x > 16 for x in result['offsets'])):
        raise ValueError('Combat branch offsets must be unique sorted1..16')
    result['offsets'] = [x for x in result['offsets'] if x <= result['horizon']]
    if not result['offsets']: raise ValueError('No reporting offset inside combat branch')
    return result


def capture_combat_candidate(step, state, memory, actions, grid, prediction):
    result = capture_contact_candidate(step, state, memory, actions, grid, prediction)
    result['source_split'] = 'train'
    return result


def _rows(candidates):
    rows, first_wall = [], {}
    for item in sorted(candidates, key=lambda x:x['step']):
        if (item.get('memory_provenance') != 'actualownprefix' or item.get('new_observed_rgb_frames') != 0
                or item.get('source_split') != 'train'):
            raise ValueError('Combat branches require actual TRAIN own-prefix memory and no observations')
        states = item['state'].float().cpu().numpy(); pred = item['prediction'].float().cpu().numpy()
        maps = item['map'].cpu().numpy()
        for actor, initial in enumerate(states):
            previous = initial
            for offset, now in enumerate(pred[actor]):
                wall = (not circle_is_clear(maps[actor], now[:2]*16, .2)
                    or not segment_is_clear(maps[actor], previous[:2]*16, now[:2]*16, .2))
                if now[13] >= .5 and now[12] > 0:
                    wall |= not circle_is_clear(maps[actor], now[8:10]*16, .25)
                    if previous[13] >= .5 and previous[12] > 0:
                        wall |= not segment_is_clear(maps[actor], previous[8:10]*16, now[8:10]*16, .25)
                if wall:
                    first_wall.setdefault(actor, item['step']+offset+1); break
                previous = now
            rows.append({'item':item, 'actor':actor, 'step':item['step'], 'state_numpy':initial.copy(),
                'map_numpy':maps[actor].copy(), 'prediction_numpy':pred[actor].copy()})
    for row in rows:
        first = first_wall.get(row['actor'])
        row['first_wall_step'] = first
        row['pre_wall'] = first is not None and row['step'] < first <= row['step']+len(row['prediction_numpy'])
    rows = [row for row in rows if row['first_wall_step'] is None or row['step'] < row['first_wall_step']]
    rows.sort(key=lambda row:(not row['pre_wall'], row['step'] != 0, row['step'], row['actor']))
    return rows, first_wall


def _source_labels(row, protocol, dt):
    """Canonical counterfactual Source replay AFTER own-prefix capture only."""
    state, grid = row['state_numpy'], row['map_numpy']
    if (not np.isfinite(state).all() or state[4] <= 0 or not circle_is_clear(grid,state[:2]*16,.2)
            or state[13] >= .5 and state[12] > 0 and not circle_is_clear(grid,state[8:10]*16,.25)):
        raise ValueError('invalid_own_anchor_geometry_or_player_dead')
    game = Game(seed=0, endless=True, enemies_enabled=True)
    game.map = np.asarray(grid,np.uint8).copy(); game.map_height,game.map_width = game.map.shape
    game.set_state_vector(state)
    canonical = game.state_vector()
    if not np.allclose(canonical,state,rtol=0,atol=protocol['canonical_atol']):
        raise ValueError('noncanonical_own_anchor_no_repair')
    actions = row['item']['actions'][row['actor'],:protocol['horizon']].float().cpu().numpy()
    labels, events, hits, attacks = [state.copy()], [], [], []
    for action in actions:
        event = game.step(action,dt=dt)
        labels.append(game.state_vector().copy());events.append(Game.event_vector(event))
        hits.append(event.get('hit_id',-1));attacks.append([float(0 in event.get('attacker_ids',[]))])
    labels = np.stack(labels); events = np.asarray(events,np.float32)
    own = row['prediction_numpy'][:len(actions)]; count = len(own)
    true_alive = (labels[:,13] >= .5) & (labels[:,12] > 0)
    own_alive = (own[:,13] >= .5) & (own[:,12] > 0)
    revived = ~true_alive[:-1] & true_alive[1:]
    error = np.linalg.norm((own[:,8:10]-labels[1:count+1,8:10])*16,axis=-1)
    spawn_bad = revived[:count] & ((~own_alive) | (error > protocol['npc_error_tiles']))
    life_bad = own_alive != true_alive[1:count+1]
    health_bad = np.abs(own[:,12]-labels[1:count+1,12]) > 1e-5
    player_health_bad = np.abs(own[:,4]-labels[1:count+1,4]) > 1e-5
    continuing = true_alive[:count] & true_alive[1:count+1]
    motion_bad = continuing & (error > protocol['npc_error_tiles'])
    first = np.flatnonzero(spawn_bad | life_bad | health_bad | player_health_bad | motion_bad)
    cohort = ('spawn' if spawn_bad.any() or revived.any() else 'combat' if (life_bad|health_bad|player_health_bad).any()
              else 'npc_wall' if row['pre_wall'] else 'npc_motion' if motion_bad.any() else 'contrast')
    return {**row, 'labels':labels, 'events':events, 'hits':np.asarray(hits,np.int64),
        'attacks':np.asarray(attacks,np.float32), 'cohort':cohort,
        'first_source_error':row['step']+int(first[0])+1 if len(first) else None,
        'canonical_max_abs_difference':float(np.max(np.abs(canonical-state)))}


def select_combat_branches(candidates, *, combat_branch_protocol, dt=1/30):
    protocol = normalize_combat_branch_protocol(combat_branch_protocol)
    rows, walls = _rows(candidates); skips = Counter(); probed = []; probes = 0
    for row in rows:
        if row['item']['actions'].shape[1] < protocol['horizon']:
            skips['terminal_less_than_full_horizon'] += 1; continue
        if probes >= protocol['max_source_probes']: break
        probes += 1
        try: probed.append(_source_labels(row,protocol,dt))
        except ValueError as error: skips[str(error)] += 1
    earliest = {}
    for row in probed:
        error = row['first_source_error']
        if error is not None: earliest[row['actor']] = min(earliest.get(row['actor'],error),error)
    # A prior harmless drift/health error must not remove the later exact
    # pre-wall or pre-spawn own-memory query. Every anchor is checked legal.
    eligible = [r for r in probed if r['pre_wall'] or r['cohort']=='spawn'
        or r['actor'] not in earliest or r['step'] < earliest[r['actor']]]
    eligible.sort(key=lambda r:(not r['pre_wall'], r['first_source_error'] is None, r['step'],r['actor']))
    chosen=[]
    initial=[r for r in eligible if r['step']==0]
    if initial: chosen.append(initial[0])
    for cohort in ('spawn','npc_wall','combat','npc_motion','contrast'):
        matching=[r for r in eligible if r['cohort']==cohort and all(r is not x for x in chosen)]
        if matching and len(chosen)<protocol['max_branches']:
            used={r['actor'] for r in chosen}
            chosen.append(next((r for r in matching if r['actor'] not in used),matching[0]))
    for row in eligible:
        if len(chosen)>=protocol['max_branches']:break
        if all(row is not x for x in chosen):chosen.append(row)
    return chosen, {'source_probes':probes, 'legal_source_probes':len(probed), 'skip_counts':dict(skips),
        'complete_first_wall_actor_count':len(walls),
        'source_probe_cohorts':dict(Counter(r['cohort'] for r in probed)),
        'selected_cohorts':dict(Counter(r['cohort'] for r in chosen)),
        'first_error_scope':'complete_wall_scan_and_bounded_Source_counterfactual_probes',
        'selected_anchors':[{'actor':r['actor'],'step':r['step'],'cohort':r['cohort'],
            'first_wall_step':r['first_wall_step'],'first_source_error':r['first_source_error']}for r in chosen]}


def combat_branches_backward(model, candidates, *, combat_branch_protocol, amp=None, check_gradients=True,loss_groups=None):
    protocol=normalize_combat_branch_protocol(combat_branch_protocol)
    if protocol is None or getattr(model.config,'navigation_only',True):
        raise ValueError('Combat branches require explicit opt-in and the actual one-enemy profile')
    chosen,selection=select_combat_branches(candidates,combat_branch_protocol=protocol,dt=1/model.config.simulation_hz)
    metrics={'loss':0.,'branches':len(chosen),**selection,'protocol_version':1,
        'memory_provenance':'actualownprefix','initial_state_provenance':'actualownprefix',
        'prior_initial_memory_preserved':True,'new_observed_rgb_frames':0,'source_future_inputs':0,
        'state_resets':0,'memory_resets':0,'renderer_calls':0,'optimizer_steps':0,
        'numeric_state_feedback':'detached','h_z_bptt_preserved':True}
    if not chosen:return metrics
    initial=torch.stack([r['item']['state'][r['actor']]for r in chosen]).detach()
    memory=torch.stack([r['item']['memory'][r['actor']]for r in chosen]).detach()
    actions=torch.stack([r['item']['actions'][r['actor'],:protocol['horizon']]for r in chosen]).detach()
    grid=torch.stack([r['item']['map'][r['actor']]for r in chosen]).detach()
    target=torch.tensor(np.stack([r['labels']for r in chosen]),device=initial.device,dtype=torch.float32)
    autocast=nullcontext if amp is None else amp
    with torch.no_grad(),autocast():fmap=model.encode_map(grid).detach()
    state=initial;outputs=[];first_result=None;shape=memory.shape
    with autocast():
        for index in range(protocol['horizon']):
            result=model.transition(state.detach(),actions[:,index],fmap,memory)
            if (result['state'].shape!=initial.shape or result['memory'].shape!=shape
                    or not bool(torch.isfinite(result['state']).all() and torch.isfinite(result['memory']).all())):
                raise FloatingPointError('Nonfinite own-memory combat branch; no repair')
            if first_result is None:first_result=result
            state,memory=result['state'],result['memory'];outputs.append(state)
        prediction=torch.stack(outputs,1)
        alive=(target[:,:,13]>=.5)&(target[:,:,12]>0);mask=alive[:,:-1]&alive[:,1:]
        xy=prediction.sum()*0;wall=xy
        if loss_groups is not None:
            player_xy=xy;enemy_xy=xy;player_wall=xy;enemy_wall=xy
        for offset in protocol['offsets']:
            player=F.smooth_l1_loss(prediction[:,offset-1,:2]*16,target[:,offset,:2]*16)
            error=F.smooth_l1_loss(prediction[:,offset-1,8:10]*16,target[:,offset,8:10]*16,reduction='none').mean(-1)
            npc=(error*mask[:,offset-1]).sum()/mask[:,offset-1].sum().clamp_min(1)
            xy=xy+(player+npc)/len(protocol['offsets'])
            if loss_groups is not None:
                player_xy=player_xy+player/len(protocol['offsets'])
                enemy_xy=enemy_xy+npc/len(protocol['offsets'])
        for index in range(protocol['horizon']):
            previous=initial if index==0 else prediction[:,index-1]
            geometry,_=trajectory_state_losses(prediction[:,index:index+1],target[:,index],target[:,index+1:index+2],grid,
                offsets=[1],xy_weight=0.,previous_prediction=previous)
            wall=wall+(geometry['ar_wall_player']+geometry['ar_wall_npc'])/protocol['horizon']
            if loss_groups is not None:
                player_wall=player_wall+geometry['ar_wall_player']/protocol['horizon']
                enemy_wall=enemy_wall+geometry['ar_wall_npc']/protocol['horizon']
        state_term=state_loss(prediction.flatten(0,1),target[:,1:].flatten(0,1))
        one={'states':target[:,:2].clone(),'target_states':target[:,1:2], 'actions':actions[:,:1],
            'events':torch.tensor(np.stack([r['events'][:1]for r in chosen]),device=initial.device),
            'hit_ids':torch.tensor(np.stack([r['hits'][:1]for r in chosen]),device=initial.device),
            'attack_flags':torch.tensor(np.stack([r['attacks'][:1]for r in chosen]),device=initial.device)}
        one['states'][:,0]=initial
        supervision=prepare_combat_supervision(one,context=1)
        combat,counts=combat_supervised_terms(first_result,initial,target[:,1],actions[:,0],supervision=supervision,index=0)
        anchor=sum(combat.values())
        objective=protocol['xy_weight']*xy+protocol['state_weight']*state_term+protocol['wall_weight']*wall+protocol['anchor_combat_weight']*anchor
        weighted=protocol['weight']*objective
        if loss_groups is not None:
            from .rssm_loss_balance import split_state_loss,combat_loss_groups
            state_parts=split_state_loss(prediction.flatten(0,1),target[:,1:].flatten(0,1))
            anchor_parts=combat_loss_groups(combat)
            groups={
                'player':protocol['xy_weight']*player_xy+protocol['wall_weight']*player_wall,
                'enemy':protocol['xy_weight']*enemy_xy+protocol['wall_weight']*enemy_wall,
                'interaction':prediction.sum()*0}
            for group in groups:
                groups[group]=protocol['weight']*(groups[group]+protocol['state_weight']*state_parts[group]
                    +protocol['anchor_combat_weight']*anchor_parts.get(group,0))
    if not bool(torch.isfinite(weighted)):raise FloatingPointError('Nonfinite own-combat branch loss')
    actual=weighted if loss_groups is None else loss_groups.combine(groups)
    if not bool(torch.isfinite(actual)):raise FloatingPointError('Nonfinite balanced own-combat branch loss')
    actual.backward()
    if check_gradients:finite_gradient_norm(model.named_parameters(),context='Own-combat branch backward')
    metrics.update(loss=float(weighted.detach()),unweighted_loss=float(objective.detach()),xy_loss=float(xy.detach()),
        state_loss=float(state_term.detach()),wall_loss=float(wall.detach()),anchor_combat_loss=float(anchor.detach()),
        horizon=protocol['horizon'],forward_transitions=len(chosen)*protocol['horizon'],backward_calls=1,
        canonical_max_abs_difference=max(r['canonical_max_abs_difference']for r in chosen),
        source_hit_count=int(sum((r['hits']>=0).sum()for r in chosen)),
        source_attack_count=int(sum(r['attacks'].sum()for r in chosen)),
        source_respawn_count=int(sum(((r['labels'][:-1,13]<.5)&(r['labels'][1:,13]>=.5)).sum()for r in chosen)),
        anchor_source_counts={name:float(value)for name,value in counts.items()})
    if loss_groups is not None:metrics['balanced_loss']=float(actual.detach())
    for offset in protocol['offsets']:
        error=((prediction[:,offset-1,8:10].detach()-target[:,offset,8:10])*16).norm(dim=-1)
        eligible=mask[:,offset-1];metrics[f'npc_xy_offset{offset}_count']=int(eligible.sum())
        metrics[f'npc_xy_offset{offset}_tiles_mean']=float(error[eligible].mean()) if eligible.any() else None
        metrics[f'player_xy_offset{offset}_tiles_mean']=float(((prediction[:,offset-1,:2].detach()-target[:,offset,:2])*16).norm(dim=-1).mean())
    return metrics
