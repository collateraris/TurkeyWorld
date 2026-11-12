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
COMBAT_BRANCH_V2_DEFAULTS = {**COMBAT_BRANCH_DEFAULTS, 'version': 2, 'max_branches': 8,
    'max_direct_source_queries': 128, 'local_contact_weight': .25}


def normalize_combat_branch_protocol(value):
    if value is None: return None
    if not isinstance(value, dict):
        raise ValueError('Unknown own-combat branch settings')
    version = value.get('version', 1)
    if type(version) is not int or version not in (1, 2):
        raise ValueError('Combat branch version must be1 or2')
    defaults = COMBAT_BRANCH_DEFAULTS if version == 1 else COMBAT_BRANCH_V2_DEFAULTS
    if set(value)-set(defaults):
        raise ValueError('Unknown own-combat branch settings')
    result = {**defaults, **value}; result['offsets'] = list(result['offsets'])
    # Larger B64 runs opt in through their immutable revision. Existing v2
    # defaults remain8/64/128; accepting an explicit budget never enlarges one.
    bounds = [('max_branches', 4 if version == 1 else 16), ('horizon', 16),
        ('max_source_probes', 64 if version == 1 else 128)]
    if version == 2: bounds.append(('max_direct_source_queries', 256))
    for key, limit in bounds:
        if not isinstance(result[key], int) or isinstance(result[key], bool) or not 1 <= result[key] <= limit:
            raise ValueError('Invalid combat branch bound: '+key)
    for key in ('weight', 'xy_weight', 'state_weight', 'wall_weight', 'anchor_combat_weight',
                'canonical_atol', 'npc_error_tiles', *(['local_contact_weight'] if version == 2 else [])):
        if not isinstance(result[key], (int,float)) or isinstance(result[key],bool) or not math.isfinite(result[key]) or result[key] < 0:
            raise ValueError('Invalid finite combat branch weight/tolerance: '+key)
    if not 0 < result['canonical_atol'] <= 2e-7 or result['numeric_state_feedback'] != 'detached':
        raise ValueError('Combat branches require strict canonical anchors and detached numeric feedback')
    if (not result['offsets'] or result['offsets'] != sorted(set(result['offsets']))
            or any(not isinstance(x,int) or isinstance(x,bool) or x < 1 or x > 16 for x in result['offsets'])):
        raise ValueError('Combat branch offsets must be unique sorted1..16')
    result['offsets'] = [x for x in result['offsets'] if x <= result['horizon']]
    if not result['offsets']: raise ValueError('No reporting offset inside combat branch')
    if version == 2 and result['max_direct_source_queries'] < result['max_branches']*result['horizon']:
        raise ValueError('V2 direct Source budget must cover every selected branch timestep')
    return result


def capture_combat_candidate(step, state, memory, actions, grid, prediction):
    result = capture_contact_candidate(step, state, memory, actions, grid, prediction)
    result['source_split'] = 'train'
    return result


def _rows(candidates, *, recovery=False):
    rows, first_wall, respawns = [], {}, {}
    for item in sorted(candidates, key=lambda x:x['step']):
        if (item.get('memory_provenance') != 'actualownprefix' or item.get('new_observed_rgb_frames') != 0
                or item.get('source_split') != 'train'):
            raise ValueError('Combat branches require actual TRAIN own-prefix memory and no observations')
        states = item['state'].float().cpu().numpy(); pred = item['prediction'].float().cpu().numpy()
        maps = item['map'].cpu().numpy()
        for actor, initial in enumerate(states):
            previous = initial
            if recovery:
                for offset, now in enumerate(pred[actor]):
                    if not (previous[13] >= .5 and previous[12] > 0) and now[13] >= .5 and now[12] > 0:
                        respawns.setdefault(actor, []).append(item['step']+offset+1)
                    previous = now
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
        if recovery:
            row['post_spawn_recovery'] = (row['state_numpy'][13] >= .5 and row['state_numpy'][12] > 0
                and any(0 <= row['step']-tick <= 16 for tick in respawns.get(row['actor'], [])))
    rows = [row for row in rows if row['first_wall_step'] is None or row['step'] < row['first_wall_step']
        or recovery and row['post_spawn_recovery']]
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
    if protocol['version'] == 2 and row.get('post_spawn_recovery') and cohort not in ('spawn', 'combat'):
        cohort = 'post_spawn_recovery'
    return {**row, 'labels':labels, 'events':events, 'hits':np.asarray(hits,np.int64),
        'attacks':np.asarray(attacks,np.float32), 'cohort':cohort,
        'first_source_error':row['step']+int(first[0])+1 if len(first) else None,
        'canonical_max_abs_difference':float(np.max(np.abs(canonical-state)))}


def select_combat_branches(candidates, *, combat_branch_protocol, dt=1/30, late_prefix_protocol=None):
    protocol = normalize_combat_branch_protocol(combat_branch_protocol)
    rows, walls = _rows(candidates, recovery=protocol['version'] == 2)
    if protocol['version'] == 2:
        # First-contact and legal post-spawn anchors must enter the bounded
        # probe set rather than disappearing behind all step0 actors.
        rows.sort(key=lambda r:(not r['pre_wall'], not r.get('post_spawn_recovery', False),
            r['step'] != 0, r['step'], r['actor']))
    late = None
    if late_prefix_protocol is not None:
        from .rssm_late_prefix_recovery import (normalize_late_prefix_protocol,
            order_late_prefix_probes, is_late_prefix, select_late_prefix_rows)
        late = normalize_late_prefix_protocol(late_prefix_protocol)
        if protocol['version'] != 2:
            raise ValueError('Late own-prefix recovery requires combat branches version2')
        rows = order_late_prefix_probes(rows, late_prefix_protocol=late, budget=protocol['max_source_probes'])
    skips = Counter(); probed = []; probes = 0
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
        or protocol['version'] == 2 and r.get('post_spawn_recovery', False)
        or late is not None and is_late_prefix(r, late)
        or r['actor'] not in earliest or r['step'] < earliest[r['actor']]]
    eligible.sort(key=lambda r:(not r['pre_wall'], r['first_source_error'] is None, r['step'],r['actor']))
    chosen=[]
    initial=[r for r in eligible if r['step']==0]
    if initial: chosen.append(initial[0])
    if protocol['version'] == 2:
        contacts = sorted((r for r in eligible if r['pre_wall']),
            key=lambda r:(r['first_wall_step'], r['step'], r['actor']))
        for row in contacts:
            if len(chosen) >= min(protocol['max_branches'], 3): break
            if all(row is not old for old in chosen): chosen.append(row)
        recovery = [r for r in eligible if r.get('post_spawn_recovery', False)
            and all(r is not old for old in chosen)]
        if recovery and len(chosen) < protocol['max_branches']:
            chosen.append(recovery[0])
    cohorts = (('spawn', 'npc_wall', 'post_spawn_recovery', 'combat', 'npc_motion', 'contrast')
        if protocol['version'] == 2 else ('spawn','npc_wall','combat','npc_motion','contrast'))
    for cohort in cohorts:
        matching=[r for r in eligible if r['cohort']==cohort and all(r is not x for x in chosen)]
        if matching and len(chosen)<protocol['max_branches']:
            used={r['actor'] for r in chosen}
            chosen.append(next((r for r in matching if r['actor'] not in used),matching[0]))
    for row in eligible:
        if len(chosen)>=protocol['max_branches']:break
        if all(row is not x for x in chosen):chosen.append(row)
    late_report = {}
    if late is not None:
        chosen, late_report = select_late_prefix_rows(eligible, chosen,
            late_prefix_protocol=late, max_branches=protocol['max_branches'])
    report = {'source_probes':probes, 'legal_source_probes':len(probed), 'skip_counts':dict(skips),
        **late_report,
        'complete_first_wall_actor_count':len(walls),
        'source_probe_cohorts':dict(Counter(r['cohort'] for r in probed)),
        'selected_cohorts':dict(Counter(r['cohort'] for r in chosen)),
        'first_error_scope':'complete_wall_scan_and_bounded_Source_counterfactual_probes',
        'selected_anchors':[{'actor':r['actor'],'step':r['step'],'cohort':r['cohort'],
            'first_wall_step':r['first_wall_step'],'first_source_error':r['first_source_error']}for r in chosen]}
    if protocol['version'] == 2:
        report.update(selected_first_contact_count=int(sum(r['pre_wall'] for r in chosen)),
            selected_post_spawn_recovery_count=int(sum(r.get('post_spawn_recovery', False) for r in chosen)))
    return chosen, report


def _source_step_at_own_state(state, action, grid, protocol, dt):
    """Loss-only one-step label from an exact legal own numerical state.

    No repair, rasterization, model calls or state/memory replacement occurs.
    A terminal player is a valid conditional query; event eligibility handles
    its absent actions instead of fabricating another living combat state.
    """
    if (not np.isfinite(state).all() or not circle_is_clear(grid, state[:2]*16, .2)
            or state[13] >= .5 and state[12] > 0 and not circle_is_clear(grid, state[8:10]*16, .25)):
        raise ValueError('invalid_direct_own_state_geometry')
    game = Game(seed=0, endless=True, enemies_enabled=True)
    game.map = np.asarray(grid, np.uint8).copy()
    game.map_height, game.map_width = game.map.shape
    game.set_state_vector(state)
    canonical = game.state_vector()
    if not np.allclose(canonical, state, rtol=0, atol=protocol['canonical_atol']):
        raise ValueError('noncanonical_direct_own_state_no_repair')
    event = game.step(action, dt=dt)
    return {'previous': state.copy(), 'target': game.state_vector().copy(),
        'events': Game.event_vector(event), 'hit': event.get('hit_id', -1),
        'attack': [float(0 in event.get('attacker_ids', []))]}


def _direct_head_terms(results, previous_states, actions, grid, protocol, dt, *, contact_protocol=None,
                       kinematic_protocol=None):
    """Global reduction over all valid conditional actor x timestep queries.

    Flattening the valid pairs into a single B_valid x H1 batch gives the same
    event-balanced denominators as B x H, without claiming that independently
    queried Source next states form a contiguous Source rollout. Every prior
    remains the already computed own-memory prediction at that exact state.
    """
    count, horizon = actions.shape[:2]
    if count*horizon > protocol['max_direct_source_queries']:
        raise ValueError('Selected branches exceed the immutable direct Source query budget')
    initial_device = actions.device
    own = torch.stack(previous_states, 1).detach().float().cpu().numpy()
    controls, maps = actions.detach().float().cpu().numpy(), grid.detach().cpu().numpy()
    valid, labels, skips = [], [], Counter()
    per_step = [dict(queries=count, valid=0, hit=0, attack=0, respawn=0) for _ in range(horizon)]
    # This happens after the entire autonomous branch has completed. Labels
    # cannot change a subsequent neural condition or its recurrent memory.
    for index in range(horizon):
        for actor in range(count):
            try:
                label = _source_step_at_own_state(own[actor, index], controls[actor, index],
                    maps[actor], protocol, dt)
            except ValueError as error:
                skips[str(error)] += 1
                continue
            valid.append((index, actor)); labels.append(label)
            per_step[index]['valid'] += 1
            per_step[index]['hit'] += int(label['hit'] >= 0)
            per_step[index]['attack'] += int(label['attack'][0] > .5)
            per_step[index]['respawn'] += int(not (label['previous'][13] >= .5 and label['previous'][12] > 0)
                and label['target'][13] >= .5 and label['target'][12] > 0)
    metrics = {'direct_source_query_attempts': count*horizon, 'direct_source_step_calls': len(valid),
        'direct_source_invalid_skip_counts': dict(skips), 'direct_supervised_queries': len(valid),
        'direct_supervised_horizon': horizon, 'direct_per_step_counts': per_step,
        'direct_source_hit_count': sum(row['hit'] for row in per_step),
        'direct_source_attack_count': sum(row['attack'] for row in per_step),
        'direct_source_respawn_count': sum(row['respawn'] for row in per_step),
        'direct_late_respawn_count': sum(row['respawn'] for row in per_step[1:]),
        'direct_label_scope': 'one_Source_step_from_exact_actual_own_previous_state',
        'direct_source_queries_after_own_rollout': True, 'direct_source_feedback': False,
        'direct_target_rgb_encoded': 0}
    zero = results[0]['state'].sum()*0
    if not valid:
        return {name: zero for name in ('combat_events', 'combat_target', 'combat_attack',
            'combat_damage_npc', 'combat_damage_player', 'combat_death_npc', 'combat_spawn', 'combat_npc_motion')}, {}, metrics, zero
    tensor = lambda values, dtype=torch.float32: torch.tensor(np.asarray(values), device=initial_device, dtype=dtype)
    previous = tensor([row['previous'] for row in labels])
    target = tensor([row['target'] for row in labels])
    selected_actions = torch.stack([actions[actor, index] for index, actor in valid]).detach()
    batch = {'states': torch.stack([previous, target], 1), 'target_states': target[:, None],
        'actions': selected_actions[:, None],
        'events': tensor([row['events'] for row in labels])[:, None],
        'hit_ids': tensor([row['hit'] for row in labels], torch.long)[:, None],
        'attack_flags': tensor([row['attack'] for row in labels])[:, None]}
    names = ('state', 'event_logits', 'target_logits', 'attack_logits', 'shot', 'respawn_flags', 'player_pass_logits',
        'spawn_logits', 'spawn_anchor_logits', 'spawn_fallback_logits')
    prior = {name: torch.cat([results[index][name][actor:actor+1] for index, actor in valid], 0)
        for name in names if name in results[0]}
    supervision = prepare_combat_supervision(batch, context=1, simulation_hz=1/dt,
        target_supervision='shot_distance_balanced', kinematic_protocol=kinematic_protocol)
    terms, counts = combat_supervised_terms(prior, previous, target, selected_actions,
        supervision=supervision, index=0)
    if kinematic_protocol is not None:
        metrics.update({name: float(value) for name, value in counts.items() if name.startswith('combat_kinematic_')})
        metrics.update({name+'_loss': float(value.detach()) for name, value in terms.items()
                        if name.startswith('combat_kinematic_')})
    local_contact = zero
    if contact_protocol is not None and protocol['local_contact_weight']:
        from .contact_objectives import (normalize_contact_protocol, prepare_contact_targets,
            differentiable_contact_terms)
        config = normalize_contact_protocol(contact_protocol)
        contact_labels = prepare_contact_targets(previous, target, selected_actions,
            contact_protocol=config, dt=dt)
        selected_grid = torch.stack([grid[actor] for _, actor in valid])
        contact = differentiable_contact_terms(prior, previous, target, selected_actions, selected_grid,
            contact_protocol=config, supervision=contact_labels)
        local_contact = protocol['local_contact_weight']*sum(config[name+'_weight']*value
            for name, value in contact.items())
        metrics.update(local_contact_active=True, local_contact_weight=protocol['local_contact_weight'],
            local_contact_loss=float(local_contact.detach()), local_contact_source_cohort_counts=contact_labels['groups'],
            **{'local_contact_'+name: float(value.detach()) for name, value in contact.items()})
    else:
        metrics.update(local_contact_active=False, local_contact_loss=0.)
    return terms, counts, metrics, local_contact


def combat_branches_backward(model, candidates, *, combat_branch_protocol, amp=None, check_gradients=True,
                             loss_groups=None, contact_protocol=None, coordinate_gradient_protocol=None,
                             kinematic_protocol=None, late_prefix_protocol=None):
    from .rssm_coordinate_gradients import normalize_coordinate_gradient_protocol, integrate_coordinate_gradients
    coordinate_gradient_protocol = normalize_coordinate_gradient_protocol(coordinate_gradient_protocol)
    protocol=normalize_combat_branch_protocol(combat_branch_protocol)
    if protocol is None or getattr(model.config,'navigation_only',True):
        raise ValueError('Combat branches require explicit opt-in and the actual one-enemy profile')
    chosen,selection=select_combat_branches(candidates,combat_branch_protocol=protocol,dt=1/model.config.simulation_hz,
        **({'late_prefix_protocol': late_prefix_protocol} if late_prefix_protocol is not None else {}))
    metrics={'loss':0.,'branches':len(chosen),**selection,'protocol_version':protocol['version'],
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
    results, own_previous = [], []
    with autocast():
        for index in range(protocol['horizon']):
            if protocol['version'] == 2: own_previous.append(state.detach())
            result=model.transition(state.detach(),actions[:,index],fmap,memory)
            if (result['state'].shape!=initial.shape or result['memory'].shape!=shape
                    or not bool(torch.isfinite(result['state']).all() and torch.isfinite(result['memory']).all())):
                raise FloatingPointError('Nonfinite own-memory combat branch; no repair')
            if first_result is None:first_result=result
            if protocol['version'] == 2: results.append(result)
            # Local Source heads retain the original one-step derivative;
            # only carried states and absolute trajectory errors integrate XY.
            integrated = integrate_coordinate_gradients(state, result, coordinate_gradient_protocol)
            state,memory=integrated['state'],result['memory'];outputs.append(state)
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
        local_contact = prediction.sum()*0
        if protocol['version'] == 1:
            one={'states':target[:,:2].clone(),'target_states':target[:,1:2], 'actions':actions[:,:1],
                'events':torch.tensor(np.stack([r['events'][:1]for r in chosen]),device=initial.device),
                'hit_ids':torch.tensor(np.stack([r['hits'][:1]for r in chosen]),device=initial.device),
                'attack_flags':torch.tensor(np.stack([r['attacks'][:1]for r in chosen]),device=initial.device)}
            one['states'][:,0]=initial
            supervision=prepare_combat_supervision(one,context=1)
            combat,counts=combat_supervised_terms(first_result,initial,target[:,1],actions[:,0],supervision=supervision,index=0)
        else:
            combat, counts, direct_metrics, local_contact = _direct_head_terms(results, own_previous, actions, grid,
                protocol, 1/model.config.simulation_hz, contact_protocol=contact_protocol,
                kinematic_protocol=kinematic_protocol)
            metrics.update(direct_metrics)
        anchor=sum(combat.values())
        objective=protocol['xy_weight']*xy+protocol['state_weight']*state_term+protocol['wall_weight']*wall+protocol['anchor_combat_weight']*anchor
        if protocol['version'] == 2: objective = objective+local_contact
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
            if protocol['version'] == 2:
                groups['player'] = groups['player']+protocol['weight']*local_contact
    if not bool(torch.isfinite(weighted)):raise FloatingPointError('Nonfinite own-combat branch loss')
    if protocol['version'] == 2:
        spawn_part = protocol['weight']*protocol['anchor_combat_weight']*combat['combat_spawn']
        metrics.update(component_spawn_raw=float(spawn_part.detach()),
            component_recovery_raw=float((weighted-spawn_part).detach()))
    if loss_groups is None:
        actual = weighted
    elif protocol['version'] == 1:
        actual = loss_groups.combine(groups)
    else:
        recovery = {**groups, 'enemy': groups['enemy']-spawn_part}
        actual = loss_groups.combine(recovery, component='recovery')
        actual = actual+loss_groups.combine({'enemy': spawn_part}, component='spawn')
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
        anchor_source_counts={name:float(value)for name,value in counts.items()
                              if not name.startswith('combat_kinematic_')})
    if loss_groups is not None:metrics['balanced_loss']=float(actual.detach())
    for offset in protocol['offsets']:
        error=((prediction[:,offset-1,8:10].detach()-target[:,offset,8:10])*16).norm(dim=-1)
        eligible=mask[:,offset-1];metrics[f'npc_xy_offset{offset}_count']=int(eligible.sum())
        metrics[f'npc_xy_offset{offset}_tiles_mean']=float(error[eligible].mean()) if eligible.any() else None
        metrics[f'player_xy_offset{offset}_tiles_mean']=float(((prediction[:,offset-1,:2].detach()-target[:,offset,:2])*16).norm(dim=-1).mean())
    return metrics
