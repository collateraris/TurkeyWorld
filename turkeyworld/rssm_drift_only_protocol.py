"""Temporary coordinate-only training with preserved full Adam and old ledgers."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from .rssm_replay_curriculum import EMA_KEY, ReplayComponentLossBalancer


DRIFT_ONLY_PROTOCOL = 'rssm_coordinate_drift_only_v1'
ALLOWED_PREFIXES = (
    'vjepa_backend.core.', 'vjepa_backend.obs_encoder.',
    'vjepa_backend.condition.', 'vjepa_backend.map_projection.',
    'vjepa_backend.spatial_encoder.', 'vjepa_backend.rssm_readout.',
    'vjepa_backend.readout_neck.', 'vjepa_backend.player_pass_head.',
    'vjepa_backend.enemy_rssm.core.', 'vjepa_backend.enemy_rssm.conditioner.',
    'vjepa_backend.enemy_rssm.observation_projection.',
    'vjepa_backend.enemy_rssm.readout.', 'vjepa_backend.enemy_rssm.motion_head.',
)
SETTINGS = {'version': 1, 'objective_protocol': DRIFT_ONLY_PROTOCOL,
    'player_xy_weight': 1., 'enemy_xy_weight': 1., 'coordinate_units': 'map_tiles',
    'normalization': 'none_drift_only_fixed_weights', 'kl_floor': 0.,
    'non_coordinate_heads_frozen': True, 'full_optimizer_parameter_names_preserved': True,
    'ema_values_frozen': True, 'legacy_ema_ledger_clocks_only': True,
    'auxiliary_source_queries': 0, 'index_reservation_only': True,
    'microbatch_size': 64, 'gradient_accumulation_steps': 4,
    'effective_batch_size': 256, 'validation_horizon': 450,
    'source_future_feedback': False, 'optimizer_reset': False,
    'allowed_parameter_prefixes': list(ALLOWED_PREFIXES)}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode('utf-8')).hexdigest()


def frozen_balance_values(state):
    """Exclude advancing clocks and the canonical short/long selection alias."""
    horizon = state[EMA_KEY]
    return {'settings': deepcopy(state['settings']),
        'component_settings': deepcopy(state['components']['settings']),
        'component_priorities': deepcopy(state['components']['priorities']),
        'navigation_actor_ema': deepcopy(state['banks']['navigation']['ema']),
        'navigation_component_ema': deepcopy(state['components']['banks']['navigation']['ema']),
        'horizon_banks': {name: {key: deepcopy(bank[key]) for key in ('actor_ema','component_ema')}
                          for name, bank in horizon['banks'].items()}}


def configure_drift_only_training(model, *, optimizer_parameter_names=None):
    """Freeze parameters, with grad=None so Adam momentum and decay also stop."""
    from .rssm_migration import configure_rssm_training
    configure_rssm_training(model)
    if not getattr(model.config, 'enemy_rssm_enabled', False):
        raise ValueError('Coordinate-only continuation requires the existing dual RSSMs')
    available = dict(model.named_parameters())
    canonical = list(optimizer_parameter_names) if optimizer_parameter_names is not None else [
        name for name, parameter in available.items() if parameter.requires_grad]
    if len(canonical) != len(set(canonical)) or any(name not in available for name in canonical):
        raise ValueError('Coordinate-only policy must preserve the complete canonical Adam names')
    allowed = [name for name in canonical if name.startswith(ALLOWED_PREFIXES)]
    if not all(any(name.startswith(prefix) for name in allowed)
               for prefix in ('vjepa_backend.core.','vjepa_backend.enemy_rssm.core.')):
        raise ValueError('Both existing RSSM cores must remain trainable for coordinate loss')
    selected = set(allowed)
    for name, parameter in available.items():
        parameter.requires_grad_(name in selected)
        if name not in selected: parameter.grad = None
    return {'objective_protocol': DRIFT_ONLY_PROTOCOL, 'trainable_parameter_names': allowed,
        'frozen_optimizer_parameter_names': [name for name in canonical if name not in selected],
        'canonical_optimizer_parameter_names': canonical,
        'trainable_parameters': sum(available[name].numel() for name in allowed),
        'optimizer_parameter_count': len(canonical), 'non_coordinate_heads_frozen': True,
        'optimizer_rebuilt': False, 'inference_forward_changed': False}


def advance_frozen_balance_clock(balancer, *, step):
    """Advance compatibility counters once; perform no numerical EMA update."""
    state = balancer.state
    ReplayComponentLossBalancer(state)
    if type(step) is not int or step != state['last_optimizer_step']+1:
        raise ValueError('Frozen EMA ledger advances only after one corresponding Adam update')
    before = frozen_balance_values(state)
    components = state['components']; horizon = state[EMA_KEY]
    state['committed_updates'] += 1; state['last_optimizer_step'] = step
    state['banks']['combat']['updates'] += 1
    components['committed_updates'] += 1; components['last_optimizer_step'] = step
    components['banks']['combat']['updates'] += 1
    bank = horizon['banks'][horizon['active']]
    bank['updates'] += 1; bank['last_optimizer_step'] = step
    if before != frozen_balance_values(state):
        raise AssertionError('Coordinate-only commit changed a frozen EMA value')


def summarize_drift_groups(reports, *, step):
    if not reports:
        raise ValueError('Cannot report an incomplete coordinate-only update')
    groups = {}
    for actor in ('player','enemy','interaction'):
        values = [report[actor]['raw'] for report in reports]
        if any(type(value) not in (int,float) or not math.isfinite(value) or value < 0 for value in values):
            raise FloatingPointError('Nonfinite coordinate-only accumulated objective')
        if any(report[actor]['scale'] != 1 or report[actor]['weight'] != 1
               or report[actor]['kl_floor'] != 0 or report[actor]['weighted'] != report[actor]['raw']
               for report in reports):
            raise ValueError('Coordinate-only weights or disabled KL floor changed within four micros')
        value = sum(values)/len(values)
        groups[actor] = {'raw':value, 'normalized':value, 'weighted':value,
            'scale':1., 'weight':1., 'effective_weight':1., 'kl_floor':0., 'active':value>0}
    return {**groups, 'total':sum(value['weighted'] for value in groups.values()),
        'kl_floor':0., 'normalization':SETTINGS['normalization'], 'loss_protocol':DRIFT_ONLY_PROTOCOL,
        'microbatches':len(reports), 'step':step, 'environment':'combat', 'ema_learning':False}


def drift_only_plan(state, *, source_checkpoint, source_checkpoint_sha256,
                    base_plan_sha256, optimizer_parameter_names):
    q = state['rssm_training']
    return {'version': 1, 'objective_protocol': DRIFT_ONLY_PROTOCOL, 'settings': deepcopy(SETTINGS),
        'source_checkpoint': str(source_checkpoint), 'source_checkpoint_sha256': source_checkpoint_sha256,
        'start_optimizer_step': state['global_training_step'], 'base_plan_sha256': base_plan_sha256,
        'coordinate_plan_sha256': q['coordinate_curriculum']['sha256'],
        'replay_plan_sha256': q['replay_curriculum']['sha256'],
        'optimizer_parameter_names': list(optimizer_parameter_names),
        'frozen_ema_values_sha256': _digest(frozen_balance_values(q['loss_balance_state']))}


def _boundary(state):
    from .rssm_replay_curriculum import _check_source_boundary
    q, step = _check_source_boundary(state)
    if q.get('replay_curriculum', {}).get('last_optimizer_step') != step:
        raise ValueError('Coordinate-only training requires a complete long-replay boundary')
    ReplayComponentLossBalancer(q['loss_balance_state'])
    return q, step


def bind_drift_only(state, plan, *, path, sha256, source_checkpoint_sha256,
                    loaded_checkpoint_sha256, base_plan_sha256, optimizer_parameter_names):
    q, step = _boundary(state)
    expected = drift_only_plan(state, source_checkpoint=plan.get('source_checkpoint',''),
        source_checkpoint_sha256=source_checkpoint_sha256, base_plan_sha256=base_plan_sha256,
        optimizer_parameter_names=optimizer_parameter_names)
    expected['start_optimizer_step'] = plan.get('start_optimizer_step')
    raw = Path(path).read_bytes()
    if (not sha256 or not source_checkpoint_sha256 or not base_plan_sha256 or not plan.get('source_checkpoint')
            or type(plan.get('start_optimizer_step')) is not int or not 0 <= plan['start_optimizer_step'] <= step
            or json.dumps(expected, sort_keys=True) != json.dumps(plan, sort_keys=True)
            or hashlib.sha256(raw).hexdigest() != sha256 or json.loads(raw.decode('utf-8')) != plan):
        raise ValueError('Coordinate-only objective, source, Adam names or frozen EMA identity changed')
    revision = q.get('drift_only_revision')
    if revision is None:
        if step != plan['start_optimizer_step'] or loaded_checkpoint_sha256 != source_checkpoint_sha256:
            raise ValueError('Initial coordinate-only phase requires its exact full saved Adam boundary')
        revision = {'version':1, 'objective_protocol':DRIFT_ONLY_PROTOCOL, 'plan':str(path), 'sha256':sha256,
            'start_optimizer_step':step, 'last_optimizer_step':step, 'updates':0,
            'initial_loaded_checkpoint_sha256':loaded_checkpoint_sha256,
            'frozen_ema_values_sha256':plan['frozen_ema_values_sha256'],
            'optimizer_parameter_names_sha256':_digest(plan['optimizer_parameter_names']),
            'sampler_draws_at_start':q['combat_recovery_revision']['sampler_state']['draws'],
            'used_aux_fragments':0, 'auxiliary_source_queries':0}
        q['drift_only_revision'] = revision
        state.setdefault('objective_history', []).append({'last_step':step, 'loss_protocol':state['loss_protocol'],
            'new_objective_protocol':DRIFT_ONLY_PROTOCOL, 'reason':'user_requested_temporarily_only_coordinate_drift',
            'optimizer_reset':False, 'ema_values_frozen':True, 'other_training_cases_used':False})
        state.update(loss=None, latest_metrics=None, loss_groups=None)
    else:
        count = step-plan['start_optimizer_step']
        if (revision.get('sha256') != sha256 or revision.get('start_optimizer_step') != plan['start_optimizer_step']
                or revision.get('last_optimizer_step') != step or revision.get('updates') != count
                or revision.get('initial_loaded_checkpoint_sha256') != source_checkpoint_sha256
                or revision.get('frozen_ema_values_sha256') != plan['frozen_ema_values_sha256']
                or revision.get('optimizer_parameter_names_sha256') != _digest(plan['optimizer_parameter_names'])
                or q['combat_recovery_revision']['sampler_state']['draws']-revision.get('sampler_draws_at_start',-1) != count*256
                or revision.get('used_aux_fragments') != 0 or revision.get('auxiliary_source_queries') != 0):
            raise ValueError('Saved coordinate-only phase lost its complete clock or reservation ledger')
    state['objective_protocol'] = DRIFT_ONLY_PROTOCOL
    state['training_protocol']['drift_only'] = {'plan_sha256':sha256, **deepcopy(SETTINGS)}
    state['training_protocol'].update(auxiliary_source_training_active=False,
        combat_head_training_active=False, posterior_training_active=False,
        memory_overshooting_active=False, kinematic_training_active=False,
        combat_branch_training_scope='xy_only', actual_auxiliary_fragments=0,
        index_reservation_only=True)
    q.update(objective_protocol=DRIFT_ONLY_PROTOCOL, auxiliary_source_queries=0, used_aux_fragments=0,
             index_reservation_only=True, ema_values_frozen=True,
             phase_label='RSSM: только дрейф координат Player/Enemy')
    return revision


def commit_drift_only_update(state, *, step, sampler_draws):
    q, actual = _boundary(state); revision = q['drift_only_revision']
    count = revision['updates']+1
    if (step != actual or step != revision['last_optimizer_step']+1
            or step-revision['start_optimizer_step'] != count
            or sampler_draws-revision['sampler_draws_at_start'] != count*256
            or _digest(frozen_balance_values(q['loss_balance_state'])) != revision['frozen_ema_values_sha256']):
        raise ValueError('Coordinate-only phase commits once with frozen EMA and exact index reservations')
    revision.update(updates=count, last_optimizer_step=step)
    return revision
