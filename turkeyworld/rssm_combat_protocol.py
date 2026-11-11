"""Explicit navigation-to-combat continuation without changing learned tensors."""
from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import torch

from .models import NeuralWorldModel, WorldModelConfig
from .rssm_migration import configure_rssm_training
from .distill_renderer import _save
from .training_pipeline import now


PROTOCOL = 'rssm_one_enemy_combat_source_v1'


@contextmanager
def environment_profile(model, *, navigation_only):
    """Select the recorded initial environment, never a future-state correction."""
    if not isinstance(navigation_only, bool):
        raise TypeError('Environment navigation_only must be a boolean')
    objects = [model, model.vjepa_backend]
    previous = [(value, value.config) for value in objects]
    try:
        for value, config in previous:
            value.config = replace(config, navigation_only=navigation_only)
        yield
    finally:
        for value, config in previous:
            value.config = config


def continued_combat_state(previous, *, run_id, source_checkpoint, source_sha256):
    state = copy.deepcopy(previous)
    step = state['global_training_step']
    old = copy.deepcopy(state['rssm_training'])
    state.setdefault('objective_history', []).append({'loss_protocol': state.get('loss_protocol'),
        'last_step': step, 'reason': 'user_requested_one_enemy_movement_and_combat',
        'navigation_quality_passed': bool(old.get('last_validation_passed', False))})
    state['navigation_training_history'] = old
    state.update(run_id=run_id, status='preparing', stage='rssm_combat_dynamics',
        current=step, total=None, loss_protocol=PROTOCOL,
        quality_status='One-enemy movement/combat and navigation replay not qualified',
        rssm_validation=None, prediction_preview=None, latest_metrics=None,
        renderer_training={'active': False, 'adversarial_active': False, 'codec_frozen': True,
            'dynamics_active': True},
        combat_migration={'source_checkpoint': str(source_checkpoint), 'source_sha256': source_sha256,
            'source_step': step, 'reason': 'user_requested', 'weights_preserved': True,
            'optimizer_preserved': True, 'rng_preserved': True,
            'only_model_config_change': {'navigation_only': [True, False]}})
    # Old revision ledgers are immutable history, not active continuation clocks.
    for key in ('contact_revision', 'continuation', 'navigation_revision'):
        state['rssm_training'].pop(key, None)
    state['rssm_training'].update(phase='combat', phase_label='RSSM: movement, one enemy and combat',
        combat_start_step=step, combat_completed=0, combat_long_updates=0, navigation_replay_updates=0,
        combat_auxiliary_draws=0, navigation_auxiliary_draws=0,
        navigation_only=False, renderer_frozen=True, validation_every=250, preview_every=10,
        rollout_horizon=450, total_updates=None, source_updates=None,
        quality_streak=0, quality_required_passes=2, last_validation_step=None,
        last_validation_passed=False, last_validation_report=None)
    state['training_protocol'].update(validation_every=250, preview_every=10,
        navigation_only=False, update_cycle=['combat', 'combat', 'combat', 'navigation'],
        known_environment_profile_per_update=True, combat_losses_on_prior_before_source_rgb=True,
        source_posterior_is_separate_auxiliary_stream=True,
        own_memory_contact_branches_navigation_only=True)
    return state


def prepare_combat_continuation(source_path, destination, *, run_id):
    """Write a new full checkpoint; old checkpoint, Adam and RNG bytes stay intact."""
    source_path, destination = Path(source_path).resolve(), Path(destination).resolve()
    if destination.exists() or source_path == destination:
        raise FileExistsError('Combat continuation never overwrites an existing checkpoint')
    payload = torch.load(source_path, map_location='cpu', weights_only=False)
    config = dict(payload['config'])
    if (config.get('architecture') != 'dreamer_rssm' or config.get('navigation_only') is not True
            or payload['training_steps'] != payload['global_training_step']
            or payload['run_state']['global_training_step'] != payload['training_steps']):
        raise ValueError('Combat continuation requires a complete navigation RSSM Adam boundary')
    required = ('optimizer', 'optimizer_parameter_names', 'rng_state', 'torch_rng_state',
                'cuda_rng_state', 'rssm_migration')
    if any(key not in payload for key in required):
        raise ValueError('Full Adam/RNG/migration state is required for combat continuation')
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    target_config = {**config, 'navigation_only': False}
    with torch.random.fork_rng(devices=[]):
        model = NeuralWorldModel(WorldModelConfig(**target_config))
    model.load_state_dict(payload['model'], strict=True)
    configure_rssm_training(model)
    names = [name for name, value in model.named_parameters() if value.requires_grad]
    if names != payload['optimizer_parameter_names']:
        raise ValueError('Semantic combat switch changed Adam parameter identities')
    # Test strict state compatibility without replacing/cloning any source tensor.
    if any(not torch.equal(value, model.state_dict()[key]) for key, value in payload['model'].items()):
        raise AssertionError('Combat config switch changed learned world tensors')
    state = continued_combat_state(payload['run_state'], run_id=run_id,
        source_checkpoint=source_path, source_sha256=digest)
    result = {**payload, 'config': target_config, 'run_state': state,
        'rssm_training': state['rssm_training'], 'rssm_combat_migration': state['combat_migration']}
    _save(destination, result)
    return {**state['combat_migration'], 'destination': str(destination),
        'model_tensor_count': len(payload['model']), 'optimizer_parameter_count': len(names),
        'optimizer_state_count': len(payload['optimizer']['state']),
        'all_model_tensors_bitexact': True, 'all_existing_adam_states_preserved': True,
        'all_rng_fields_preserved': True, 'navigation_history_preserved': True,
        'prepared_at': now()}


def commit_combat_update(state, *, step, environment, batch_size):
    from .rssm_batch_revision import batch_commit_size
    training = state['rssm_training']
    expected = training['combat_start_step']+training['combat_completed']+1
    if (step != expected or environment not in ('combat', 'navigation') or batch_size != batch_commit_size(state)
            or state['global_training_step'] != step-1):
        raise ValueError('Combat clocks commit only after one successful preserved-Adam update')
    state['global_training_step'] = state['current'] = step
    training['combat_completed'] += 1
    training['source_completed'] += 1
    training['auxiliary_draws'] += batch_size
    training['combat_long_updates' if environment == 'combat' else 'navigation_replay_updates'] += 1
    training['combat_auxiliary_draws' if environment == 'combat' else 'navigation_auxiliary_draws'] += batch_size


class CombatTrainingSampler:
    """Target-window cohorts include player death, not only NPC death."""
    def __init__(self, data, *, context=1, horizon=16, cache_path=None):
        import numpy as np
        from .combat_sampling import CombatSampling
        self.base = CombatSampling(data, context=context, cache_path=cache_path)
        self.horizon = horizon
        self.groups = dict(self.base._index(horizon))
        extra = {'player_death': [], 'player_hit': []}
        for episode, info in enumerate(self.base.records):
            with np.load(info['path'], allow_pickle=False) as saved:
                states = saved['states']; events = saved['events']
            masks = {'player_death': (states[:-1, 4] > 0) & (states[1:, 4] <= 0),
                     'player_hit': events[:, 3] > .5}
            for start in range(info['frames']-context-horizon+1):
                first = start+context-1
                for name, mask in masks.items():
                    if mask[first:first+horizon].any(): extra[name].append((episode, start))
        self.groups.update({name: np.asarray(pairs, np.int64).reshape(-1, 2) for name, pairs in extra.items()})
        active = [name for name, pairs in self.groups.items() if name != 'uniform' and len(pairs)]
        self.weights = {'uniform': .25, **{name: .75/len(active) for name in active}} if active else {'uniform': 1.}
        self.last_sample = None

    def sample(self, count, rng):
        import numpy as np
        names = list(self.weights)
        labels = rng.choice(len(names), size=count, p=list(self.weights.values()))
        pairs = np.empty((count, 2), np.int64)
        selected = {}
        for index, name in enumerate(names):
            rows = np.flatnonzero(labels == index)
            if len(rows):
                group = self.groups[name]
                pairs[rows] = group[rng.integers(len(group), size=len(rows))]
                selected[name] = len(rows)
        self.last_sample = {'selected_classes': selected, **self.base.sample_report(pairs, self.horizon)}
        return pairs

    def summary(self):
        return {**self.base.summary(), 'weights': self.weights,
            'eligible_windows': {name: len(pairs) for name, pairs in self.groups.items()},
            'player_death_is_a_distinct_target_window_cohort': True}
