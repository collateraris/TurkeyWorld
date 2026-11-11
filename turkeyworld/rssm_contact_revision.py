"""Immutable opt-in contact loss/data revision at an existing Adam boundary."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .navigation_experiment import experiment_fingerprint
from .staged_curriculum import join_batches


COHORTS = ('free', 'blocked', 'slide', 'turn_idle')


def bind_contact_revision(state, path, *, base_plan_sha, step):
    path = Path(path).resolve()
    training = state['rssm_training']
    previous = training.get('contact_revision')
    if not path.exists():
        if previous:
            raise ValueError('Saved contact revision requires its immutable plan')
        return None
    plan = json.loads(path.read_text(encoding='utf-8'))
    cycle = plan.get('latent_training_cycle')
    if cycle is not None and (not isinstance(cycle, list) or not cycle
            or any(mode not in ('sample', 'argmax') for mode in cycle)):
        raise ValueError('Invalid RSSM latent training cycle')
    if (plan.get('version') != 2 or plan.get('base_plan_sha256') != base_plan_sha
            or plan.get('microbatch_size') != 16 or plan.get('gradient_accumulation_steps') != 1
            or plan.get('horizon') != 450 or plan.get('auxiliary_horizon') != 16
            or plan.get('new_auxiliary_examples') != 12 or plan.get('old_auxiliary_examples') != 4
            or plan.get('cohorts') != list(COHORTS)
            or plan.get('validation_every') != 100 or plan.get('preview_every') != 10
            or plan.get('loss_settings', {}).get('version') != 2
            or step < plan.get('start_optimizer_step', step+1)):
        raise ValueError('Unsupported or changed RSSM contact revision')
    root = Path(plan['dataset_directory']).resolve()
    manifest_path = root/'manifest.json'
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != plan['dataset_manifest_sha256']:
        raise ValueError('RSSM contact manifest changed')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('status') != 'complete' or not manifest.get('records'):
        raise ValueError('RSSM contact data is incomplete')
    for item in manifest['records']:
        source = (root/item['relative_path']).resolve()
        if root not in source.parents or hashlib.sha256(source.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('RSSM contact data is missing, modified or escaped its directory')
    listed = {item['relative_path'].replace('\\', '/') for item in manifest['records']}
    actual = {source.relative_to(root).as_posix() for source in root.rglob('*.npz')}
    if actual != listed:
        raise ValueError('RSSM contact directory contains unlisted or missing episodes')
    fingerprint = experiment_fingerprint(plan)
    if previous:
        if (previous.get('fingerprint') != fingerprint or previous.get('last_optimizer_step') != step
                or previous.get('new_auxiliary_draws') != previous['updates']*12
                or previous.get('old_auxiliary_draws') != previous['updates']*4
                or training['source_completed'] != previous['source_updates_at_start']+previous['updates']):
            raise ValueError('RSSM contact revision identity or successful-update clocks changed')
    else:
        if (step != plan['start_optimizer_step'] or training['phase'] != 'source'
                or training['distillation_completed'] != training['distillation_updates']):
            raise ValueError('Contact revision must begin at its saved Source/Adam boundary')
        previous = {**plan, 'fingerprint': fingerprint, 'status': 'training', 'updates': 0,
            'source_updates_at_start': training['source_completed'], 'new_auxiliary_draws': 0,
            'old_auxiliary_draws': 0, 'cohort_draws': {key: 0 for key in COHORTS},
            'last_optimizer_step': step, 'baseline_pending': True,
            'weights_preserved': True, 'optimizer_preserved': True, 'rng_preserved': True,
            'source_future_feedback': False, 'engine_corrections': 0}
        training['contact_revision'] = previous
        state.setdefault('objective_history', []).append({
            'loss_protocol': state.get('loss_protocol'), 'last_step': step,
            'note': 'Contact revision changes objective values; previous totals have a different definition'})
    state['loss_protocol'] = 'rssm_source_contact_balanced_v2'
    return previous


def commit_contact_revision(revision, *, step, selected):
    if step != revision['last_optimizer_step']+1 or sum(selected.values()) != 12:
        raise ValueError('Contact draws commit only after exactly one successful Adam update')
    revision['updates'] += 1
    revision['new_auxiliary_draws'] += 12
    revision['old_auxiliary_draws'] += 4
    revision['last_optimizer_step'] = step
    for key in COHORTS:
        revision['cohort_draws'][key] += selected.get(key, 0)


class ContactAuxiliarySampler:
    """Twelve cohort-balanced new fragments plus four existing fragments."""
    def __init__(self, new_pool, old_pool):
        self.new_pool, self.old_pool = new_pool, old_pool
        self.buckets = {key: [] for key in COHORTS}
        for index, episode in enumerate(new_pool.episodes):
            with np.load(episode['path'], allow_pickle=False) as saved:
                if 'contact_first16_cohort' in saved:
                    cohort = str(saved['contact_first16_cohort'])
                else:
                    labels = saved['axis_pass_targets'][:16]
                    active = saved['axis_command_active'][:16].astype(bool)
                    free = active & (labels >= .9)
                    blocked = active & (labels <= .1)
                    slide = (free.any(-1) & blocked.any(-1)).any()
                    cohort = 'slide' if slide else 'blocked' if blocked.any() else 'free' if active.any() else 'turn_idle'
            if cohort not in self.buckets:
                raise ValueError('Contact dataset contains an unsupported sampling cohort: '+cohort)
            self.buckets[cohort].append(index)
        if any(not values for values in self.buckets.values()):
            raise ValueError('Contact data must contain free, blocked, sliding and idle/turn fragments')

    def sample(self, revision, rng):
        selected = {key: 3 for key in COHORTS}
        indices = [int(rng.choice(self.buckets[key])) for key in COHORTS for _ in range(3)]
        # Keep the default Source spawn/action branches in replay, in addition
        # to ordinary old examples. The old recovery corpus puts them first.
        old = rng.integers(len(self.old_pool.episodes), size=4)
        old[0] = revision['updates'] % min(16, len(self.old_pool.episodes))
        new_pairs = np.column_stack([indices, np.zeros(12, dtype=np.int64)])
        old_pairs = np.column_stack([old, np.zeros(4, dtype=np.int64)])
        batches = [self.new_pool.batch(12, context=1, horizon=16, indices=new_pairs),
            self.old_pool.batch(4, context=1, horizon=16, indices=old_pairs)]
        return join_batches(batches), selected
