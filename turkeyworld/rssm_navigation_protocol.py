"""Opt-in navigation revision identity and explicit corner/single-wall replay.

Old training, contact and continuation plans remain immutable. This plan binds
the new data/objective/schedule at an existing successful Adam boundary.
"""
from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path

import numpy as np

from .navigation_experiment import experiment_fingerprint
from .rssm_contact_revision import COHORTS
from .rssm_continuation import file_sha256, verified_episode_paths
from .staged_curriculum import join_batches
from .training_pipeline import atomic_json


AXIS_OUTCOMES = ('free_free', 'free_blocked', 'blocked_free', 'blocked_blocked')


def verify_restored_boundary(model, optimizer, rng, metadata):
    """Measure the real restored GPU state before sampling or the next update."""
    import torch
    compared = {'tensors': 0}
    def same(old, current):
        if torch.is_tensor(old):
            compared['tensors'] += 1
            return (torch.is_tensor(current) and old.dtype == current.dtype
                and old.shape == current.shape
                and torch.equal(old.detach().cpu(), current.detach().cpu()))
        if isinstance(old, np.ndarray):
            return isinstance(current, np.ndarray) and np.array_equal(old, current)
        if isinstance(old, dict):
            return (isinstance(current, dict) and old.keys() == current.keys()
                    and all(same(value, current[key]) for key, value in old.items()))
        if isinstance(old, (tuple, list)):
            return (type(old) == type(current) and len(old) == len(current)
                    and all(same(a, b) for a, b in zip(old, current)))
        return type(old) == type(current) and old == current
    checks = {'model': same(metadata['model'], model.state_dict()),
        'optimizer': same(metadata['optimizer'], optimizer.state_dict()),
        'rng_state': same(metadata['rng_state'], rng.bit_generator.state),
        'torch_rng_state': same(metadata['torch_rng_state'], torch.get_rng_state())}
    if next(model.parameters()).device.type == 'cuda':
        checks['cuda_rng_state'] = same(metadata['cuda_rng_state'], torch.cuda.get_rng_state_all())
    if not all(checks.values()):
        raise ValueError('Navigation resume did not preserve the saved state: '+str(checks))
    steps = sorted({int(value['step']) for value in optimizer.state.values() if 'step' in value})
    return {'passed': True, 'step': metadata['training_steps'],
        'checks_bitexact': checks, 'compared': compared,
        'optimizer_parameter_states': len(optimizer.state), 'optimizer_steps': steps}


def bind_navigation_revision(state, path, *, dataset_root, base_plan_sha,
                             contact_plan_path, continuation_plan_path,
                             checkpoint_path, step, branch_protocol):
    training = state['rssm_training']
    saved = training.get('navigation_revision')
    path = Path(path)
    if dataset_root is None:
        if saved or path.exists():
            raise ValueError('Saved navigation revision requires --navigation-revision-root')
        return None
    from .rssm_navigation_revision import (new_revision_ledger, navigation_revision_setting,
                                            validate_revision_ledger)
    if not training.get('continuation') or training.get('phase') != 'source':
        raise ValueError('Navigation revision requires the restored Source continuation')
    paths, manifest_sha = verified_episode_paths(dataset_root)
    report = state.get('rssm_validation', {})
    if path.exists():
        plan = json.loads(path.read_text(encoding='utf-8'))
        expected_panels = plan['validation_fingerprints']
    else:
        expected_panels = {key: report.get(key, {}).get('fingerprint')
                           for key in ('long450', 'one_frame450')}
    if not all(isinstance(value, str) and value for value in expected_panels.values()):
        raise ValueError('Navigation revision requires both pinned held-out450 panels')
    settings = {'version': 1, 'base_plan_sha256': base_plan_sha,
        'contact_plan_sha256': file_sha256(contact_plan_path),
        'continuation_plan_sha256': file_sha256(continuation_plan_path),
        'dataset_directory': str(Path(dataset_root).resolve()),
        'dataset_manifest_sha256': manifest_sha,
        'microbatch_size': 16, 'gradient_accumulation_steps': 1,
        'main_horizon': 450, 'main_tbptt_chunk': 16,
        'main_contexts': [1, 2], 'main_window_start': 0,
        'source_auxiliary_horizon': 16,
        'corner_auxiliary_examples': 8, 'singlewall_auxiliary_examples': 4,
        'legacy_auxiliary_examples': 4,
        'contact_branch_protocol': copy.deepcopy(branch_protocol),
        'schedule': 'balanced_16_joint_pool_context_latent_v1',
        'schedule_cycle': [navigation_revision_setting(index) for index in range(16)],
        'validation_every': 250, 'preview_every': 10,
        'validation_fingerprints': expected_panels,
        'best_selection': 'both450_point_and_swept_walls_then_p95_then_mean',
        'source_future_feedback': False, 'renderer_frozen': True,
        'optimizer_and_rng_preserved': True}
    if path.exists():
        if any(plan.get(key) != value for key, value in settings.items()):
            raise ValueError('Navigation revision plan/data/objective identity changed')
    else:
        if saved:
            raise ValueError('Saved navigation revision requires its immutable plan')
        plan = {**settings, 'start_optimizer_step': step,
            'source_completed_at_start': training['source_completed'],
            'source_checkpoint_at_binding': str(Path(checkpoint_path).resolve()),
            'source_checkpoint_sha256': file_sha256(checkpoint_path)}
        atomic_json(path, plan)
    fingerprint = experiment_fingerprint(plan)
    if saved:
        updates = step-plan['start_optimizer_step']
        if (saved.get('fingerprint') != fingerprint
                or saved.get('revision_updates') != updates
                or saved.get('last_optimizer_step') != step
                or training['source_completed'] != plan['source_completed_at_start']+updates):
            raise ValueError('Navigation revision successful-update clocks changed')
        validate_revision_ledger(saved, step=step, source_completed=training['source_completed'])
    else:
        if (step != plan['start_optimizer_step']
                or training['source_completed'] != plan['source_completed_at_start']
                or file_sha256(checkpoint_path) != plan['source_checkpoint_sha256']):
            raise ValueError('Navigation revision boundary checkpoint changed')
        saved = {**plan, **new_revision_ledger(step, training['source_completed'], fingerprint),
            'fingerprint': fingerprint, 'baseline_pending': True,
            'weights_preserved': True, 'optimizer_preserved': True, 'rng_preserved': True}
        training['navigation_revision'] = saved
        state.setdefault('objective_history', []).append({
            'loss_protocol': state.get('loss_protocol'), 'last_step': step,
            'note': 'Added own-memory contact branches and corner replay; totals change definition'})
        state.setdefault('navigation_revision_history', []).append({
            'fingerprint': fingerprint, 'start_optimizer_step': step,
            'dataset_directory': plan['dataset_directory']})
        training['continuation'].update(consecutive_passes=0, status='training')
    state['loss_protocol'] = 'rssm_navigation_own_memory_contacts_v3'
    state['training_protocol'].update(
        navigation_revision_id=fingerprint, main_contexts=[1, 2],
        main_window_start=0, main_first_frame_and_action_included=True,
        auxiliary_corner_fragments=8, auxiliary_singlewall_fragments=4,
        auxiliary_legacy_fragments=4, own_memory_contact_branches=True,
        contact_branch_protocol=copy.deepcopy(branch_protocol),
        independent_joint_sampling=True, validation_every=250, preview_every=10)
    return saved, paths


class NavigationContactSampler:
    """Eight actual corner-axis outcomes, four wall cohorts, four legacy clips."""
    def __init__(self, new_pool, old_pool, root):
        self.new_pool, self.old_pool = new_pool, old_pool
        manifest = json.loads((Path(root)/'manifest.json').read_text(encoding='utf-8'))
        records = {Path(record['relative_path']).name: record for record in manifest['records']
                   if record.get('split') == 'train'}
        self.corner = {key: [] for key in AXIS_OUTCOMES}
        self.single = {key: [] for key in COHORTS}
        self.cohort = {}
        for index, episode in enumerate(new_pool.episodes):
            path = Path(episode['path'])
            record = records.get(path.name)
            if record is None:
                raise ValueError('Navigation sampler episode is not a pinned TRAIN member')
            with np.load(path, allow_pickle=False) as data:
                cohort = str(data['contact_first16_cohort'])
            if cohort not in COHORTS:
                raise ValueError('Navigation sampler has an unsupported Source cohort')
            self.cohort[index] = cohort
            if record.get('pool') == 'corner':
                outcomes = record.get('axis_outcomes_first16', [])
                if not isinstance(outcomes, list) or set(outcomes)-set(AXIS_OUTCOMES):
                    raise ValueError('Corner pool requires actual first16 axis outcomes')
                for outcome in set(outcomes):
                    self.corner[outcome].append(index)
            elif record.get('pool') == 'singlewall':
                self.single[cohort].append(index)
            else:
                raise ValueError('Navigation replay pool provenance is missing')
        if any(not values for values in (*self.corner.values(), *self.single.values())):
            raise ValueError('Navigation data must cover all four corner outcomes and wall cohorts')

    def sample(self, revision, rng):
        chosen = [int(rng.choice(self.corner[key])) for key in AXIS_OUTCOMES for _ in range(2)]
        chosen += [int(rng.choice(self.single[key])) for key in COHORTS]
        selected = dict(Counter(self.cohort[index] for index in chosen))
        legacy = rng.integers(len(self.old_pool.episodes), size=4)
        legacy[0] = revision['updates'] % min(16, len(self.old_pool.episodes))
        new_pairs = np.column_stack([chosen, np.zeros(12, dtype=np.int64)])
        old_pairs = np.column_stack([legacy, np.zeros(4, dtype=np.int64)])
        batch = join_batches([
            self.new_pool.batch(12, context=1, horizon=16, indices=new_pairs),
            self.old_pool.batch(4, context=1, horizon=16, indices=old_pairs)])
        return batch, selected
