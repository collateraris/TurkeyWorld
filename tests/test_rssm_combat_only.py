"""Temporary combat-only scheduling preserves every prior optimizer clock."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from test_rssm_loss_balance_revision import fixture as balance_fixture
from turkeyworld.rssm_combat_protocol import commit_combat_update
from turkeyworld.rssm_loss_balance import bind_loss_balance_revision, GroupLossBalancer, LOSS_PROTOCOL
from turkeyworld.rssm_training_schedule import (
    COMBAT_ONLY_SCHEDULE, bind_combat_only_revision,
    next_training_environment, commit_combat_only_update,
)


BASE_CYCLE = ['combat', 'combat', 'combat', 'navigation']


def advance(state, balancer, environment='combat'):
    """Use the actual shared commit/EMA APIs in their trainer order."""
    training = state['rssm_training']
    step = state['global_training_step']+1
    accumulator = balancer.new_microbatch(environment)
    accumulator.combine({'player': torch.tensor(3.1), 'enemy': torch.tensor(5.1),
                         'interaction': torch.tensor(2.)}, floors={'player': 1.1, 'enemy': 1.1})
    report = balancer.summarize([accumulator.report()]*4, environment=environment, step=step)
    commit_combat_update(state, step=step, environment=environment, batch_size=128)
    for name in ('feedback_revision', 'combat_recovery_revision', 'batch_revision'):
        training[name]['last_optimizer_step'] = step
        training[name]['updates'] += 1
    balancer.commit(report, step=step)
    training['loss_balance_revision']['last_optimizer_step'] = step
    training['loss_balance_revision']['updates'] += 1
    training['loss_balance_state'] = balancer.state_dict()
    if training.get('combat_only_revision') is not None:
        commit_combat_only_update(state, step=step, environment=environment)


def fixture(folder):
    state, balance_plan, balance_kwargs, _, parent = balance_fixture(folder)
    _, balancer = bind_loss_balance_revision(state, balance_plan, **balance_kwargs)
    # Existing combat AND navigation EMA must survive the curriculum handoff.
    advance(state, balancer, 'combat')
    advance(state, balancer, 'navigation')
    state['training_protocol']['update_cycle'] = BASE_CYCLE.copy()
    state['rssm_training'].update(quality_streak=0, last_validation_passed=False,
        last_validation_report='heldout_both_tasks_failed.json')
    state['rssm_validation'] = {'step': state['global_training_step'], 'passed': False,
        'navigation': {'passed': False}, 'combat': {'passed': False}, 'source_future_feedback': False}
    plan = {'version': 1, 'training_schedule': COMBAT_ONLY_SCHEDULE,
        'update_cycle': ['combat'], 'navigation_replay_enabled': False,
        'navigation_validation_enabled': True, 'optimizer_reset': False,
        'source_future_feedback': False, 'base_plan_sha256': 'base',
        'loss_balance_plan_sha256': balance_kwargs['sha256'],
        'source_checkpoint_sha256': 'archived_boundary',
        'start_optimizer_step': state['global_training_step']}
    path = Path(folder)/'combat_only.json'
    path.write_text(json.dumps(plan, sort_keys=True), encoding='utf-8')
    kwargs = dict(path=path, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_checkpoint_sha256='archived_boundary', loaded_checkpoint_sha256='archived_boundary',
        base_plan_sha256='base', loss_balance_plan_sha256=balance_kwargs['sha256'])
    return state, plan, kwargs, balancer, parent, balance_kwargs['path']


class CombatOnlyScheduleTests(unittest.TestCase):
    def test_bind_preserves_weights_adam_rng_ema_old_cycle_counters_and_failed_validation(self):
        with TemporaryDirectory() as folder:
            state, plan, kwargs, _, parent, balance_path = fixture(folder)
            old = deepcopy(state)
            original_bytes = {path: path.read_bytes() for path in (parent, balance_path, kwargs['path'])}
            rng = torch.get_rng_state().clone()
            revision = bind_combat_only_revision(state, plan, **kwargs)
            self.assertEqual(state['loss_protocol'], LOSS_PROTOCOL)
            self.assertEqual(state['training_schedule'], COMBAT_ONLY_SCHEDULE)
            self.assertEqual(state['training_protocol']['update_cycle'], ['combat'])
            self.assertFalse(state['training_protocol']['navigation_replay_enabled'])
            self.assertTrue(state['training_protocol']['navigation_validation_enabled'])
            self.assertEqual(revision['updates'], 0)
            for key in ('global_training_step', 'current', 'completed_fragments', 'rng_state', 'rssm_validation'):
                self.assertEqual(state[key], old[key])
            for key, value in old['rssm_training'].items():
                self.assertEqual(state['rssm_training'][key], value, key)
            for name, tensor in old['model'].items():
                self.assertTrue(torch.equal(state['model'][name], tensor))
            for name, tensor in old['optimizer']['state'][0].items():
                self.assertTrue(torch.equal(state['optimizer']['state'][0][name], tensor))
            self.assertTrue(torch.equal(state['torch_rng_state'], old['torch_rng_state']))
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(state['curriculum_history'][-1]['previous_update_cycle'], BASE_CYCLE)
            self.assertEqual(state['rssm_training']['quality_streak'], 0)
            for path, data in original_bytes.items(): self.assertEqual(path.read_bytes(), data)
            count = len(state['curriculum_history'])
            self.assertIs(bind_combat_only_revision(state, plan, **kwargs), revision)
            self.assertEqual(len(state['curriculum_history']), count)

    def test_all_four_previous_cycle_phases_now_select_combat_without_resetting_context_clock(self):
        for completed in range(608, 612):
            state = {'global_training_step': 4481, 'rssm_training': {'combat_completed': completed}}
            self.assertEqual(next_training_environment(state, BASE_CYCLE, completed), BASE_CYCLE[completed % 4])
            state['rssm_training']['combat_only_revision'] = {
                'training_schedule': COMBAT_ONLY_SCHEDULE, 'start_optimizer_step': 4481,
                'last_optimizer_step': 4481, 'updates': 0}
            old = deepcopy(state)
            self.assertEqual(next_training_environment(state, BASE_CYCLE, completed), 'combat')
            self.assertEqual(state, old)
            with self.assertRaises(ValueError):
                next_training_environment(state, BASE_CYCLE, completed+1)

    def test_multiple_actual_commits_and_resume_keep_navigation_counters_and_ema_bitexact(self):
        with TemporaryDirectory() as folder:
            state, plan, kwargs, balancer, _, _ = fixture(folder)
            old = deepcopy(state['rssm_training']); start = state['global_training_step']
            revision = bind_combat_only_revision(state, plan, **kwargs)
            for index in range(4):
                completed = state['rssm_training']['combat_completed']
                self.assertEqual(next_training_environment(state, BASE_CYCLE, completed), 'combat')
                advance(state, balancer)
                self.assertEqual(revision['updates'], index+1)
                self.assertEqual(revision['last_optimizer_step'], start+index+1)
                self.assertEqual(state['rssm_training']['navigation_replay_updates'], old['navigation_replay_updates'])
                self.assertEqual(state['rssm_training']['navigation_auxiliary_draws'], old['navigation_auxiliary_draws'])
                self.assertEqual(balancer.state_dict()['banks']['navigation'], old['loss_balance_state']['banks']['navigation'])
            training = state['rssm_training']
            self.assertEqual(training['combat_long_updates'], old['combat_long_updates']+4)
            self.assertEqual(training['combat_auxiliary_draws'], old['combat_auxiliary_draws']+4*128)
            self.assertEqual(training['combat_completed'], old['combat_completed']+4)
            self.assertEqual(training['loss_balance_state']['committed_updates'], old['loss_balance_state']['committed_updates']+4)
            # Serializing/restoring cannot replay the handoff or reset its counters.
            path = Path(folder)/'saved.pt'; torch.save(state, path)
            restored = torch.load(path, weights_only=False, map_location='cpu')
            before = deepcopy(restored['rssm_training'])
            self.assertEqual(bind_combat_only_revision(restored, plan, **{**kwargs,
                'loaded_checkpoint_sha256': 'later_complete_checkpoint'}), revision)
            self.assertEqual(restored['rssm_training'], before)
            resumed_balancer = GroupLossBalancer(restored['rssm_training']['loss_balance_state'])
            advance(restored, resumed_balancer)
            advance(state, balancer)
            self.assertEqual(restored['rssm_training'], state['rssm_training'])
            self.assertEqual(restored['global_training_step'], state['global_training_step'])

    def test_changed_identity_wrong_boundary_and_disabled_navigation_validation_fail_before_binding(self):
        with TemporaryDirectory() as folder:
            for key, value in (('version', True), ('update_cycle', BASE_CYCLE),
                ('navigation_replay_enabled', True), ('navigation_validation_enabled', False),
                ('optimizer_reset', True), ('source_future_feedback', True),
                ('base_plan_sha256', 'wrong'), ('loss_balance_plan_sha256', 'wrong'),
                ('source_checkpoint_sha256', 'wrong'), ('start_optimizer_step', 1)):
                state, plan, kwargs, _, _, _ = fixture(folder)
                changed = deepcopy(plan); changed[key] = value
                with self.assertRaises(ValueError): bind_combat_only_revision(state, changed, **kwargs)
                self.assertNotIn('combat_only_revision', state['rssm_training'])
            state, plan, kwargs, _, _, _ = fixture(folder)
            with self.assertRaises(ValueError):
                bind_combat_only_revision(state, plan, **{**kwargs, 'loaded_checkpoint_sha256': 'wrong_weights_same_step'})
            self.assertNotIn('combat_only_revision', state['rssm_training'])

    def test_missing_revision_does_not_silently_restore_navigation_and_bad_saved_counts_reject(self):
        with TemporaryDirectory() as folder:
            state, plan, kwargs, _, _, _ = fixture(folder)
            bind_combat_only_revision(state, plan, **kwargs)
            missing = deepcopy(state); missing['rssm_training'].pop('combat_only_revision')
            with self.assertRaises(ValueError):
                next_training_environment(missing, BASE_CYCLE, missing['rssm_training']['combat_completed'])
            for key in ('navigation_replay_updates', 'navigation_auxiliary_draws', 'combat_auxiliary_draws'):
                broken = deepcopy(state); broken['rssm_training'][key] += 1
                with self.assertRaises(ValueError): bind_combat_only_revision(broken, plan, **kwargs)
            changed = deepcopy(state); changed['rssm_training']['combat_only_revision']['sha256'] = 'changed_plan'
            with self.assertRaises(ValueError): bind_combat_only_revision(changed, plan, **kwargs)

    def test_wrong_environment_or_duplicate_commit_is_rejected_without_advancing_ledger(self):
        with TemporaryDirectory() as folder:
            state, plan, kwargs, balancer, _, _ = fixture(folder)
            bind_combat_only_revision(state, plan, **kwargs)
            old = deepcopy(state['rssm_training']['combat_only_revision'])
            with self.assertRaises(ValueError):
                commit_combat_only_update(state, step=state['global_training_step']+1, environment='combat')
            with self.assertRaises(ValueError):
                commit_combat_only_update(state, step=state['global_training_step'], environment='navigation')
            self.assertEqual(state['rssm_training']['combat_only_revision'], old)
            advance(state, balancer)
            after = deepcopy(state['rssm_training']['combat_only_revision'])
            with self.assertRaises(ValueError):
                commit_combat_only_update(state, step=state['global_training_step'], environment='combat')
            self.assertEqual(state['rssm_training']['combat_only_revision'], after)


if __name__ == '__main__': unittest.main()
