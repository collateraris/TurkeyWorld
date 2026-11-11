from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from turkeyworld.engine import Game
from turkeyworld.rssm_combat_recovery import (RECOVERY_PROTOCOL,
    bind_recovery_revision, evaluate_melee_negatives, recovery_quality_gate)
from turkeyworld.rssm_spawn_curriculum import source_recovery_labels
from turkeyworld.staged_curriculum import StagedSequenceData
from turkeyworld.structured_state import structured_transition
from test_rssm_spawn_curriculum import make_source_files, KnownSpawnModel


def binding_fixture():
    state = {'global_training_step': 4130, 'completed_fragments': 66080,
        'loss_protocol': 'combat_numeric_feedback_v2', 'training_protocol': {},
        'rssm_training': {'combat_completed': 257, 'navigation_completed': 120,
            'quality_streak': 1, 'last_validation_passed': True,
            'last_validation_report': 'old/validation.json',
            'feedback_revision': {'numeric_state_feedback': 'detached'}},
        'model': {'weight': torch.tensor([1., 2.])},
        'optimizer': {'step': torch.tensor(4130.), 'exp_avg': torch.tensor([.3, .4])},
        'rng': np.random.default_rng(173).bit_generator.state}
    revision = {'version': 3, 'start_optimizer_step': 4130,
        'base_plan_sha256': 'base', 'data_manifest_sha256': 'data',
        'negative_manifest_sha256': 'negative', 'source_checkpoint_sha256': 'checkpoint',
        'numeric_state_feedback': 'detached', 'source_future_feedback': False,
        'sampler': {'batch': 16, 'spawn': 4, 'attack_negative': 4, 'ordinary': 8},
        'expanded_spawn_validation': 'all_long_val_events', 'validation_every': 250,
        'combat_branch_protocol': {}}
    kwargs = dict(sha256='revision-sha', path='recovery_revision.json',
        base_plan_sha256='base', data_sha256='data', negative_sha256='negative',
        source_checkpoint_sha256='checkpoint')
    return state, revision, kwargs


def gate_fixture():
    base = {'passed': True, 'failure_reasons': [], 'summary': {'navigation_xy': .04},
        'thresholds': {'navigation_xy_max': .1}}
    spawn = {'source_held_out': True, 'events': 128, 'source_episode_count': 64,
        'missing_anchor_classes': [], 'overall': {'metrics': {
            'spawn_decoded_top1_tile_accuracy': {'mean': .85, 'denominator': 128},
            'spawn_decoded_xy_error_tiles': {'mean': .2, 'denominator': 128}}}}
    melee = {'negative_transitions': 302, 'positive_transitions': 63,
        'negative_false_positive_rate': .05, 'positive_recall': .9}
    return base, spawn, melee


class AlwaysAttackModel(KnownSpawnModel):
    def __init__(self, attack):
        super().__init__()
        self.attack = attack

    def transition(self, state, action, fmap, memory):
        spawn = super().transition(state, action, fmap, memory)
        return {**spawn, **structured_transition(state, state, action,
            state.new_tensor([[-8., 8.]]).expand(len(state), -1),
            state.new_full((len(state), 1), 8. if self.attack else -8.),
            dt=1/30, endless=True, spawn_positions=spawn['spawn_positions'],
            spawn_valid=spawn['spawn_valid'])}


class CombatRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_binding_preserves_model_adam_rng_clocks_and_resume_keeps_new_progress(self):
        state, revision, kwargs = binding_fixture()
        before = deepcopy(state)
        rng = torch.get_rng_state().clone()
        ledger, branches = bind_recovery_revision(state, revision, **kwargs)
        self.assertEqual(state['loss_protocol'], RECOVERY_PROTOCOL)
        self.assertEqual(branches['offsets'], [1, 4, 8, 16])
        self.assertEqual(state['rssm_training']['quality_streak'], 0)
        self.assertFalse(state['rssm_training']['last_validation_passed'])
        self.assertEqual(state['objective_history'][0]['last_validation_report'], 'old/validation.json')
        self.assertTrue(state['objective_history'][0]['last_validation_passed'])
        for key in ('global_training_step', 'completed_fragments', 'rng'):
            self.assertEqual(state[key], before[key])
        for group in ('model', 'optimizer'):
            for name, value in before[group].items():
                torch.testing.assert_close(state[group][name], value, rtol=0, atol=0)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertEqual(state['rssm_training']['combat_completed'], 257)
        self.assertEqual(state['rssm_training']['navigation_completed'], 120)
        state['global_training_step'] += 2
        ledger.update(last_optimizer_step=4132, updates=2, combat_updates=1,
            sampler_state={'draws': 16}, baseline_complete=True)
        state['rssm_training'].update(quality_streak=1, last_validation_passed=True)
        expected = deepcopy(state)
        restored, _ = bind_recovery_revision(state, revision, **kwargs)
        self.assertIs(restored, ledger)
        self.assertEqual(len(state['objective_history']), 1)
        self.assertEqual(restored['updates'], 2)
        self.assertEqual(restored['sampler_state'], {'draws': 16})
        self.assertEqual(state['rssm_training'], expected['rssm_training'])

    def test_changed_identity_missing_numeric_fix_and_wrong_boundary_fail_before_mutation(self):
        changes = [('base_plan_sha256', 'bad'), ('data_manifest_sha256', 'bad'),
            ('negative_manifest_sha256', 'bad'), ('source_checkpoint_sha256', 'bad'),
            ('numeric_state_feedback', 'attached'), ('source_future_feedback', True),
            ('start_optimizer_step', 4129), ('validation_every', 100),
            ('combat_branch_protocol', None)]
        for key, value in changes:
            with self.subTest(key=key):
                state, revision, kwargs = binding_fixture()
                revision[key] = value
                with self.assertRaises(ValueError):
                    bind_recovery_revision(state, revision, **kwargs)
                self.assertNotIn('combat_recovery_revision', state['rssm_training'])
                self.assertNotIn('objective_history', state)
                self.assertEqual(state['global_training_step'], 4130)
        state, revision, kwargs = binding_fixture()
        state['rssm_training'].pop('feedback_revision')
        with self.assertRaises(ValueError):
            bind_recovery_revision(state, revision, **kwargs)
        state, revision, kwargs = binding_fixture()
        bind_recovery_revision(state, revision, **kwargs)
        with self.assertRaises(ValueError):
            bind_recovery_revision(state, revision, **{**kwargs, 'sha256': 'changed'})
        state['global_training_step'] += 1
        with self.assertRaises(ValueError):
            bind_recovery_revision(state, revision, **kwargs)

    def test_gate_retains_navigation_failures_replaces_only_short_spawn_and_checks_full_coverage(self):
        base, spawn, melee = gate_fixture()
        original = deepcopy((base, spawn, melee))
        self.assertTrue(recovery_quality_gate(base, spawn, melee)['passed'])
        self.assertEqual((base, spawn, melee), original)
        base['failure_reasons'] = ['navigation_xy', 'spawn_decoded_top1_tile_accuracy']
        base['passed'] = False
        result = recovery_quality_gate(base, spawn, melee)
        self.assertEqual(result['failure_reasons'], ['navigation_xy'])
        self.assertFalse(result['passed'])
        for metric in spawn['overall']['metrics']:
            with self.subTest(metric=metric):
                b, s, m = gate_fixture()
                s['overall']['metrics'][metric]['denominator'] = 127
                self.assertFalse(recovery_quality_gate(b, s, m)['passed'])
        for field, value in [('missing_anchor_classes', [7]), ('source_episode_count', 63)]:
            b, s, m = gate_fixture(); s[field] = value
            self.assertFalse(recovery_quality_gate(b, s, m)['passed'])
        for field, value in [('negative_transitions', 63), ('positive_transitions', 7),
                ('negative_false_positive_rate', .101), ('positive_recall', .79),
                ('negative_false_positive_rate', None)]:
            b, s, m = gate_fixture(); m[field] = value
            self.assertFalse(recovery_quality_gate(b, s, m)['passed'])

    def test_all_real_source_negatives_and_positives_have_independent_denominators_without_engine(self):
        with TemporaryDirectory() as folder:
            paths = make_source_files(Path(folder)/'val', split='val')
            expected = {'negative': 0, 'positive': 0}
            for path in paths:
                with np.load(path) as saved:
                    labels = source_recovery_labels(saved['states'], saved['attack_flags'], float(saved['dt']))
                expected['negative'] += int(labels['attack_negative'].sum())
                expected['positive'] += int(labels['attack_positive'].sum())
            self.assertGreater(expected['negative'], 0)
            self.assertGreater(expected['positive'], 0)
            data = StagedSequenceData(paths[0].parent, 'cpu', mode='stream')
            results = []
            with patch.object(Game, 'step', side_effect=AssertionError('No runtime Source engine')), \
                    patch.object(Game, 'render', side_effect=AssertionError('No future RGB')):
                for attack in (False, True):
                    results.append(evaluate_melee_negatives(AlwaysAttackModel(attack), data,
                        Path(folder)/str(attack), batch_size=3))
            for report in results:
                self.assertEqual(report['negative_transitions'], expected['negative'])
                self.assertEqual(report['positive_transitions'], expected['positive'])
                self.assertEqual(report['runtime_engine_calls'], 0)
                self.assertEqual(report['future_rgb_inputs'], 0)
                self.assertTrue(report['source_held_out'])
                self.assertFalse(report['source_future_feedback'])
            self.assertEqual(results[0]['panel_fingerprint'], results[1]['panel_fingerprint'])
            self.assertEqual(results[0]['negative_false_positive_rate'], 0.)
            self.assertEqual(results[0]['positive_recall'], 0.)
            self.assertEqual(results[1]['negative_false_positive_rate'], 1.)
            self.assertEqual(results[1]['positive_recall'], 1.)
            data.split = 'train'
            with self.assertRaises(ValueError):
                evaluate_melee_negatives(AlwaysAttackModel(True), data, Path(folder)/'invalid')


if __name__ == '__main__':
    unittest.main()
