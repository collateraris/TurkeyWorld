"""Periodic validation preserves training state and reports honest counts."""
import copy
import random
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from turkeyworld.behavior_validation import behavior_summary, evaluate_behavior, behavior_validation_due


def report():
    rate = {'mean': .25, 'numerator': 8., 'denominator': 32}
    stat = {'mean': .5, 'p95': .9, 'n': 224, 'nonfinite': 0}
    return {'cohorts': {
        'uniform': {'windows': 128},
        'attack': {'binary': {'attack_accepted': {'source_positives': 32, 'recall': 0.,
                                                'precision': None, 'f1': 0.}}},
        'spawn': {'metrics': {'spawn_decoded_top1_tile_accuracy': rate,
                             'spawn_decoded_xy_error_tiles': {'mean': 7.8, 'numerator': 249.6, 'denominator': 32}}},
        'hit': {'metrics': {'hit_target_accuracy': {'mean': .625, 'numerator': 20, 'denominator': 32}}}},
        'heldout_trajectory': {'count': 32, 'windows_with_wall_collision': 10,
            'all_states_finite': True, 'all_memory_finite_and_shape_valid': True,
            'key_offsets': {'16': {'player_xy_error_tiles': stat, 'npc_xy_error_tiles': stat}}}}


class BehaviorValidationTests(unittest.TestCase):
    def test_stopped_boundary_is_checked_once_after_resume(self):
        self.assertFalse(behavior_validation_due(799, {'step': 725, 'loss_protocol': 'v5'}, 'v5'))
        self.assertTrue(behavior_validation_due(800, {'step': 725, 'loss_protocol': 'v5'}, 'v5'))
        self.assertFalse(behavior_validation_due(800, {'step': 800, 'loss_protocol': 'v5'}, 'v5'))
        self.assertFalse(behavior_validation_due(899, {'step': 800, 'loss_protocol': 'v5'}, 'v5'))
        self.assertTrue(behavior_validation_due(900, {'step': 800, 'loss_protocol': 'v5'}, 'v5'))
        self.assertTrue(behavior_validation_due(800, {'step': 800, 'loss_protocol': 'old'}, 'v5'))

    def test_zero_recall_and_missing_precision_keep_denominators(self):
        summary = behavior_summary(report(), step=800, protocol='v5')
        self.assertEqual(summary['attack_recall'], 0.)
        self.assertIsNone(summary['attack_precision'])
        self.assertEqual(summary['attack_positive_cases'], 32)
        self.assertEqual(summary['spawn_tile_accuracy']['denominator'], 32)
        self.assertEqual(summary['step'], 800)
        self.assertEqual(summary['loss_protocol'], 'v5')
        self.assertFalse(summary['future_source_feedback'])

    def test_training_modes_caches_and_all_rng_states_restore(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(.2)).train()
        model[0]._spawn_walkable = torch.ones(1, 16, 16, dtype=torch.bool)
        model[0]._map_condition = torch.ones(1, 16, 16, dtype=torch.long)
        data = SimpleNamespace(rng=np.random.default_rng(73))
        saved = (torch.random.get_rng_state().clone(), copy.deepcopy(np.random.get_state()),
                 random.getstate(), copy.deepcopy(data.rng.bit_generator.state))

        def mutate(*args, **kwargs):
            model.eval()
            model[0]._spawn_walkable = torch.zeros(4, 16, 16, dtype=torch.bool)
            model[0]._map_condition = torch.zeros(4, 16, 16, dtype=torch.long)
            torch.rand(5); np.random.random(5); random.random(); data.rng.random(5)
            return report()

        with tempfile.TemporaryDirectory() as directory, patch('turkeyworld.behavior_validation.evaluate_heads', mutate):
            summary = evaluate_behavior(model, data, directory, step=800, protocol='v5')
        self.assertEqual(summary['every'], 100)
        self.assertTrue(model.training and model[0].training and model[1].training)
        self.assertEqual(tuple(model[0]._map_condition.shape), (1, 16, 16))
        self.assertTrue(bool(model[0]._map_condition.all()))
        self.assertTrue(bool(model[0]._spawn_walkable.all()))
        self.assertTrue(torch.equal(saved[0], torch.random.get_rng_state()))
        self.assertEqual(saved[1][0], np.random.get_state()[0])
        np.testing.assert_array_equal(saved[1][1], np.random.get_state()[1])
        self.assertEqual(saved[2], random.getstate())
        self.assertEqual(saved[3], data.rng.bit_generator.state)

    def test_exception_also_restores_training_state(self):
        model = torch.nn.Linear(2, 2).train()
        with tempfile.TemporaryDirectory() as directory, patch('turkeyworld.behavior_validation.evaluate_heads', side_effect=RuntimeError('probe')):
            with self.assertRaisesRegex(RuntimeError, 'probe'):
                evaluate_behavior(model, SimpleNamespace(), directory, step=800, protocol='v5')
        self.assertTrue(model.training)


if __name__ == '__main__':
    unittest.main()
