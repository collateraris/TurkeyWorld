"""Real native32 RF3/RF5 validation, truthful supports and isolated state."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import test_renderer_rf5_bundle as bundle_support
from turkeyworld.engine import Game
from turkeyworld.renderer_rf5_phase import configure_rf5_training
from turkeyworld.renderer_rf5_validation import evaluate_rf5, save_rf5_preview
from turkeyworld.renderer_student import _hash_tensors


class TinyHeldOutSource:
    def __init__(self):
        self.split = 'val'
        self.rng = np.random.default_rng(77)
        self.draws = 0
        self.calls = []
        self.rows = [{'category': name} for name in
                     ('large', 'partial', 'distant', 'empty_navigation', 'empty_hidden')]
        parts = {name: [] for name in ('state', 'map', 'target', 'background', 'object_mask')}
        for index in range(5):
            game = Game(seed=1000 + index, endless=True, enemies_enabled=index != 3)
            if game.enemies:
                game.enemies[0].update(x=3.5 if index != 4 else 1.5, y=2.5)
            target = game.render(size=32, overlays=False)
            enemies = game.enemies
            game.enemies = []
            background = game.render(size=32, overlays=False)
            game.enemies = enemies
            parts['state'].append(game.state_vector())
            parts['map'].append(game.map.copy())
            parts['target'].append(target)
            parts['background'].append(background)
            parts['object_mask'].append(np.any(target != background, axis=-1))
        self.values = {}
        for name, values in parts.items():
            value = torch.from_numpy(np.stack(values))
            if name in ('target', 'background'):
                value = value.permute(0, 3, 1, 2).float() / 255
            elif name == 'object_mask':
                value = value[:, None].float()
            elif name == 'map':
                value = value.long()
            else:
                value = value.float()
            self.values[name] = value
        self.values.update(previous_state=self.values['state'].clone(),
            previous_rgb=self.values['target'].clone(), action=torch.zeros(5, 4),
            dynamic=torch.zeros(5, dtype=torch.bool),
            navigation_only=torch.tensor([False, False, False, True, False]))

    def batch(self, size, *, indices=None):
        if indices is None:
            raise AssertionError('Validation must use only explicit held-out indices')
        self.calls.append(list(indices))
        ids = torch.tensor(indices, dtype=torch.long)
        return {key: value[ids].clone() for key, value in self.values.items()}

    def panel_fingerprint(self, panel):
        return hashlib.sha256(json.dumps(list(panel)).encode()).hexdigest()


class RendererRF5ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.support = bundle_support.RendererRF5BundleTests
        cls.support.setUpClass()
        cls.source = TinyHeldOutSource()
        cls.model = deepcopy(cls.support.expanded)
        cls.model.flow_renderer.steps = 3
        configure_rf5_training(cls.model)
        cls.panel, cls.geometry_panel = [0, 1, 2, 3, 4], [0, 2, 3]
        cls.baseline, cls.arrays = evaluate_rf5(cls.model, cls.source, cls.panel,
            cls.geometry_panel, step=0, batch_size=2)

    @classmethod
    def tearDownClass(cls):
        cls.support.tearDownClass()

    def test_real_three_step_reports_keep_native_cohorts_and_never_relabel_reference_as_training_gain(self):
        report = self.baseline
        self.assertEqual(report['solver_steps'], 3)
        self.assertEqual(report['protocol'], 'renderer_rf_expanded_experts_v1')
        self.assertIn('rf3', report)
        self.assertNotIn('rf1', report)
        self.assertEqual(report['frames'], 5)
        self.assertEqual(report['rf3']['overall']['frames'], 5)
        self.assertGreater(report['rf3']['visible_enemy']['frames'], 0)
        self.assertEqual(report['rf3']['categories']['empty_navigation']['frames'], 1)
        self.assertEqual(report['rf3']['categories']['empty_hidden']['frames'], 1)
        self.assertIn('rf3', report['source_wall_validation'])
        self.assertIn('rf3', report['depth_banded_wall_validation'])
        self.assertIn('rf3', report['geometry_wall_guard_validation'])
        self.assertTrue(report['base_frozen'])
        self.assertFalse(report['base_loss_in_train'])
        self.assertFalse(report['gameplay_qualified'])
        self.assertFalse(report['automatic_active_promotion'])
        self.assertIn('RF3', report['thresholds']['applies_to'])

    def test_actual_objective_weights_the_unequal_tail_and_preserves_all_RNG_modes_weights_and_draws(self):
        source = TinyHeldOutSource()
        model = deepcopy(self.model)
        original_rng = (torch.get_rng_state().clone(), deepcopy(np.random.get_state()),
                        random.getstate(), deepcopy(source.rng.bit_generator.state))
        modes = {name: item.training for name, item in model.named_modules()}
        weights = _hash_tensors(model.state_dict())
        observed = []
        def actual_batch_objective(model, batch, **kwargs):
            size = len(batch['state'])
            observed.append((size, kwargs))
            return torch.tensor({2: 2., 1: 8.}[size]), {}
        with patch('turkeyworld.renderer_rf5_objectives.rf5_refinement_loss',
                   side_effect=actual_batch_objective):
            report, _ = evaluate_rf5(model, source, self.panel,
                                     self.geometry_panel, step=3, batch_size=2)
        self.assertEqual([size for size, _ in observed], [2, 2, 1])
        self.assertAlmostEqual(report['val_loss'], (2*2 + 2*2 + 1*8) / 5)
        self.assertTrue(all(args['kind'] == 'main' and args['checkpoint_velocity'] is False
                            for _, args in observed))
        self.assertNotEqual(report['val_loss'], report['rf3']['overall']['mae_mean'])
        self.assertEqual(_hash_tensors(model.state_dict()), weights)
        self.assertEqual({name: item.training for name, item in model.named_modules()}, modes)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertTrue(torch.equal(original_rng[0], torch.get_rng_state()))
        self.assertTrue(np.array_equal(original_rng[1][1], np.random.get_state()[1]))
        self.assertEqual(original_rng[2], random.getstate())
        self.assertEqual(original_rng[3], source.rng.bit_generator.state)
        self.assertEqual(source.draws, 0)
        json.dumps(report, allow_nan=False)

    def test_same_weights_are_not_improvement_and_foreign_panel_or_solver_baselines_reject(self):
        report, _ = evaluate_rf5(self.model, self.source, self.panel,
            self.geometry_panel, step=1, batch_size=2, baseline=self.baseline)
        self.assertFalse(report['experimental_selection']['validated_improvement'])
        self.assertIn('npc_foreground_not_improved',
                      report['experimental_selection']['failure_reasons'])
        self.assertIn('RF3', report['experimental_selection']['reference'])
        self.assertEqual(report['base'], self.baseline['base'])
        for key, value in [('panel_fingerprint', 'foreign'),
                           ('geometry_panel_fingerprint', 'foreign'), ('solver_steps', 5)]:
            bad = deepcopy(self.baseline)
            bad[key] = value
            with self.subTest(field=key), self.assertRaises(ValueError):
                evaluate_rf5(self.model, self.source, self.panel,
                    self.geometry_panel, step=1, batch_size=2, baseline=bad)

    def test_parent_RF1_is_explicit_separate_reference_and_legacy_five_step_report_still_supported(self):
        parent = {'summary': {'ssim_mean': .7}, 'rf1': {'visible_enemy': {'foreground_mae': .1}},
                  'panel_fingerprint': self.source.panel_fingerprint(self.panel)}
        model = deepcopy(self.model)
        model.flow_renderer.steps = 5
        report, _ = evaluate_rf5(model, self.source, self.panel,
            self.geometry_panel, step=2, batch_size=2, parent_rf1=parent)
        self.assertIn('rf5', report)
        self.assertNotIn('rf3', report)
        self.assertEqual(report['solver_steps'], 5)
        self.assertEqual(report['parent_rf1_reference']['solver_steps'], 1)
        self.assertIn('not an SGD improvement', report['parent_rf1_reference']['note'])
        bad = deepcopy(parent)
        bad['panel_fingerprint'] = 'foreign'
        with self.assertRaises(ValueError):
            evaluate_rf5(model, self.source, self.panel, self.geometry_panel,
                         step=2, parent_rf1=bad)

    def test_four_row_preview_is_native_Source_frozenBase_actual_RF_and_codec(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'preview.png'
            save_rf5_preview(self.arrays, path, 3, solver_steps=3)
            with Image.open(path) as image:
                self.assertEqual(image.size, (768, 868))
        self.assertEqual(self.arrays['source'].shape, (5, 3, 32, 32))
        self.assertEqual(self.arrays['rgb'].shape, (5, 3, 32, 32))


if __name__ == '__main__':
    unittest.main()
