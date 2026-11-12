import unittest

import torch

from turkeyworld.rssm_gradient_calibration import ComponentGradientMeter, reprioritize_calibration
from turkeyworld.rssm_component_balance import SETTINGS
from turkeyworld.rssm_loss_balance import GroupLossAccumulator


class ComponentGradientMeterTests(unittest.TestCase):
    def test_real_autograd_vectors_reconstruct_base_and_never_update_parameters_or_grad(self):
        player = torch.nn.Parameter(torch.tensor([1., 2.]))
        enemy = torch.nn.Parameter(torch.tensor([3., 4.]))
        names = [('vjepa_backend.core.test', player), ('vjepa_backend.enemy_rssm.core.test', enemy)]
        original = [value.clone() for _, value in names]
        coefficients = {'player': .5, 'enemy': .25, 'interaction': 1.}
        meter = ComponentGradientMeter(names, coefficients)
        groups = GroupLossAccumulator({'player': 2., 'enemy': 4., 'interaction': 1.}, gradient_meter=meter)
        base = player.sum()+2*enemy.sum()
        first = groups.combine({'player': base})
        spawn = enemy.square().sum()
        second = groups.combine({'enemy': spawn}, component='spawn')
        recovery = (player+enemy).square().sum()
        third = groups.combine({'interaction': recovery}, component='recovery')
        self.assertIsNone(player.grad); self.assertIsNone(enemy.grad)
        (first+second+third).backward()
        result = meter.finish()
        self.assertAlmostEqual(result['base_gradient_norm'], (2*.5**2+2*1.**2)**.5, places=6)
        row = result['components']['spawn']['enemy']
        self.assertEqual(row['raw'], 25.)
        self.assertAlmostEqual(row['raw_gradient_norm'], 10.)
        self.assertEqual(row['route_gradient_norms']['player_core'], 0.)
        self.assertEqual(row['route_gradient_norms']['enemy_core'], 10.)
        self.assertGreaterEqual(row['chosen_priority'], SETTINGS['minimum_priority'])
        self.assertLessEqual(row['chosen_priority'], 1.)
        for (_, value), before in zip(names, original):
            torch.testing.assert_close(value, before, rtol=0, atol=0)

    def test_gradients_add_across_independent_graphs_and_bad_routes_fail(self):
        value = torch.nn.Parameter(torch.tensor([2.]))
        meter = ComponentGradientMeter([('vjepa_backend.enemy_rssm.core.x', value)],
            {'player': 1., 'enemy': 1., 'interaction': 1.})
        loss = 3*value.square().sum()
        meter.capture('spawn', 'enemy', value.square().sum())
        meter.capture('spawn', 'enemy', 2*value.square().sum())
        (loss+value.sum()).backward()
        report = meter.finish()
        self.assertEqual(report['components']['spawn']['enemy']['raw_gradient_norm'], 12.)
        self.assertEqual(report['base_gradient_norm'], 1.)
        with self.assertRaises(ValueError): meter.capture('spawn', 'player', value.sum())

    def test_cpu_repricing_preserves_measurements_and_detects_old_floor_budget_violation(self):
        original = {'base_gradient_norm': 1., 'source_step': 4944, 'source_checkpoint_sha256': 'same',
            'components': {'spawn': {'enemy': {'raw': 1., 'raw_gradient_norm': 100.}},
                'recovery': {name: {'raw': 1., 'raw_gradient_norm': 100.}
                    for name in ('player', 'enemy', 'interaction')}}}
        updated = reprioritize_calibration(original)
        self.assertTrue(updated['passed_priority_budget_check'])
        self.assertEqual(updated['source_checkpoint_sha256'], 'same')
        for component, actors in original['components'].items():
            for name, row in actors.items():
                self.assertEqual(updated['components'][component][name]['raw'], row['raw'])
                self.assertEqual(updated['components'][component][name]['raw_gradient_norm'], row['raw_gradient_norm'])
        old_settings = {**SETTINGS, 'minimum_priority': .05}
        self.assertFalse(reprioritize_calibration(original, settings=old_settings)['passed_priority_budget_check'])


if __name__ == '__main__': unittest.main()
