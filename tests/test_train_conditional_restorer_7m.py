"""7M continuation preserves both mixed-age Adam groups and the SSIM loop."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from test_train_conditional_restorer_2m import (
    trained_parent, LoopHarness, parent_validation)
from turkeyworld.conditional_restorer_expansion import expand_conditional_restorer
from turkeyworld.conditional_restorer_7m import expand_conditional_restorer_7m
from turkeyworld.renderer_native_phase import _same
from turkeyworld.train_conditional_restorer import ConditionalRestorerTraining
from turkeyworld.train_conditional_restorer_2m import adopt_restorer_optimizer
from turkeyworld.train_conditional_restorer_7m import (
    PARAMETERS, TENSORS, Restorer7MTraining, adopt_restorer_7m_optimizer)


def trained_2m_parent():
    old, first, inputs = trained_parent()
    previous, _ = expand_conditional_restorer(old)
    optimizer = adopt_restorer_optimizer(previous, first, lr=.0001)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        previous(*inputs).square().mean().backward()
        optimizer.step()
    parent = dict(step=2, total_restorer_updates=4,
        phase=dict(parent_restorer_updates=2),
        optimizer=deepcopy(optimizer.state_dict()),
        optimizer_parameter_names=['flow_renderer.conditional_restorer.' + name
                                   for name, _ in previous.named_parameters()])
    return previous, parent, inputs


class SevenLoopHarness(Restorer7MTraining):
    prepare = LoopHarness.prepare
    update = LoopHarness.update
    payload = LoopHarness.payload
    save = LoopHarness.save
    export = LoopHarness.export
    preview = LoopHarness.preview

    def __init__(self, root, *, start=1999):
        super().__init__(SimpleNamespace(run_dir=Path(root), steps=None, probe=False, device='cpu'))
        self.start = start
        self.scores = {2000: .85, 2100: .850001}
        self.updates, self.previews, self.exports, self.saves = [], [], [], []


class Restorer7MDriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_actual130_old_mixed_age_moments_options_preserved_and65_new_fresh(self):
        previous, parent, inputs = trained_2m_parent()
        expanded, _ = expand_conditional_restorer_7m(previous, seed=175)
        original = deepcopy(parent)
        old_weights = deepcopy(previous.state_dict())
        rng = torch.get_rng_state().clone()
        optimizer = adopt_restorer_7m_optimizer(expanded, parent, lr=.0001)
        saved = optimizer.state_dict()
        self.assertEqual(sum(p.numel() for p in expanded.parameters()), PARAMETERS)
        self.assertEqual(len(list(expanded.parameters())), TENSORS)
        self.assertEqual((PARAMETERS, TENSORS), (7000313, 195))
        self.assertEqual(len(saved['param_groups']), 3)
        self.assertTrue(_same(saved['param_groups'][:2], original['optimizer']['param_groups']))
        self.assertTrue(_same(saved['state'], original['optimizer']['state']))
        self.assertEqual(saved['param_groups'][2]['params'], list(range(130, 195)))
        self.assertTrue(set(saved['param_groups'][2]['params']).isdisjoint(saved['state']))
        self.assertEqual(saved['param_groups'][2]['lr'], .0001)
        self.assertTrue(_same(previous.state_dict(), old_weights))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        optimizer.zero_grad(set_to_none=True)
        expanded(*inputs).square().mean().backward()
        optimizer.step()
        after = optimizer.state_dict()
        self.assertEqual(len(after['state']), 195)
        self.assertEqual({float(after['state'][i]['step']) for i in range(65)}, {5.})
        self.assertEqual({float(after['state'][i]['step']) for i in range(65, 130)}, {3.})
        self.assertEqual({float(after['state'][i]['step']) for i in range(130, 195)}, {1.})
        self.assertTrue(all(torch.isfinite(value).all() for state in after['state'].values()
                            for value in state.values() if torch.is_tensor(value)))
        self.assertGreater(float(expanded.expansion.output.weight.grad.abs().sum()), 0)
        self.assertTrue(_same(parent, original), 'Inherited optimizer moments must not alias parent payload RAM')

    def test_ordered_names_cumulative_clock_mixed_ages_and_finite_moments_fail_closed(self):
        previous, parent, _ = trained_2m_parent()
        expanded, _ = expand_conditional_restorer_7m(previous)
        changes = [lambda x: x['optimizer_parameter_names'].reverse(),
                   lambda x: x.__setitem__('total_restorer_updates', x['step']),
                   lambda x: x['optimizer']['param_groups'][1]['params'].reverse(),
                   lambda x: x['optimizer']['state'][0]['step'].fill_(2),
                   lambda x: x['optimizer']['state'][65]['step'].fill_(4),
                   lambda x: x['optimizer']['state'][0]['exp_avg'].fill_(float('nan')),
                   lambda x: x['optimizer']['state'][129].__setitem__('exp_avg_sq', torch.zeros(1))]
        for index, change in enumerate(changes):
            bad = deepcopy(parent); change(bad)
            with self.subTest(index=index), self.assertRaises(ValueError):
                adopt_restorer_7m_optimizer(expanded, bad, lr=.0001)
        for invalid in (0., -.1, float('nan'), True):
            with self.subTest(lr=invalid), self.assertRaises(ValueError):
                adopt_restorer_7m_optimizer(expanded, parent, lr=invalid)

    def test_real7m_loop_continues_past2000_until_first_full_strict_RF_goal(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = SevenLoopHarness(directory)
            with patch.object(ConditionalRestorerTraining, 'validate', parent_validation):
                trainer.run()
            self.assertEqual(trainer.updates, list(range(2000, 2101)))
            self.assertFalse(trainer.goal_history[1]['goal_met'])
            self.assertTrue(trainer.goal_history[2]['goal_met'])
            self.assertEqual(trainer.state['status'], 'completed')
            self.assertIsNone(trainer.state['total'])
            self.assertIn('~7M', trainer.state['model'])
            self.assertEqual(trainer.best['step'], 2100)
            receipt = json.loads((Path(directory)/'goal_reached.json').read_text(encoding='utf-8'))
            self.assertTrue(receipt['decision']['goal_met'])

    def test_7m_user_stop_is_not_goal_or_budget_achievement(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/'stop_training').write_text('explicit user stop', encoding='ascii')
            trainer = SevenLoopHarness(directory, start=2200)
            trainer.run()
            self.assertEqual(trainer.state['status'], 'stopped')
            self.assertFalse(trainer.goal_met)
            self.assertEqual(trainer.updates, [])
            self.assertFalse((Path(directory)/'goal_reached.json').exists())


if __name__ == '__main__':
    unittest.main()
