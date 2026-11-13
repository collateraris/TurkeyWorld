from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import torch

from turkeyworld.rssm_combat_memory_protocol import file_sha
from turkeyworld.rssm_moe_budget import (BUDGET_PROTOCOL, FLAT_CAP_PROTOCOL,
    budget_plan, flat_cap_plan, bind_budget_policy, curriculum_completion)
from turkeyworld.rssm_moe_protocol import (MOE_HORIZONS, moe_training_plan,
    bind_moe_training, commit_moe_update, moe_training_horizon)


class MoEFlatCapTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'source.pt'
        self.source.write_bytes(b'complete-original-optimizer-boundary')
        self.state = {'global_training_step': 5352, 'rssm_training': {}}
        plan = moe_training_plan(self.state, source_checkpoint=self.source,
            source_checkpoint_sha256=file_sha(self.source), architecture_migration_sha256='a'*64)
        path = self.root / 'moe.json'
        path.write_text(json.dumps(plan), encoding='utf-8')
        bind_moe_training(self.state, plan, path=path, sha256=file_sha(path),
            loaded_checkpoint_sha256=file_sha(self.source))

    @property
    def gate(self):
        return self.state['rssm_training']['moe_training']

    def update(self, loss=1.):
        horizon = moe_training_horizon(self.state)
        self.state['global_training_step'] += 1
        return commit_moe_update(self.state, step=self.state['global_training_step'],
            horizon=horizon, train_loss=loss)

    def activate(self, cap=50, legacy=False):
        source = self.root / ('legacy_source.pt' if legacy else 'flat_source.pt')
        source.write_bytes(f'complete-boundary-{self.state["global_training_step"]}'.encode())
        plan = (budget_plan(self.state, source_checkpoint=source) if legacy else
                flat_cap_plan(self.state, source_checkpoint=source, cap=cap))
        path = self.root / ('budget_v2.json' if legacy else 'budget_v3.json')
        path.write_text(json.dumps(plan), encoding='utf-8')
        bind_budget_policy(self.state, plan, path=path, sha256=file_sha(path),
            loaded_checkpoint_sha256=file_sha(source))
        return plan, path

    def bind(self, plan, path, state=None):
        return bind_budget_policy(state if state is not None else self.state, plan,
            path=path, sha256=file_sha(path), loaded_checkpoint_sha256='b'*64)

    def test_fiftieth_commit_advances_and_equal_threshold_is_not_quality(self):
        self.activate()
        for _ in range(49):
            self.update(.25)
        self.assertEqual((self.gate['current_horizon'], self.gate['stage_updates']), (4, 49))
        self.assertEqual(self.gate['consecutive_below_threshold'], 0)
        self.update(.25)
        self.assertEqual((self.gate['current_horizon'], self.gate['stage_updates']), (8, 0))
        self.assertEqual(self.gate['last_advance_reason'], 'stage_budget')
        self.assertEqual(self.gate['advance_history'][-1]['completed_updates_at_previous_stage'], 50)
        self.assertFalse(self.gate['goal_met'])
        self.assertFalse(self.gate['budget_exhausted'])

    def test_v2_overdue_source_advances_once_preserving_original_history_and_state(self):
        for _ in range(190):
            self.update()
        old_plan, old_path = self.activate(legacy=True)
        for _ in range(50):
            self.update()
        for _ in range(4):
            self.update(.2)
        self.assertEqual((self.gate['current_horizon'], self.gate['stage_updates']), (8, 54))
        # Representative external weights, optimizer moments and all RNGs are
        # independent of this metadata-only activation, not reconstructed here.
        linear = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(linear.parameters(), lr=1e-4)
        linear(torch.ones(1, 2)).sum().backward(); optimizer.step()
        weights = {k: v.clone() for k, v in linear.state_dict().items()}
        moments = deepcopy(optimizer.state_dict())
        self.state['rssm_training']['historical_ledger'] = {'last_step': 4482, 'updates': 152}
        self.state['private_sampler'] = {'draws': 1234, 'rng': np.random.default_rng(173).bit_generator.state}
        previous = deepcopy(self.state)
        old_bytes = old_path.read_bytes()
        torch_rng, numpy_rng, python_rng = torch.get_rng_state(), np.random.get_state(), random.getstate()
        plan, path = self.activate()
        self.assertEqual(self.state['global_training_step'], previous['global_training_step'])
        self.assertEqual((self.gate['current_horizon'], self.gate['stage_updates'],
                          self.gate['consecutive_below_threshold']), (16, 0, 0))
        self.assertEqual(self.gate['horizon_update_counts']['4'], 190)
        self.assertEqual(self.gate['horizon_update_counts']['8'], 54)
        self.assertEqual(self.gate['updates'], 244)
        self.assertEqual(self.gate['advance_policy']['updates'], 0)
        self.assertEqual(self.gate['advance_policy']['source_gate'], previous['rssm_training']['moe_training'])
        self.assertEqual(self.gate['advance_history'][0], previous['rssm_training']['moe_training']['advance_history'][0])
        self.assertEqual(self.gate['advance_history'][-1]['stage_budget'], 50)
        self.assertTrue(self.gate['advance_history'][-1]['activation_boundary'])
        self.assertEqual(self.state['private_sampler'], previous['private_sampler'])
        self.assertEqual(self.state['rssm_training']['historical_ledger'], previous['rssm_training']['historical_ledger'])
        self.assertEqual(old_bytes, old_path.read_bytes())
        self.assertEqual(old_plan['kind'], BUDGET_PROTOCOL)
        for k, v in linear.state_dict().items(): self.assertTrue(torch.equal(v, weights[k]))
        after = optimizer.state_dict()
        self.assertEqual(after['param_groups'], moments['param_groups'])
        for i, values in after['state'].items():
            for k, v in values.items(): self.assertTrue(torch.equal(v, moments['state'][i][k]))
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))
        self.assertEqual(random.getstate(), python_rng)
        self.assertEqual(np.random.get_state()[0], numpy_rng[0])
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_rng[1])
        self.assertEqual(np.random.get_state()[2:], numpy_rng[2:])
        saved = deepcopy(self.state)
        self.bind(plan, path)
        self.assertEqual(self.state, saved)  # No repeated boundary advancement.

    def test_non_overdue_streak_and_partial_stage_survive_v2_to_v3_and_exact_resume(self):
        for _ in range(190): self.update()
        self.activate(legacy=True)
        for _ in range(20): self.update()
        for _ in range(4): self.update(.2)
        plan, path = self.activate()
        self.assertEqual((self.gate['stage_updates'], self.gate['consecutive_below_threshold']), (24, 4))
        for _ in range(5): self.update(.2)
        self.assertEqual((self.gate['current_horizon'], self.gate['consecutive_below_threshold']), (8, 9))
        restored = json.loads(json.dumps(self.state))
        self.bind(plan, path, state=restored)
        self.assertEqual(restored, self.state)
        self.update(.2)
        self.assertEqual((self.gate['current_horizon'], self.gate['stage_updates'], self.gate['consecutive_below_threshold']), (16, 0, 0))
        self.assertEqual(self.gate['last_advance_reason'], 'loss_streak')

    def test_final_fiftieth_commit_stops_without_quality_and_rejects_fifty_first(self):
        self.activate()
        for horizon in MOE_HORIZONS[:-1]:
            self.assertEqual(moe_training_horizon(self.state), horizon)
            for _ in range(50): self.update()
        for _ in range(49): self.update()
        self.assertIsNone(curriculum_completion(self.gate))
        self.update()
        self.assertEqual(curriculum_completion(self.gate), 'completed_budget')
        self.assertEqual((self.gate['current_horizon'], self.gate['stage_updates']), (450, 50))
        self.assertFalse(self.gate['goal_met'])
        self.assertTrue(self.gate['budget_exhausted'])
        self.assertEqual(self.gate['completion_reason'], 'stage_budget')
        before = deepcopy(self.gate)
        with self.assertRaises(ValueError):
            commit_moe_update(self.state, step=self.state['global_training_step']+1,
                              horizon=450, train_loss=1.)
        self.assertEqual(self.gate, before)
        # The trainer's pre-loop and post-commit guard saves without an SGD call.
        from turkeyworld.train_rssm_moe import RSSMMoETraining
        trainer = SimpleNamespace(moe=self.gate, state=self.state, save=Mock())
        self.assertTrue(RSSMMoETraining.finish_completed_curriculum(trainer, None, None, {}))
        self.assertEqual(self.state['status'], 'completed_budget')
        trainer.save.assert_called_once_with(None, None, {})

    def test_final_ten_low_updates_stop_early_with_loss_goal_not_budget(self):
        self.activate()
        for _ in range(30*10 + 9): self.update(.1)
        self.assertEqual((self.gate['current_horizon'], self.gate['consecutive_below_threshold']), (450, 9))
        self.assertIsNone(curriculum_completion(self.gate))
        self.update(.1)
        self.assertEqual(curriculum_completion(self.gate), 'completed_loss_gate')
        self.assertTrue(self.gate['goal_met'])
        self.assertFalse(self.gate['budget_exhausted'])
        self.assertEqual(self.gate['stage_updates'], 10)

    def test_cap100_supported_without_changing_active_cap50_defaults(self):
        self.activate(cap=100)
        for _ in range(99): self.update()
        self.assertEqual((self.gate['current_horizon'], self.gate['stage_updates']), (4, 99))
        self.update()
        self.assertEqual(self.gate['current_horizon'], 8)
        for cap in (0, 49, 51, True, 50.):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                flat_cap_plan(self.state, source_checkpoint=self.source, cap=cap)

    def test_parent_identity_history_formula_and_terminal_tampering_rejected(self):
        for _ in range(190): self.update()
        self.activate(legacy=True)
        for _ in range(54): self.update()
        plan, path = self.activate()
        original = deepcopy(self.state)
        changes = [
            lambda g: g['advance_history'][0].update(stage_budget=100),
            lambda g: g['advance_policy']['source_gate']['advance_history'][0].update(stage_budget=100),
            lambda g: g['advance_policy'].update(previous_policy_sha256='f'*64),
            lambda g: g.update(budget_exhausted=True),
            lambda g: g['advance_policy'].update(budget_base=49),
        ]
        for change in changes:
            self.state = deepcopy(original)
            change(self.gate)
            with self.assertRaises(ValueError): moe_training_horizon(self.state)
        self.state = deepcopy(original)
        self.gate['advance_policy']['budget_base'] = 100  # Recognized, but not the pinned plan.
        with self.assertRaises(ValueError): self.bind(plan, path)
        self.state = deepcopy(original)
        bad = deepcopy(plan); bad['source_policy_sha256'] = 'f'*64
        path.write_text(json.dumps(bad), encoding='utf-8')
        with self.assertRaises(ValueError): self.bind(bad, path)
        self.assertEqual(self.gate['advance_policy']['kind'], FLAT_CAP_PROTOCOL)


if __name__ == '__main__':
    unittest.main()
