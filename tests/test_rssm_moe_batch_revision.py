"""A real batch change must not rewrite optimizer, gate or old sample history."""
from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from turkeyworld.rssm_combat_memory_protocol import file_sha
from turkeyworld.rssm_moe_budget import short_cap_plan, bind_budget_policy
from turkeyworld.rssm_moe_protocol import (
    moe_training_plan, bind_moe_training, commit_moe_update, moe_training_horizon,
)
from turkeyworld.rssm_moe_batch_revision import (
    PROTOCOL, SETTINGS, batch_revision_plan, bind_moe_batch_revision,
    moe_batch_settings, microbatch_layout, commit_moe_batch_revision,
)


class MoEBatchRevisionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'source.pt'
        self.source.write_bytes(b'full-complete-source-boundary')
        self.state = {'global_training_step': 5352, 'rssm_training': {'old': {'ema': [2., 7.]}}}
        self.moe_plan = moe_training_plan(self.state, source_checkpoint=self.source,
            source_checkpoint_sha256=file_sha(self.source), architecture_migration_sha256='a' * 64)
        self.moe_path = self.write('moe.json', self.moe_plan)
        bind_moe_training(self.state, self.moe_plan, path=self.moe_path,
            sha256=file_sha(self.moe_path), loaded_checkpoint_sha256=file_sha(self.source))
        plan = short_cap_plan(self.state, source_checkpoint=self.source, cap=20)
        path = self.write('cap20.json', plan)
        self.budget_plan, self.budget_path = plan, path
        bind_budget_policy(self.state, plan, path=path, sha256=file_sha(path),
                           loaded_checkpoint_sha256=file_sha(self.source))
        for _ in range(23):
            self.update(.8)
        for _ in range(4):
            self.update(.1)
        self.state['rssm_training']['navigation_replay_revision'] = {
            'updates': 7, 'navigation_main_actors': 280, 'enemy_motion_main_actors': 448,
            'combat_main_actors': 1064, 'discarded_combat_rng_draws': 728,
            'sampler_state': {'draws': 280, 'rng': np.random.default_rng(17).bit_generator.state},
            'enemy_sampler_state': {'draws': 448, 'rng': np.random.default_rng(18).bit_generator.state},
        }
        self.plan = batch_revision_plan(self.state, source_checkpoint=self.source)
        self.path = self.write('micro16.json', self.plan)

    def write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value), encoding='utf-8')
        return path

    @property
    def gate(self):
        return self.state['rssm_training']['moe_training']

    def bind(self, state=None, loaded=None):
        return bind_moe_batch_revision(self.state if state is None else state, self.plan,
            path=self.path, sha256=file_sha(self.path),
            loaded_checkpoint_sha256=loaded or file_sha(self.source))

    def update(self, loss=.8, **kwargs):
        horizon = moe_training_horizon(self.state)
        self.state['global_training_step'] += 1
        if self.gate.get('batch_revision'):
            kwargs = {'microbatch_size': 16, 'effective_batch_size': 64, **kwargs}
        return commit_moe_update(self.state, step=self.state['global_training_step'],
                                 horizon=horizon, train_loss=loss, **kwargs)

    def test_activation_preserves_every_old_gate_field_history_streak_and_private_pcg(self):
        before = deepcopy(self.state)
        python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        record = self.bind()
        old_gate = deepcopy(self.gate)
        del old_gate['batch_revision']; del old_gate['batch_protocol']
        self.assertEqual(old_gate, before['rssm_training']['moe_training'])
        self.assertEqual((self.gate['current_horizon'], self.gate['stage_updates'],
                          self.gate['consecutive_below_threshold']), (8, 7, 4))
        self.assertEqual(self.state['rssm_training']['navigation_replay_revision'],
                         before['rssm_training']['navigation_replay_revision'])
        self.assertEqual(record['source_navigation_replay_revision'],
                         before['rssm_training']['navigation_replay_revision'])
        self.assertEqual(record['source_gate'], before['rssm_training']['moe_training'])
        self.assertEqual(record['updates'], 0)
        self.assertEqual(random.getstate(), python_rng)
        self.assertTrue(np.array_equal(np.random.get_state()[1], numpy_rng[1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))
        self.assertEqual(self.state['rssm_training']['old'], before['rssm_training']['old'])
        self.assertEqual(self.state['training_protocol']['moe_training'], before['training_protocol']['moe_training'])

    def test_exact_rotation_and_weights_preserve_effective38_16_10_without_enemy_dilution(self):
        self.bind()
        layouts = [microbatch_layout(self.state, i) for i in range(4)]
        self.assertEqual([r['combat'] for r in layouts], [9, 10, 9, 10])
        self.assertEqual([r['enemy_motion'] for r in layouts], [4] * 4)
        self.assertEqual([r['navigation'] for r in layouts], [3, 2, 3, 2])
        self.assertEqual([r['discarded_combat_rng_draws'] for r in layouts], [7, 6, 7, 6])
        for key, total in [('combat', 38), ('enemy_motion', 16), ('navigation', 10)]:
            self.assertEqual(sum(row[key] for row in layouts), total)
        for key, value in [('combat_player_weight', 38 / 64), ('navigation_player_router_weight', 10 / 64),
                           ('combat_enemy_weight', 38 / 54), ('enemy_motion_enemy_weight', 16 / 54)]:
            self.assertAlmostEqual(sum(row[key] for row in layouts) / 4, value)
        settings = moe_batch_settings(self.state)
        self.assertEqual((settings['microbatch_size'], settings['gradient_accumulation_steps'],
                          settings['effective_batch_size'], settings['auxiliary_batch_size'],
                          settings['late_own_prefix_actors']), (16, 4, 64, 16, 2))
        self.assertEqual(settings['auxiliary_sampling_policy']['four_micro_shooting_counts'],
                         dict(near=4, mid=4, far=4, miss=2, legacy=2))
        for invalid in (-1, 4, True, .0):
            with self.assertRaises(ValueError): microbatch_layout(self.state, invalid)

    def test_one_complete_update_advances_old_four_streak_and_accounts_piecewise_horizons(self):
        before = deepcopy(self.gate)
        record = self.bind()
        self.update(.1)
        self.assertEqual((self.gate['current_horizon'], self.gate['consecutive_below_threshold']), (16, 0))
        self.assertEqual(self.gate['advance_history'][:len(before['advance_history'])], before['advance_history'])
        self.assertEqual(self.gate['main_examples'], before['main_examples'] + 64)
        self.assertEqual(self.gate['main_transitions'], before['main_transitions'] + 8 * 64)
        self.update(); self.update()
        self.assertEqual(record['updates'], 3)
        self.assertEqual(record['main_examples'], 192)
        self.assertEqual(record['main_transitions'], (8 + 16 + 16) * 64)
        self.assertEqual(self.gate['main_examples'], before['main_examples'] + 192)
        self.assertEqual(self.gate['main_transitions'], before['main_transitions'] + (8 + 16 + 16) * 64)
        self.assertEqual(record['horizon_update_counts']['8'], 1)
        self.assertEqual(record['horizon_update_counts']['16'], 2)
        self.assertEqual(record['source_gate'], before)
        self.assertEqual(moe_training_horizon(self.state), 16)

    def test_exact_resume_old_moe_bind_then_new_revision_no_second_activation(self):
        self.bind(); self.update(.1); self.update()
        restored = json.loads(json.dumps(self.state))
        before = deepcopy(restored)
        bind_moe_training(restored, self.moe_plan, path=self.moe_path,
            sha256=file_sha(self.moe_path), loaded_checkpoint_sha256='f' * 64)
        self.bind(restored, loaded='f' * 64)
        bind_budget_policy(restored, self.budget_plan, path=self.budget_path,
            sha256=file_sha(self.budget_path), loaded_checkpoint_sha256='f' * 64)
        self.assertEqual(restored, before)
        self.update(); expected = deepcopy(self.state)
        self.state = restored; self.update()
        self.assertEqual(self.state, expected)

    def test_twenty_update_cap_counts_seven_old_and_thirteen_new_without_stage_reset(self):
        before = deepcopy(self.gate)
        self.bind()
        for _ in range(12): self.update(.8)
        self.assertEqual((self.gate['current_horizon'], self.gate['stage_updates']), (8, 19))
        self.update(.8)
        self.assertEqual((self.gate['current_horizon'], self.gate['stage_updates']), (16, 0))
        entry = self.gate['advance_history'][-1]
        self.assertEqual(entry['advance_reason'], 'stage_budget')
        self.assertEqual(entry['completed_updates_at_previous_stage'], 20)
        self.assertEqual(self.gate['horizon_update_counts']['8'], 20)
        self.assertEqual(self.gate['batch_revision']['horizon_update_counts']['8'], 13)
        self.assertEqual(self.gate['main_examples'], before['main_examples'] + 13 * 64)
        self.assertFalse(self.gate['goal_met'])

    def test_old_and_new_batch_dimensions_cannot_be_silently_interchanged(self):
        before = deepcopy(self.gate)
        with self.assertRaises(ValueError): self.update(microbatch_size=16, effective_batch_size=64)
        self.assertEqual(self.gate, before)
        self.state['global_training_step'] -= 1
        self.bind(); before = deepcopy(self.gate)
        with self.assertRaises(ValueError): self.update(microbatch_size=64, effective_batch_size=256)
        self.assertEqual(self.gate, before)
        self.state['global_training_step'] -= 1
        self.update()
        with self.assertRaises(ValueError):
            commit_moe_batch_revision(self.state, step=self.state['global_training_step'], horizon=8)

    def test_missing_revision_or_changed_history_counters_settings_and_source_pcg_rejected(self):
        self.bind(); self.update()
        for mutate in (
            lambda g: g.pop('batch_revision'),
            lambda g: g['batch_revision']['settings'].update(microbatch_size=64),
            lambda g: g['batch_revision'].update(updates=2),
            lambda g: g['batch_revision'].update(main_examples=256),
            lambda g: g['batch_revision']['source_gate'].update(main_examples=0),
            lambda g: g['batch_revision']['source_navigation_replay_revision']['sampler_state'].update(draws=0),
            lambda g: g.update(main_examples=g['updates'] * 256),
        ):
            bad = deepcopy(self.state); mutate(bad['rssm_training']['moe_training'])
            with self.assertRaises(ValueError): moe_training_horizon(bad)

    def test_plan_and_activation_pin_exact_checkpoint_gate_and_old_private_samplers(self):
        for mutate in (
            lambda s: s['rssm_training']['navigation_replay_revision']['enemy_sampler_state'].update(draws=0),
            lambda s: s['rssm_training']['moe_training'].update(last_train_loss=.2),
        ):
            bad = deepcopy(self.state); mutate(bad)
            with self.assertRaises(ValueError): self.bind(bad)
        with self.assertRaises(ValueError): self.bind(loaded='f' * 64)
        bad_plan = deepcopy(self.plan); bad_plan['settings']['auxiliary_batch_size'] = 64
        bad_path = self.write('tampered.json', bad_plan)
        with self.assertRaises(ValueError):
            bind_moe_batch_revision(self.state, bad_plan, path=bad_path, sha256=file_sha(bad_path),
                                    loaded_checkpoint_sha256=file_sha(self.source))
        self.source.write_bytes(b'changed-full-source')
        with self.assertRaises(ValueError): self.bind()

    def test_no_tensor_optimizer_or_rng_state_is_touched_by_plan_or_binding(self):
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        model(torch.ones(2, 2)).square().mean().backward(); optimizer.step()
        weights = deepcopy(model.state_dict()); moments = deepcopy(optimizer.state_dict())
        self.state['model'] = model.state_dict(); self.state['optimizer'] = optimizer.state_dict()
        self.bind()
        for name, value in weights.items(): self.assertTrue(torch.equal(value, model.state_dict()[name]))
        for index, old in moments['state'].items():
            for key, value in old.items():
                self.assertTrue(torch.equal(value, optimizer.state_dict()['state'][index][key]))
        self.assertEqual(moments['param_groups'], optimizer.state_dict()['param_groups'])


if __name__ == '__main__':
    unittest.main()
