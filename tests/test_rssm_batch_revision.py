"""Effective128 revision preserves history and requires explicit32x4 commits."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from turkeyworld.rssm_batch_revision import BATCH_PROTOCOL, bind_batch_revision
from turkeyworld.rssm_combat_protocol import commit_combat_update
from turkeyworld.rssm_spawn_curriculum import RecoveryCombatSampler
from test_rssm_spawn_curriculum import make_source_files
from types import SimpleNamespace


def fixture(folder):
    parent = Path(folder)/'v3.json'
    parent.write_text(json.dumps({'version': 3, 'base_plan_sha256': 'base',
        'feedback_revision_sha256': 'feedback'}), encoding='utf-8')
    parent_sha = hashlib.sha256(parent.read_bytes()).hexdigest()
    state = {'global_training_step': 4260, 'current': 4260, 'loss_protocol': 'recovery_v3',
        'completed_fragments': 68160, 'training_protocol': {'microbatch_size': 16,
            'gradient_accumulation_steps': 1, 'effective_batch_size': 16},
        'rssm_training': {'combat_start_step': 3873, 'combat_completed': 387,
            'source_completed': 4060, 'auxiliary_draws': 68160, 'combat_long_updates': 291,
            'navigation_replay_updates': 96, 'combat_auxiliary_draws': 4656,
            'navigation_auxiliary_draws': 1536, 'quality_streak': 0,
            'last_validation_report': 'old_4250.json', 'last_validation_passed': False,
            'feedback_revision': {'sha256': 'feedback', 'last_optimizer_step': 4260,
                'numeric_state_feedback': 'detached', 'updates': 381},
            'combat_recovery_revision': {'sha256': parent_sha, 'plan': str(parent),
                'last_optimizer_step': 4260, 'updates': 101, 'combat_updates': 76,
                'sampler_state': {'draws': 1216, 'negative_draws': 304}}},
        'model': {'weight': torch.tensor([.25, .75])},
        'optimizer': {'state': {0: {'step': torch.tensor(4260.), 'exp_avg': torch.tensor([.1, .2])}}},
        'rng_state': np.random.default_rng(173).bit_generator.state,
        'torch_rng_state': torch.get_rng_state().clone()}
    plan = {'version': 4, 'start_optimizer_step': 4260, 'base_plan_sha256': 'base',
        'parent_recovery_sha256': parent_sha, 'feedback_revision_sha256': 'feedback',
        'source_checkpoint_sha256': 'source', 'microbatch_size': 32,
        'gradient_accumulation_steps': 4, 'effective_batch_size': 128,
        'gradient_reduction': 'mean_over_microbatches', 'same_environment_per_optimizer_step': True,
        'same_context_per_optimizer_step': True, 'same_latent_mode_per_optimizer_step': True,
        'source_future_feedback': False, 'optimizer_reset': False, 'rollback': False}
    kwargs = dict(sha256='v4-sha', path=Path(folder)/'v4.json', parent_recovery_sha=parent_sha,
        source_checkpoint_sha='source', cli_microbatch=32, cli_accum=4)
    return state, plan, kwargs, parent


def commit(state, ledger, *, environment='combat'):
    step = state['global_training_step']+1
    commit_combat_update(state, step=step, environment=environment, batch_size=128)
    for name in ('feedback_revision', 'combat_recovery_revision'):
        state['rssm_training'][name]['last_optimizer_step'] = step
        state['rssm_training'][name]['updates'] += 1
    ledger.update(last_optimizer_step=step, updates=ledger['updates']+1)


class BatchRevisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_exact_boundary_preserves_tensors_adam_rng_old_plans_and_resume_clocks(self):
        with TemporaryDirectory() as folder:
            state, plan, kwargs, parent = fixture(folder)
            old = deepcopy(state); parent_bytes = parent.read_bytes(); rng = torch.get_rng_state().clone()
            ledger = bind_batch_revision(state, plan, **kwargs)
            self.assertEqual(state['loss_protocol'], BATCH_PROTOCOL)
            self.assertEqual(state['rssm_training']['combat_recovery_revision'], old['rssm_training']['combat_recovery_revision'])
            self.assertEqual(state['rssm_training']['feedback_revision'], old['rssm_training']['feedback_revision'])
            self.assertEqual(state['global_training_step'], 4260)
            self.assertEqual(state['rssm_training']['auxiliary_draws'], 68160)
            self.assertEqual(state['rng_state'], old['rng_state'])
            for group in ('model',):
                for name, tensor in old[group].items():
                    self.assertTrue(torch.equal(tensor, state[group][name]))
            for name, tensor in old['optimizer']['state'][0].items():
                self.assertTrue(torch.equal(tensor, state['optimizer']['state'][0][name]))
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            self.assertTrue(torch.equal(state['torch_rng_state'], old['torch_rng_state']))
            self.assertEqual(parent.read_bytes(), parent_bytes)
            self.assertEqual(ledger['updates'], 0)
            commit(state, ledger)
            commit(state, ledger, environment='navigation')
            self.assertEqual(state['global_training_step'], 4262)
            self.assertEqual(state['rssm_training']['auxiliary_draws'], 68160+256)
            self.assertEqual(state['rssm_training']['combat_auxiliary_draws'], 4656+128)
            self.assertEqual(state['rssm_training']['navigation_auxiliary_draws'], 1536+128)
            before_resume = deepcopy(state['rssm_training'])
            self.assertIs(bind_batch_revision(state, plan, **kwargs), ledger)
            self.assertEqual(state['rssm_training'], before_resume)
            self.assertEqual(len(state['objective_history']), 1)
            self.assertEqual(ledger['updates'], 2)
            self.assertEqual(parent.read_bytes(), parent_bytes)

    def test_changed_pins_cli_semantics_and_missing_feedback_fail_before_binding(self):
        changes = {'version': True, 'microbatch_size': 16, 'gradient_accumulation_steps': 2,
            'effective_batch_size': 32, 'gradient_reduction': 'sum',
            'same_environment_per_optimizer_step': False, 'same_context_per_optimizer_step': False,
            'same_latent_mode_per_optimizer_step': False, 'optimizer_reset': True,
            'rollback': True, 'source_future_feedback': True, 'source_checkpoint_sha256': 'wrong',
            'base_plan_sha256': 'wrong', 'feedback_revision_sha256': 'wrong',
            'parent_recovery_sha256': 'wrong', 'start_optimizer_step': 4259}
        with TemporaryDirectory() as folder:
            for name, value in changes.items():
                with self.subTest(name=name):
                    state, plan, kwargs, _ = fixture(folder); plan[name] = value
                    with self.assertRaises(ValueError): bind_batch_revision(state, plan, **kwargs)
                    self.assertNotIn('batch_revision', state['rssm_training'])
                    self.assertNotIn('objective_history', state)
            state, plan, kwargs, _ = fixture(folder)
            with self.assertRaises(ValueError):
                bind_batch_revision(state, plan, **{**kwargs, 'cli_accum': 1})
            state['rssm_training']['feedback_revision']['numeric_state_feedback'] = 'attached'
            with self.assertRaises(ValueError): bind_batch_revision(state, plan, **kwargs)
            state, plan, kwargs, parent = fixture(folder)
            parent.write_text(parent.read_text()+' ', encoding='utf-8')
            with self.assertRaises(ValueError): bind_batch_revision(state, plan, **kwargs)

    def test_effective128_requires_revision_and_v4_cannot_silently_commit16_or_reset_ledger(self):
        with TemporaryDirectory() as folder:
            state, plan, kwargs, _ = fixture(folder)
            with self.assertRaises(ValueError):
                commit_combat_update(state, step=4261, environment='combat', batch_size=128)
            ledger = bind_batch_revision(state, plan, **kwargs)
            for batch in (16, 32):
                with self.assertRaises(ValueError):
                    commit_combat_update(state, step=4261, environment='combat', batch_size=batch)
            self.assertEqual(state['global_training_step'], 4260)
            commit(state, ledger)
            with self.assertRaises(ValueError):
                bind_batch_revision(state, plan, **{**kwargs, 'sha256': 'changed'})
            ledger['updates'] = 0
            with self.assertRaises(ValueError): bind_batch_revision(state, plan, **kwargs)
            with self.assertRaises(ValueError):
                commit_combat_update(state, step=4262, environment='combat', batch_size=128)

    def test_sampler32_resumes16_state_without_reset_and_four_micros_have_exact128_quotas(self):
        with TemporaryDirectory() as folder:
            paths = make_source_files(Path(folder)/'train')
            data = SimpleNamespace(split='train', directory=paths[0].parent, episodes=[{'path': p} for p in paths])
            sampler = RecoveryCombatSampler(data); rng = np.random.default_rng(173)
            sampler.sample(16, rng)
            saved = deepcopy(sampler.state_dict()); saved_rng = deepcopy(rng.bit_generator.state)
            self.assertEqual(sampler.summary(batch_size=32)['quota'], {'spawn': 8, 'attack_negative': 8, 'ordinary': 16})
            all_pairs = [sampler.sample(32, rng) for _ in range(4)]
            self.assertEqual([len(p) for p in all_pairs], [32]*4)
            self.assertEqual(sampler.draws-saved['draws'], 128)
            self.assertEqual(sampler.negative_draws-saved['negative_draws'], 32)
            self.assertEqual(sum(sampler.anchor_counts.values())-sum(saved['anchor_counts'].values()), 32)
            self.assertLessEqual(max(sampler.anchor_counts.values())-min(sampler.anchor_counts.values()), 1)
            self.assertEqual(sampler.last_sample['quota'], {'spawn': 8, 'attack_negative': 8, 'ordinary_combat': 16})
            restored = RecoveryCombatSampler(data); restored.restore_state(saved)
            restored_rng = np.random.default_rng(); restored_rng.bit_generator.state = saved_rng
            for expected in all_pairs:
                np.testing.assert_array_equal(expected, restored.sample(32, restored_rng))
            self.assertEqual(sampler.state_dict(), restored.state_dict())
            before = deepcopy(sampler.state_dict())
            with self.assertRaises(ValueError): sampler.sample(128, rng)
            self.assertEqual(sampler.state_dict(), before)

    def test_four_microbatch_backward_average_matches_one_mean_objective_adam_update(self):
        first = torch.nn.Linear(2, 1); second = deepcopy(first)
        data = torch.arange(256, dtype=torch.float32).reshape(128, 2)/256
        target = data[:, :1]*.3-data[:, 1:]*.1
        opt1 = torch.optim.AdamW(first.parameters(), lr=1e-4)
        opt2 = torch.optim.AdamW(second.parameters(), lr=1e-4)
        for start in range(0, 128, 32):
            torch.nn.functional.smooth_l1_loss(first(data[start:start+32]), target[start:start+32]).backward()
        for parameter in first.parameters(): parameter.grad.mul_(.25)
        loss = sum(torch.nn.functional.smooth_l1_loss(second(data[start:start+32]), target[start:start+32])
                   for start in range(0, 128, 32))/4
        loss.backward()
        for p, q in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-6, atol=1e-7)
        opt1.step(); opt2.step()
        for p, q in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
        self.assertEqual({int(s['step']) for s in opt1.state.values()}, {1})


if __name__ == '__main__': unittest.main()
