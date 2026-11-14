"""Player repair is an explicit child objective, not a replacement history."""
from copy import deepcopy
import json
import random
from pathlib import Path
import unittest

import numpy as np
import torch

import test_rssm_moe_horizon100 as legacy
from turkeyworld.rssm_moe_budget import h450_loss_goal_plan, curriculum_completion
from turkeyworld.rssm_moe_batch_revision import (
    batch_revision_plan, micro32_batch_revision_plan, bind_moe_batch_revision,
    moe_batch_settings,
)
from turkeyworld.rssm_moe_protocol import commit_moe_update, moe_training_horizon
from turkeyworld.rssm_player_navigation_revision import (
    PROTOCOL, digest, file_sha, player_navigation_settings, player_navigation_plan,
    bind_player_navigation_revision, mark_player_navigation_baseline,
    commit_player_navigation_revision,
)


class PlayerNavigationRevisionTests(unittest.TestCase):
    def setUp(self):
        self.old = legacy.MoEHorizon100Tests('runTest')
        self.old.setUp(); self.addCleanup(self.old.doCleanups)
        self.old.activate()
        self.old.activate(h450_loss_goal_plan, 'loss450')
        while self.old.gate['current_horizon'] != 450:
            self.old.update(.75)
        self.state, self.root, self.source = self.old.state, self.old.root, self.old.source
        for factory, name in ((batch_revision_plan, 'micro16'), (micro32_batch_revision_plan, 'micro32')):
            plan = factory(self.state, source_checkpoint=self.source)
            path = self.write(name + '.json', plan)
            bind_moe_batch_revision(self.state, plan, path=path, sha256=file_sha(path),
                                    loaded_checkpoint_sha256=file_sha(self.source))
        contact = self.root / 'contacts'
        (contact / 'short/rgb/train').mkdir(parents=True)
        (contact / 'manifest.json').write_text('{"TRAIN":6912}', encoding='utf-8')
        self.contact = contact
        self.identity = dict(protocol='contact_train_test', source_fingerprint='a' * 64,
                             cohorts=dict(x_free_y_blocked=8, x_blocked_y_free=8))
        self.state.update(loss=3.95, latest_metrics=dict(train_loss=3.95, old=True),
                          loss_groups=dict(total=3.95), preview_assets=dict(renderer='pinned'))
        self.state['rssm_training']['old_sampler'] = dict(draws=123, rng=dict(state=42))
        self.state['_model'] = dict(parameter=[1, 2, 3])
        self.state['_optimizer'] = dict(states=442, groups=7, step=6508)
        self.plan = player_navigation_plan(self.state, source_checkpoint=self.source,
            source_checkpoint_sha256=file_sha(self.source), contact_root=contact,
            contact_sampler_identity=self.identity)
        self.path = self.write('player_navigation.json', self.plan)
        self.source_state = deepcopy(self.state)

    def write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value), encoding='utf-8')
        return path

    def bind(self, *, state=None, plan=None, path=None, loaded=None):
        path = self.path if path is None else path
        return bind_player_navigation_revision(self.state if state is None else state,
            self.plan if plan is None else plan, path=path, sha256=file_sha(path),
            loaded_checkpoint_sha256=loaded or file_sha(self.source))

    def next_gate(self, loss=3.95):
        self.state['global_training_step'] += 1
        step = self.state['global_training_step']
        commit_moe_update(self.state, step=step, horizon=450, train_loss=loss,
                          microbatches=1, microbatch_size=32, effective_batch_size=32)
        return step

    def test_activation_preserves_gate_assets_old_samplers_and_all_rng(self):
        before = deepcopy(self.state)
        numpy_rng, python_rng, torch_rng = deepcopy(np.random.get_state()), random.getstate(), torch.get_rng_state().clone()
        record = self.bind()
        for key in ('moe_training', 'old_sampler'):
            self.assertEqual(self.state['rssm_training'][key], before['rssm_training'][key])
        for key in ('_model', '_optimizer', 'preview_assets', 'loss', 'latest_metrics', 'loss_groups'):
            self.assertEqual(self.state[key], before[key])
        self.assertEqual(digest(np.random.get_state()[1].tolist()), digest(numpy_rng[1].tolist()))
        self.assertEqual(random.getstate(), python_rng)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))
        self.assertEqual((record['updates'], record['contact_draws'], record['baseline_complete']), (0, 0, False))
        self.assertEqual(record['settings']['tbptt'], 64)
        self.assertEqual(record['settings']['contact']['horizons'], [4, 8, 16, 32])
        self.assertAlmostEqual(sum(record['settings']['player_weights'].values()), 1.)
        history = self.state['objective_history'][-1]
        self.assertEqual(history['source_latest_metrics'], before['latest_metrics'])
        self.assertFalse(history['previous_TOTAL_comparable'])
        self.assertEqual(self.state['active_objective_formula_revision'], PROTOCOL)
        self.assertEqual(moe_training_horizon(self.state), 450)
        self.assertEqual(moe_batch_settings(self.state)['effective_batch_size'], 32)

    def test_baseline_then_complete_commit_resume_preserve_h450_streak(self):
        record = self.bind()
        mark_player_navigation_baseline(self.state, step=self.state['global_training_step'])
        step = self.next_gate(.99)
        metrics = dict(player_contact_loss=.1, player_navigation_loss_revision=PROTOCOL,
                       player_contact_generated_transitions=160)
        private_pcg = dict(draws=4, rng=dict(state=123), protocol='contact_train_test')
        commit_player_navigation_revision(self.state, step=step,
            contact_sampler_state=private_pcg, metrics=metrics)
        self.assertEqual((record['updates'], record['contact_draws'], record['contact_generated_transitions']), (1, 4, 160))
        self.assertEqual(record['contact_sampler_state'], private_pcg)
        restored = json.loads(json.dumps(self.state))
        before = deepcopy(restored)
        self.bind(state=restored, loaded='e' * 64)  # Resumes need not reload the ancestor bytes.
        self.assertEqual(restored, before)
        self.assertEqual(restored['rssm_training']['moe_training']['final_horizon_goal']['streak'], 1)
        self.assertIsNone(curriculum_completion(restored['rssm_training']['moe_training']))
        for mutate in (lambda r:r.update(baseline_step=None),
                       lambda r:r.update(version=2),
                       lambda r:r['contact_sampler_state'].update(draws=0)):
            bad = deepcopy(restored); mutate(bad['rssm_training']['player_navigation_revision'])
            with self.assertRaises(ValueError): self.bind(state=bad, loaded='e' * 64)
        with self.assertRaises(ValueError):
            mark_player_navigation_baseline(self.state, step=step)
        with self.assertRaises(ValueError):
            commit_player_navigation_revision(self.state, step=step, contact_sampler_state=private_pcg, metrics=metrics)

    def test_no_optimizer_commit_before_fresh_baseline_or_with_wrong_private_draws(self):
        record = self.bind(); step = self.next_gate()
        before = deepcopy(record)
        metrics = dict(player_contact_loss=.1, player_navigation_loss_revision=PROTOCOL)
        with self.assertRaisesRegex(ValueError, 'baseline'):
            commit_player_navigation_revision(self.state, step=step,
                contact_sampler_state=dict(draws=4), metrics=metrics)
        self.assertEqual(record, before)
        self.state = deepcopy(self.source_state)
        record = self.bind()
        mark_player_navigation_baseline(self.state, step=self.state['global_training_step'])
        step = self.next_gate()
        before = deepcopy(record)
        for draws, loss, transitions in ((0, .1, 160), (8, .1, 160), (4, float('nan'), 160), (4, .1, -1)):
            bad_metrics = dict(metrics, player_contact_loss=loss,
                               player_contact_generated_transitions=transitions)
            with self.subTest(draws=draws, loss=loss, transitions=transitions):
                with self.assertRaises(ValueError):
                    commit_player_navigation_revision(self.state, step=step,
                        contact_sampler_state=dict(draws=draws), metrics=bad_metrics)
                self.assertEqual(record, before)

    def test_new_objective_rejects_changed_source_plan_manifest_and_horizon(self):
        with self.assertRaises(ValueError): self.bind(loaded='f' * 64)
        for mutate in (
            lambda p: p['settings'].update(tbptt=16),
            lambda p: p['settings']['contact'].update(weight=.5),
            lambda p: p.update(source_checkpoint_sha256='b' * 64),
            lambda p: p.update(contact_sampler_identity_sha256='c' * 64),
            lambda p: p.update(batch_settings=dict(microbatch_size=16, gradient_accumulation_steps=2, effective_batch_size=32)),
        ):
            bad = deepcopy(self.plan); mutate(bad)
            with self.subTest(plan=bad):
                with self.assertRaises(ValueError): self.bind(plan=bad, path=self.write('tampered.json', bad))
        (self.contact / 'manifest.json').write_text('{"TRAIN":6913}', encoding='utf-8')
        with self.assertRaises(ValueError): self.bind()

    def test_changed_goal_cannot_be_silently_reset_for_activation(self):
        self.next_gate(.99)
        self.assertEqual(self.state['rssm_training']['moe_training']['final_horizon_goal']['streak'], 1)
        with self.assertRaises(ValueError):
            player_navigation_plan(self.state, source_checkpoint=self.source,
                source_checkpoint_sha256=file_sha(self.source), contact_root=self.contact,
                contact_sampler_identity=self.identity)
        with self.assertRaises(ValueError): self.bind()


if __name__ == '__main__': unittest.main()
