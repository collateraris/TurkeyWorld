from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from turkeyworld.rssm_combat_memory_protocol import file_sha
from turkeyworld.rssm_moe_budget import (TARGETED_HORIZON_PROTOCOL, MANDATORY_HORIZON_UPDATES,
    targeted_horizon_plan, short_cap_plan, flat_cap_plan, budget_plan, bind_budget_policy,
    stage_budget, stage_requires_full_budget, curriculum_completion)
from turkeyworld.rssm_moe_protocol import (MOE_HORIZONS, moe_training_plan, bind_moe_training,
                                          commit_moe_update, moe_training_horizon)


class MoEHorizon100Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.source = self.root / 'source.pt'
        self.source.write_bytes(b'complete-boundary')
        self.state = {'global_training_step': 5352, 'rssm_training': {}}
        plan = moe_training_plan(self.state, source_checkpoint=self.source,
            source_checkpoint_sha256=file_sha(self.source), architecture_migration_sha256='a' * 64)
        path = self.root / 'moe.json'; path.write_text(json.dumps(plan), encoding='utf-8')
        bind_moe_training(self.state, plan, path=path, sha256=file_sha(path),
                          loaded_checkpoint_sha256=file_sha(self.source))
        self.activate(short_cap_plan, 'v4', cap=20)

    @property
    def gate(self): return self.state['rssm_training']['moe_training']

    def activate(self, factory=targeted_horizon_plan, tag='v5', **kwargs):
        source = self.root / (tag + '_' + str(self.state['global_training_step']) + '.pt')
        source.write_bytes(('full-boundary-' + str(self.state['global_training_step'])).encode())
        plan = factory(self.state, source_checkpoint=source, **kwargs)
        path = self.root / (tag + '.json'); path.write_text(json.dumps(plan), encoding='utf-8')
        bind_budget_policy(self.state, plan, path=path, sha256=file_sha(path),
                           loaded_checkpoint_sha256=file_sha(source))
        return plan, path

    def update(self, loss=1., eligible=True):
        horizon = moe_training_horizon(self.state); self.state['global_training_step'] += 1
        return commit_moe_update(self.state, step=self.state['global_training_step'], horizon=horizon,
                                  train_loss=loss, loss_gate_eligible=eligible)

    def test_full_stage_path_mandatory100_even_with_low_losses_and_final_stops_without_quality(self):
        plan, path = self.activate()
        self.assertEqual(plan['mandatory_horizon_updates'], MANDATORY_HORIZON_UPDATES)
        policy = self.gate['advance_policy']
        for index, horizon in enumerate(MOE_HORIZONS):
            self.assertEqual(self.gate['current_horizon'], horizon)
            mandatory = horizon in (240, 368, 450)
            self.assertEqual(stage_budget(index, policy), 100 if mandatory else 20)
            self.assertEqual(stage_requires_full_budget(index, policy), mandatory)
            self.assertEqual(self.gate['mandatory_stage_budget'], mandatory)
            self.assertEqual(self.gate['loss_streak_can_advance'], not mandatory)
            count = 100 if mandatory else 5
            for _ in range(count - 1): self.update(.01)
            self.assertEqual(self.gate['current_horizon'], horizon)
            if mandatory:
                self.assertEqual((self.gate['stage_updates'], self.gate['consecutive_below_threshold']), (99, 5))
                self.assertFalse(self.gate['goal_met']); self.assertIsNone(curriculum_completion(self.gate))
            self.update(.01)
            if horizon != 450:
                entry = self.gate['advance_history'][-1]
                self.assertEqual(entry['completed_updates_at_previous_stage'], count)
                self.assertEqual(entry['advance_reason'], 'stage_budget' if mandatory else 'loss_streak')
                self.assertEqual(entry['stage_budget'], 100 if mandatory else 20)
            else:
                self.assertEqual(curriculum_completion(self.gate), 'completed_budget')
                self.assertFalse(self.gate['goal_met']); self.assertTrue(self.gate['budget_exhausted'])
        before = deepcopy(self.gate)
        with self.assertRaises(ValueError): self.update(.01)
        self.assertEqual(self.gate, before)

    def test_current_stage_counts_streak_pcg_and_history_preserved_and_resume_exact(self):
        for _ in range(14 * 20): self.update(1.)
        for _ in range(9): self.update(1.)
        for _ in range(4): self.update(.4)
        self.assertEqual((self.gate['current_horizon'], self.gate['stage_updates'],
                          self.gate['consecutive_below_threshold']), (208, 13, 4))
        self.state['rssm_training']['private_nav_pcg'] = {'draws': 99999, 'rng': 'preserved'}
        self.state['preview_assets'] = {'renderer': 'unchanged', 'sr': 'unchanged'}
        before = deepcopy(self.state); plan, path = self.activate()
        for key in ('current_horizon', 'stage_index', 'stage_updates', 'updates', 'horizon_update_counts',
                    'advance_history', 'consecutive_below_threshold', 'main_examples', 'main_transitions'):
            self.assertEqual(self.gate[key], before['rssm_training']['moe_training'][key])
        self.assertEqual(self.state['rssm_training']['private_nav_pcg'], before['rssm_training']['private_nav_pcg'])
        self.assertEqual(self.state['preview_assets'], before['preview_assets'])
        self.assertEqual(self.gate['advance_policy']['source_gate'], before['rssm_training']['moe_training'])
        restored = json.loads(json.dumps(self.state))
        bind_budget_policy(restored, plan, path=path, sha256=file_sha(path), loaded_checkpoint_sha256='b' * 64)
        self.assertEqual(restored, self.state)
        self.update(.4)  # Existing streak4 counts; v5 does not restart or reinterpret it.
        self.assertEqual(self.gate['current_horizon'], 224)

    def test_ordinary_stage_twentieth_budget_and_strict_threshold_still_work(self):
        self.activate()
        for _ in range(19): self.update(.5)
        self.assertEqual((self.gate['current_horizon'], self.gate['stage_updates']), (4, 19))
        self.assertEqual(self.gate['consecutive_below_threshold'], 0)
        self.update(.5); self.assertEqual(self.gate['current_horizon'], 8)
        self.assertEqual(self.gate['last_advance_reason'], 'stage_budget')
        for _ in range(4): self.update(.49)
        self.update(.5); self.assertEqual(self.gate['consecutive_below_threshold'], 0)
        for _ in range(5): self.update(.49)
        self.assertEqual(self.gate['current_horizon'], 16)

    def test_mandatory_midstreak_reset_and_resume_never_enables_early_exit(self):
        plan, path = self.activate()
        while self.gate['current_horizon'] != 240: self.update(.1)
        for _ in range(7): self.update(.1)
        self.assertEqual(self.gate['consecutive_below_threshold'], 5)
        self.update(.1, eligible=False); self.assertEqual(self.gate['consecutive_below_threshold'], 0)
        for _ in range(13): self.update(.1)
        self.assertEqual((self.gate['current_horizon'], self.gate['stage_updates']), (240, 21))
        restored = json.loads(json.dumps(self.state))
        bind_budget_policy(restored, plan, path=path, sha256=file_sha(path), loaded_checkpoint_sha256='b' * 64)
        self.assertEqual(restored, self.state)
        for _ in range(78): self.update(.1)
        self.assertEqual(self.gate['current_horizon'], 240)
        self.update(.1); self.assertEqual(self.gate['current_horizon'], 256)
        self.assertEqual(self.gate['last_advance_reason'], 'stage_budget')

    def test_policy_history_flags_nan_and_fake_mandatory_loss_advance_rejected(self):
        self.activate(); original = deepcopy(self.state)
        for mutate in (
            lambda g: g['advance_policy'].update(budget_base=21),
            lambda g: g['advance_policy']['mandatory_horizon_updates'].update({'240': 99}),
            lambda g: g['advance_policy']['mandatory_horizon_updates'].update({'368': 100.}),
            lambda g: g['advance_policy'].update(mandatory_horizons_disable_loss_gate=False),
            lambda g: g['advance_policy'].update(final_stage_max_updates=100.),
            lambda g: g['advance_policy'].update(threshold=float('nan')),
            lambda g: g.update(mandatory_stage_budget=True),
            lambda g: g.update(effective_stage_budget=100),
            lambda g: g['advance_policy'].update(previous_policy_sha256='f' * 64),
            lambda g: g['advance_policy']['source_gate'].update(last_train_loss=123.)):
            self.state = deepcopy(original); mutate(self.gate)
            with self.assertRaises(ValueError): moe_training_horizon(self.state)
        self.state = deepcopy(original)
        for loss in (float('nan'), float('inf'), -.1):
            saved = deepcopy(self.gate); step = self.state['global_training_step'] + 1
            with self.assertRaises(ValueError):
                commit_moe_update({**self.state, 'global_training_step': step}, step=step,
                    horizon=4, train_loss=loss)
            self.assertEqual(self.gate, saved)
        while self.gate['current_horizon'] != 256: self.update(.1)
        self.gate['advance_history'][-1]['advance_reason'] = 'loss_streak'
        with self.assertRaises(ValueError): moe_training_horizon(self.state)

    def test_nested_v2_v3_v4_history_retains_actual_overdue190_and54_counts(self):
        # Recreate the real historical over-cap stages, rather than fabricating counts.
        original = self.gate['advance_policy']['source_gate']
        self.state['rssm_training']['moe_training'] = deepcopy(original)
        self.state['global_training_step'] = original['last_optimizer_step']
        for _ in range(190): self.update(1.)
        self.activate(budget_plan, 'legacy_v2')
        for _ in range(54): self.update(1.)
        self.activate(flat_cap_plan, 'legacy_v3', cap=50)
        self.activate(short_cap_plan, 'current_v4', cap=20)
        history = deepcopy(self.gate['advance_history'])
        old_bytes = (self.root / 'legacy_v2.json').read_bytes(), (self.root / 'legacy_v3.json').read_bytes()
        self.assertEqual([x['completed_updates_at_previous_stage'] for x in history], [190, 54])
        self.activate()
        for _ in range(5): self.update(.1)
        self.assertEqual(self.gate['current_horizon'], 32)
        self.assertEqual(self.gate['advance_history'][:2], history)
        self.assertEqual([self.gate['horizon_update_counts'][h] for h in ('4', '8')], [190, 54])
        self.assertEqual(old_bytes, ((self.root / 'legacy_v2.json').read_bytes(), (self.root / 'legacy_v3.json').read_bytes()))
        self.assertEqual(moe_training_horizon(json.loads(json.dumps(self.state))), 32)


if __name__ == '__main__': unittest.main()
