import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from turkeyworld.rssm_fusion_moe import RSSMFusionMoE, RSSMFusionMoEConfig
from turkeyworld.rssm_moe_protocol import (MOE_HORIZONS, SETTINGS, moe_training_plan,
    bind_moe_training, commit_moe_update, moe_training_horizon)
from turkeyworld.rssm_moe_training import (moe_dynamics_backward, summarize_moe_metrics,
    _direct_task_terms)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(simulation_hz=30., navigation_only=False)
        self.p_core, self.e_core = nn.Linear(12, 12), nn.Linear(10, 10)
        self.p_head, self.e_head = nn.Linear(12, 2), nn.Linear(10, 2)
        self.target, self.attack, self.events = nn.Linear(12, 2), nn.Linear(10, 1), nn.Linear(12, 4)
        self.tile, self.anchor = nn.Linear(10, 256), nn.Linear(10, 9)
        self.fusion = RSSMFusionMoE(RSSMFusionMoEConfig(width=16, depth=1, heads=4, expert_hidden=32),
            player_memory_dim=12, enemy_memory_dim=10, map_width=16)
        self.observed = []; self.calls = []

    def encode_map(self, grid):
        return torch.linspace(0, 1, 16, device=grid.device)[None].expand(len(grid), -1)

    def initialize_from_image(self, images, states, features):
        self.observed.append(float(images.max()))
        state = states[:, -1]
        return torch.cat((state[:, :2].repeat(1, 6), state[:, 8:10].repeat(1, 5)), -1)

    def transition(self, state, action, features, memory):
        self.calls.append((state.detach().clone(), memory.detach().clone()))
        p, e = memory[:, :12], memory[:, 12:]
        f = self.fusion(p, e, state, action, features, features[:, None].expand(-1, 2, -1))
        p, e = self.p_core(p).tanh(), self.e_core(e).tanh()
        passed = self.p_head(p) + f['player_pass_residual']
        motion = self.e_head(e) + f['enemy_motion_residual']
        predicted = torch.cat((state[:, :2] + .004 * passed.sigmoid(), state[:, 2:8],
            state[:, 8:10] + .002 * motion.tanh(), state[:, 10:]), -1)
        tile = self.tile(e)[:, None] + f['interaction_spawn_tile_residual']
        return {'state': predicted, 'memory': torch.cat((p, e), -1),
            'player_pass_logits': passed, 'target_logits': self.target(p) + f['interaction_target_residual'],
            'attack_logits': self.attack(e) + f['interaction_attack_residual'],
            'event_logits': self.events(p), 'spawn_logits': tile, 'spawn_fallback_logits': tile,
            'spawn_anchor_logits': self.anchor(e)[:, None] + f['interaction_spawn_anchor_residual'],
            'shot': action[:, 3], 'respawn_flags': torch.zeros(len(state), 1, dtype=torch.bool),
            'rssm_fusion_moe': f}


def data(horizon, count=4):
    state = torch.zeros(count, 20)
    state[:, :2] = .25; state[:, 3:5] = 1; state[:, 5] = 1
    state[:, 8:10] = torch.tensor([.26, .25]); state[:, 11] = 1
    state[:, 12:14] = torch.tensor([.6, 1.])
    if count > 1: state[1, 12:14] = 0
    if count > 2: state[2, 12] = .25
    if count > 3: state[3, 12:14] = 0; state[3, 19] = .01
    states = state[:, None].expand(-1, horizon + 1, -1).clone()
    for tick in range(horizon + 1):
        states[:, tick, :2] += .003 * tick
        states[:, tick, 8:10] += .001 * tick
    frames = torch.full((count, horizon + 1, 3, 32, 32), 99.)
    frames[:, 0] = 0
    return {'states': states, 'frames': frames, 'context_frames': frames[:, :1],
        'target_states': states[:, 1:].clone(), 'map': torch.zeros(count, 16, 16),
        'actions': torch.ones(count, horizon, 4)}


def oracle(state, action, grid, protocol, dt):
    target = state.copy(); target[:2] += .003; target[8:10] += .001
    alive = state[13] >= .5 and state[12] > 0
    hit = 0 if alive else -1
    if alive:
        target[12] = max(0, state[12] - .35); target[13] = target[12] > 0
    else:
        from turkeyworld.spawn import SPAWN_ANCHORS
        target[8:10] = ((np.array(SPAWN_ANCHORS[0]) + .5) / 16
                         if state[19] < .005 else np.array([3.5, 3.5]) / 16)
        target[12:14] = [.6, 1.]
    attack = float(alive and target[13] > .5)
    target[4] = max(0, state[4] - .08 * attack)
    return {'previous': state.copy(), 'target': target,
        'events': np.array([1., float(hit >= 0), float(alive and target[13] < .5), attack], np.float32),
        'hit': hit, 'attack': [attack]}


class MoEProtocolTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(); self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / 'plan.json'
        self.old = {'loss_gated_curriculum': {'current_horizon': 128, 'updates': 212},
                    'loss_balance_state': {'ema': [1., 2.]}}
        self.state = {'global_training_step': 5322, 'rssm_training': copy.deepcopy(self.old)}
        self.plan = moe_training_plan(self.state, source_checkpoint='migrated.pt',
            source_checkpoint_sha256='a' * 64, architecture_migration_sha256='b' * 64,
            data_roots={'train': 'source/train'}, source_identities={'source': 'c' * 64})
        self.path.write_text(json.dumps(self.plan), encoding='utf-8')
        self.sha = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.bind()

    def bind(self):
        return bind_moe_training(self.state, self.plan, path=self.path, sha256=self.sha,
                                 loaded_checkpoint_sha256='a' * 64)

    def step(self, loss=.1, **kwargs):
        h = moe_training_horizon(self.state); self.state['global_training_step'] += 1
        return commit_moe_update(self.state, step=self.state['global_training_step'],
                                  horizon=h, train_loss=loss, **kwargs)

    def test_all31_stages_nine_hold_tenth_advance_new_zero_final_continues(self):
        self.assertEqual(len(MOE_HORIZONS), 31)
        self.assertEqual(MOE_HORIZONS, (4, 8, 16, *range(32, 449, 16), 450))
        for stage, h in enumerate(MOE_HORIZONS):
            self.assertEqual(moe_training_horizon(self.state), h)
            for streak in range(1, 10):
                q = self.step(); self.assertEqual(q['current_horizon'], h)
                self.assertEqual(q['consecutive_below_threshold'], streak)
            q = self.step()
            self.assertEqual(q['current_horizon'], MOE_HORIZONS[min(stage + 1, 30)])
            self.assertEqual(q['consecutive_below_threshold'], 0 if stage < 30 else 10)
        self.assertTrue(q['goal_met']); self.step(); self.assertEqual(moe_training_horizon(self.state), 450)
        self.assertEqual({key: self.state['rssm_training'][key] for key in self.old}, self.old)
        self.assertEqual(self.state['moe_training_history'][0]['rssm_training'], self.old)

    def test_equality_resets_midstreak_resume_and_complete64x4_only(self):
        for _ in range(5): self.step()
        self.state = json.loads(json.dumps(self.state)); self.bind()
        self.assertEqual(self.state['rssm_training']['moe_training']['consecutive_below_threshold'], 5)
        self.assertEqual(self.step(.25)['consecutive_below_threshold'], 0)
        for _ in range(9): self.step()
        self.assertEqual(moe_training_horizon(self.state), 4)
        self.step(); self.assertEqual(moe_training_horizon(self.state), 8)
        before = copy.deepcopy(self.state)
        for loss, extras in [(float('nan'), {}), (.1, {'microbatches': 3}), (.1, {'microbatch_size': 32})]:
            self.state = copy.deepcopy(before)
            with self.assertRaises(ValueError): self.step(loss, **extras)

    def test_schema_pins_data_and_sum_combat_no_floor_or_ema(self):
        self.assertFalse(SETTINGS['ema_normalization']); self.assertFalse(SETTINGS['posterior_reconstruction_KL'])
        self.assertEqual(SETTINGS['latent_cycle'], ['argmax', 'argmax', 'sample', 'argmax'])
        self.assertTrue(SETTINGS['combat_reduction'].startswith('sum_'))
        changed = copy.deepcopy(self.plan); changed['data_roots']['train'] = 'other'
        with self.assertRaises(ValueError):
            bind_moe_training(self.state, changed, path=self.path, sha256=self.sha,
                              loaded_checkpoint_sha256='a' * 64)
        report = summarize_moe_metrics([{'moe_player_loss': .1, 'moe_enemy_loss': .1,
            'moe_combat_loss': .7, 'moe_router_loss': .01}])
        self.assertAlmostEqual(report['total'], .91)
        self.assertEqual(report['interaction'], report['combat'])

    def test_new_phase_and_resume_mark_all_tasks_active_without_rewriting_archived_flags(self):
        state = {'global_training_step': 5322,
            'rssm_training': {'objective_protocol': 'old_xy_only', 'index_reservation_only': True},
            'training_protocol': {'auxiliary_source_training_active': False,
                'combat_head_training_active': False, 'index_reservation_only': True}}
        old = copy.deepcopy(state)
        bind_moe_training(state, self.plan, path=self.path, sha256=self.sha,
                           loaded_checkpoint_sha256='a' * 64)
        self.assertEqual(state['moe_training_history'][0]['rssm_training'], old['rssm_training'])
        self.assertEqual(state['moe_training_history'][0]['training_protocol'], old['training_protocol'])
        for _ in range(2):
            protocol = state['training_protocol']
            self.assertTrue(protocol['auxiliary_source_training_active'])
            self.assertTrue(protocol['combat_head_training_active'])
            self.assertFalse(protocol['index_reservation_only'])
            self.assertFalse(protocol['ema_normalization'])
            self.assertEqual(protocol['active_task_heads'], ['player_navigation', 'enemy_navigation', 'combat'])
            self.assertEqual(state['rssm_training']['objective_protocol'], SETTINGS['objective_protocol'])
            self.assertFalse(state['rssm_training']['index_reservation_only'])
            bind_moe_training(state, self.plan, path=self.path, sha256=self.sha,
                               loaded_checkpoint_sha256='a' * 64)

    def test_unsupported_direct_branch_cannot_qualify_loss_advance_or_history(self):
        for _ in range(9): self.step(.01)
        gate = self.step(.01, loss_gate_eligible=False)
        self.assertEqual(gate['current_horizon'], 4)
        self.assertEqual(gate['consecutive_below_threshold'], 0)
        self.assertFalse(gate['last_gate_passed'])
        self.bind()
        for _ in range(10): self.step(.01)
        gate = self.state['rssm_training']['moe_training']
        self.assertEqual(gate['current_horizon'], 8)
        gate['advance_history'][0]['loss_gate_eligible'] = False
        with self.assertRaises(ValueError): moe_training_horizon(self.state)


class MoETrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_real_three_heads_both_cores_router_gradients_and_all_steps_direct_labels(self):
        torch.manual_seed(7); model = TinyModel(); source_queries = []
        def traced(*args):
            source_queries.append(len(model.calls)); return oracle(*args)
        with patch('turkeyworld.rssm_combat_branches._source_step_at_own_state', side_effect=traced):
            metrics = moe_dynamics_backward(model, data(4), data(4), expected_horizon=4)
        for head in (model.fusion.player_head, model.fusion.enemy_head, model.fusion.combat_head,
                     model.p_core, model.e_core, model.fusion.router):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in head.parameters() if p.grad is not None), 0)
        self.assertEqual(model.observed, [0., 0.])
        self.assertEqual(set(source_queries), {8, 12})
        self.assertEqual(metrics['moe_late_prefix_start'], 0)
        for key in ('moe_auxiliary', 'moe_late_own_prefix'):
            branch = metrics[key]
            self.assertEqual(branch['horizon'], 4); self.assertEqual(branch['actors'], 4)
            self.assertEqual(len(branch['direct_per_tick_counts']), 4)
            self.assertGreater(branch['direct_per_tick_counts'][-1]['spawn'], 0)
        self.assertAlmostEqual(metrics['train_loss'], sum(metrics[k] for k in
            ('moe_player_loss', 'moe_enemy_loss', 'moe_combat_loss', 'moe_router_loss')))
        json.dumps(metrics, allow_nan=False)

    def test_each_task_updates_its_specialized_head_without_other_readout_gradients(self):
        torch.manual_seed(8); model = TinyModel(); b = data(4)
        features = model.encode_map(b['map']); memory = model.initialize_from_image(b['frames'][:, :1], b['states'][:, :1], features)
        results, previous = [], []; current = b['states'][:, 0]
        for tick in range(4):
            previous.append(current.detach())
            result = model.transition(current.detach(), b['actions'][:, tick], features, memory)
            results.append(result); current, memory = result['state'], result['memory']
        with patch('turkeyworld.rssm_combat_branches._source_step_at_own_state', side_effect=oracle):
            tasks, _ = _direct_task_terms(results, previous, b['actions'], b['map'], simulation_hz=30.)
        heads = {'player': model.fusion.player_head, 'enemy': model.fusion.enemy_head, 'combat': model.fusion.combat_head}
        for name, value in tasks.items():
            model.zero_grad(set_to_none=True); value.backward(retain_graph=True)
            self.assertGreater(float(heads[name].weight.grad.abs().sum()), 0)
            for other in set(heads) - {name}:
                grad = heads[other].weight.grad
                self.assertTrue(grad is None or not bool(grad.any()), (name, other))

    def test_router_regularization_is_H_mean_not_H_dependent_floor(self):
        values = []
        for horizon in (4, 32):
            torch.manual_seed(9); model = TinyModel()
            with torch.no_grad(): model.fusion.router.weight.zero_(); model.fusion.router.bias.zero_()
            with patch('turkeyworld.rssm_combat_branches._source_step_at_own_state', side_effect=oracle):
                report = moe_dynamics_backward(model, data(horizon), data(min(horizon, 16)))
            values.append(report['moe_router_loss'])
            self.assertEqual(report['moe_late_prefix_start'], max(0, horizon - 16))
        self.assertAlmostEqual(*values, places=7)

    def test_unknown_direct_error_and_wrong_aux_horizon_fail_closed(self):
        model = TinyModel()
        with patch('turkeyworld.rssm_combat_branches._source_step_at_own_state', side_effect=ValueError('illegal')):
            with self.assertRaisesRegex(ValueError, 'illegal'):
                moe_dynamics_backward(model, data(4), data(4))
        with self.assertRaisesRegex(ValueError, r'min\(currentH,16\)'):
            moe_dynamics_backward(TinyModel(), data(4), data(16))


if __name__ == '__main__': unittest.main()
