"""A missing counterfactual label must not discard an autonomous update."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import torch

from test_rssm_moe_training import TinyModel, data, oracle
from turkeyworld.rssm_accumulation import accumulated_metrics
from turkeyworld.rssm_moe_training import (
    COMBAT_COMPONENTS, _direct_task_terms, _own_branch_backward,
    moe_dynamics_backward,
)


QUERY = 'turkeyworld.rssm_combat_branches._source_step_at_own_state'
INVALID = 'invalid_direct_own_state_geometry'


def branch_inputs(model, batch):
    features = model.encode_map(batch['map'])
    memory = model.initialize_from_image(
        batch['frames'][:, :1], batch['states'][:, :1], features)
    return batch['states'][:, 0], memory, batch['map'], batch['actions']


def direct_inputs(model, batch):
    initial, memory, grid, actions = branch_inputs(model, batch)
    features = model.encode_map(grid)
    current, results, previous = initial, [], []
    for tick in range(actions.shape[1]):
        previous.append(current.detach())
        result = model.transition(current.detach(), actions[:, tick], features, memory)
        results.append(result)
        current, memory = result['state'], result['memory']
    return results, previous, actions, grid


class InvalidSourceSupportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(117)

    def assert_empty_support(self, report, actors, horizon):
        attempts = actors * horizon
        self.assertFalse(report['direct_source_supervision_available'])
        self.assertEqual(report['direct_source_support'], 'empty')
        self.assertEqual(report['direct_source_support_fraction'], 0.)
        self.assertEqual(report['direct_queries_attempted'], attempts)
        self.assertEqual(report['direct_queries_valid'], 0)
        self.assertEqual(sum(report['direct_invalid_skips'].values()), attempts)
        self.assertEqual(report['direct_present_combat_components'], [])
        self.assertEqual(report['direct_per_tick_counts'],
                         [dict(valid=0, hit=0, attack=0, spawn=0)] * horizon)
        self.assertEqual(sum(report['direct_spawn_anchor_counts'].values()), 0)
        for name in (*COMBAT_COMPONENTS, 'combat_npc_motion',
                     'combat_kinematic_player_axes', 'combat_kinematic_npc_speed',
                     'combat_kinematic_npc_direction', 'player_pass_loss'):
            self.assertEqual(report[name], 0., name)
        self.assertFalse(report['source_future_feedback'])
        self.assertTrue(report['direct_source_labels_after_own_rollout'])
        json.dumps(report, allow_nan=False)

    def test_real_wall_invalid_queries_return_complete_empty_support_without_repair(self):
        model, batch = TinyModel(), data(4)
        batch['map'].fill_(1)
        inputs = direct_inputs(model, batch)
        before = [value.clone() for value in inputs[1]]
        with patch('turkeyworld.engine.Game.render', side_effect=AssertionError('No RGB labels')):
            tasks, report = _direct_task_terms(*inputs, simulation_hz=30.)
        self.assert_empty_support(report, 4, 4)
        self.assertEqual(report['direct_invalid_skips'], {INVALID: 16})
        self.assertEqual(set(tasks), {'player', 'enemy', 'combat'})
        for value in tasks.values():
            self.assertEqual(float(value.detach()), 0.)
            self.assertTrue(bool(torch.isfinite(value)))
        for old, now in zip(before, inputs[1]):
            self.assertTrue(torch.equal(old, now), 'The actual own pose was repaired')
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_h144_empty_aux_and_late_keep_main_xy_router_gradients_and_every_transition(self):
        model, batch, auxiliary = TinyModel(), data(144, 2), data(16, 2)
        for tape in (batch, auxiliary):
            for value in tape.values():
                value.requires_grad_()
        weights = deepcopy(model.state_dict())
        with patch(QUERY, side_effect=ValueError(INVALID)) as queries:
            report = moe_dynamics_backward(model, batch, auxiliary, expected_horizon=144)
        self.assertEqual(queries.call_count, 64)
        self.assertEqual(len(model.calls), 144 + 16 + 16)
        self.assertEqual(report['moe_late_prefix_start'], 128)
        for key in ('moe_auxiliary', 'moe_late_own_prefix'):
            self.assert_empty_support(report[key], 2, 16)
            self.assertFalse(report[key]['direct_objective_backpropagated'])
        self.assertEqual(report['moe_combat_loss'], 0.)
        self.assertEqual(report['moe_direct_empty_branch_count'], 2)
        self.assertFalse(report['moe_direct_support_gate_eligible'])
        self.assertEqual(report['moe_player_loss'], report['moe_main_player_loss'])
        self.assertEqual(report['moe_enemy_loss'], report['moe_main_enemy_loss'])
        self.assertGreater(report['moe_player_loss'], 0.)
        self.assertGreater(report['moe_enemy_loss'], 0.)
        self.assertGreater(report['moe_router_loss'], 0.)
        self.assertAlmostEqual(report['train_loss'], sum(report[name] for name in
            ('moe_player_loss', 'moe_enemy_loss', 'moe_combat_loss', 'moe_router_loss')))
        for module in (model.p_core, model.e_core, model.p_head, model.e_head,
                       model.fusion.player_head, model.fusion.enemy_head, model.fusion.router):
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(all(bool(torch.isfinite(g).all()) for g in gradients))
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0.)
        for tape in (batch, auxiliary):
            self.assertTrue(all(value.grad is None for value in tape.values()))
        self.assertEqual(model.observed, [0., 0.])
        self.assertTrue(all(torch.equal(value, model.state_dict()[name])
                            for name, value in weights.items()))
        json.dumps(report, allow_nan=False)

    def test_valid_auxiliary_survives_an_empty_late_branch_without_update_skip(self):
        model, calls = TinyModel(), 0
        def source(*args):
            nonlocal calls
            calls += 1
            if calls > 8:
                raise ValueError(INVALID)
            return oracle(*args)
        with patch(QUERY, side_effect=source):
            report = moe_dynamics_backward(model, data(4, 2), data(4, 2))
        auxiliary, late = report['moe_auxiliary'], report['moe_late_own_prefix']
        self.assertEqual(auxiliary['direct_source_support'], 'complete')
        self.assertTrue(auxiliary['direct_objective_backpropagated'])
        self.assert_empty_support(late, 2, 4)
        self.assertFalse(late['direct_objective_backpropagated'])
        self.assertEqual(report['moe_direct_empty_branch_count'], 1)
        self.assertFalse(report['moe_direct_support_gate_eligible'])
        self.assertGreater(report['moe_combat_loss'], 0.)
        self.assertEqual(report['moe_combat_loss'], auxiliary['combat'])
        self.assertEqual(report['moe_player_loss'],
                         report['moe_main_player_loss'] + auxiliary['player'])
        self.assertGreater(float(model.attack.weight.grad.abs().sum()), 0.)
        self.assertEqual(len(model.calls), 12)

    def test_current_H240_main19_aux32_late4_keeps_valid_auxiliary_and_marks_empty_late(self):
        model,calls=TinyModel(),0
        def source(*args):
            nonlocal calls
            calls+=1
            if calls>32*16:
                raise ValueError('Cannot restore an entity inside a wall')
            return oracle(*args)
        with patch(QUERY,side_effect=source):
            report=moe_dynamics_backward(model,data(240,19),data(16,32),expected_horizon=240,
                main_player_weight=19/32,main_router_weight=19/32,main_enemy_weight=19/27,
                auxiliary_actor_limit=32,late_own_prefix_actors=4)
        auxiliary,late=report['moe_auxiliary'],report['moe_late_own_prefix']
        self.assertEqual(calls,(32+4)*16)
        self.assertEqual(len(model.calls),240+16+16)
        self.assertEqual(report['moe_late_prefix_start'],224)
        self.assertEqual(auxiliary['actors'],32)
        self.assertEqual(auxiliary['direct_queries_valid'],512)
        self.assertTrue(auxiliary['direct_objective_backpropagated'])
        self.assert_empty_support(late,4,16)
        self.assertFalse(late['direct_objective_backpropagated'])
        self.assertEqual(late['direct_invalid_skips'],{'Cannot restore an entity inside a wall':64})
        self.assertEqual(report['moe_direct_empty_branch_count'],1)
        self.assertFalse(report['moe_direct_support_gate_eligible'])
        self.assertGreater(report['moe_main_player_loss'],0.)
        self.assertGreater(report['moe_main_enemy_loss'],0.)
        self.assertGreater(report['moe_router_loss'],0.)
        self.assertGreater(report['moe_combat_loss'],0.)
        self.assertEqual(report['moe_combat_loss'],auxiliary['combat'])
        self.assertEqual(report['moe_player_loss'],report['moe_main_player_loss']+auxiliary['player'])
        for module in(model.p_core,model.e_core,model.fusion.router,model.attack):
            gradients=[p.grad for p in module.parameters()if p.grad is not None]
            self.assertTrue(all(bool(torch.isfinite(g).all())for g in gradients))
            self.assertGreater(sum(float(g.abs().sum())for g in gradients),0.)
        json.dumps(report,allow_nan=False)

    def test_one_legal_tick_after_empty_support_restores_direct_head_training(self):
        model, batch = TinyModel(), data(4, 1)
        with patch(QUERY, side_effect=ValueError(INVALID)):
            empty = _own_branch_backward(model, *branch_inputs(model, batch), amp=None)
        self.assertFalse(empty['direct_objective_backpropagated'])
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        attempts = 0
        def one_legal(*args):
            nonlocal attempts
            attempts += 1
            if attempts < 4:
                raise ValueError(INVALID)
            return oracle(*args)
        with patch(QUERY, side_effect=one_legal):
            restored = _own_branch_backward(model, *branch_inputs(model, batch), amp=None)
        self.assertTrue(restored['direct_objective_backpropagated'])
        self.assertTrue(restored['direct_source_supervision_available'])
        self.assertEqual(restored['direct_source_support'], 'partial')
        self.assertEqual(restored['direct_source_support_fraction'], .25)
        self.assertEqual(restored['direct_queries_valid'], 1)
        self.assertEqual([row['valid'] for row in restored['direct_per_tick_counts']], [0, 0, 0, 1])
        for module in (model.target, model.attack, model.p_core, model.e_core):
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(all(bool(torch.isfinite(g).all()) for g in gradients))
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0.)

    def test_partial_support_is_exact_valid_row_objective_and_gradient(self):
        model, batch = TinyModel(), data(4, 4)
        batch['map'][:, 0, 0] = torch.arange(4)
        results, previous, actions, grid = direct_inputs(model, batch)
        def selected_source(state, action, actor_grid, *args):
            if actor_grid[0, 0] != 0:
                raise ValueError(INVALID)
            return oracle(state, action, actor_grid, *args)
        with patch(QUERY, side_effect=selected_source):
            mixed, report = _direct_task_terms(results, previous, actions, grid, simulation_hz=30.)
        rows = [{name: value[:1] if torch.is_tensor(value) else value
                 for name, value in row.items()} for row in results]
        with patch(QUERY, side_effect=oracle):
            only, complete = _direct_task_terms(rows, [row[:1] for row in previous],
                                                actions[:1], grid[:1], simulation_hz=30.)
        self.assertEqual(report['direct_source_support'], 'partial')
        self.assertEqual(report['direct_queries_valid'], 4)
        self.assertEqual(report['direct_invalid_skips'], {INVALID: 12})
        self.assertEqual(complete['direct_source_support'], 'complete')
        parameters = tuple(model.parameters())
        for task in ('player', 'enemy', 'combat'):
            torch.testing.assert_close(mixed[task], only[task], rtol=0, atol=0)
            a = torch.autograd.grad(mixed[task], parameters, retain_graph=True, allow_unused=True)
            b = torch.autograd.grad(only[task], parameters, retain_graph=True, allow_unused=True)
            for x, y in zip(a, b):
                self.assertEqual(x is None, y is None)
                if x is not None:
                    torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_unknown_label_failure_remains_an_error(self):
        model, batch = TinyModel(), data(4, 1)
        inputs = direct_inputs(model, batch)
        with patch(QUERY, side_effect=ValueError('broken_supervision_shape')):
            with self.assertRaisesRegex(ValueError, 'broken_supervision_shape'):
                _direct_task_terms(*inputs, simulation_hz=30.)

    def test_canonical_and_dead_corpse_wall_rejections_have_explicit_empty_support(self):
        for reason in ('noncanonical_direct_own_state_no_repair',
                       'Cannot restore an entity inside a wall'):
            with self.subTest(reason=reason):
                model, batch = TinyModel(), data(4, 1)
                with patch(QUERY, side_effect=ValueError(reason)):
                    tasks, report = _direct_task_terms(
                        *direct_inputs(model, batch), simulation_hz=30.)
                self.assert_empty_support(report, 1, 4)
                self.assertEqual(report['direct_invalid_skips'], {reason: 4})
                self.assertEqual(sum(float(v.detach()) for v in tasks.values()), 0.)

    def test_one_unsupported_micro_prevents_complete_update_loss_streak_eligibility(self):
        model, rows = TinyModel(), []
        for index in range(4):
            query = ValueError(INVALID) if index == 1 else oracle
            with patch(QUERY, side_effect=query):
                rows.append(moe_dynamics_backward(model, data(4, 2), data(4, 2)))
        self.assertEqual([row['moe_direct_support_gate_eligible'] for row in rows],
                         [True, False, True, True])
        report = accumulated_metrics(rows)
        self.assertFalse(report['moe_direct_support_gate_eligible'])
        self.assertEqual(report['moe_direct_empty_branch_count'], 2)
        self.assertEqual(len(report['microbatch_metrics']), 4)
        self.assertAlmostEqual(report['train_loss'], sum(row['train_loss'] for row in rows) / 4)
        self.assertGreater(report['train_loss'], 0.)
        json.dumps(report, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
