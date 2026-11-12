from copy import deepcopy
from types import SimpleNamespace
import json
import unittest
from unittest.mock import patch

import torch
import numpy as np
from torch import nn

from turkeyworld.rssm_late_prefix_recovery import (LATE_PREFIX_RECOVERY_PROTOCOL,
    normalize_late_prefix_protocol, continue_own_prefix_candidates,
    order_late_prefix_probes, select_late_prefix_rows, _prefix_map_caches)
from turkeyworld.rssm_combat_branches import select_combat_branches


def state(count=2):
    value = torch.zeros(count, 20)
    value[:, [0, 1, 8, 9]] = .25
    value[:, [3, 4, 5, 12, 13]] = 1.
    return value


class CountingMemory(nn.Module):
    def __init__(self):
        super().__init__()
        self.delta = nn.Parameter(torch.tensor(.00001))
        self.calls = []
        self.config = SimpleNamespace()
        self.initializations = 0

    def encode_map(self, grid):
        return self.delta.new_zeros(len(grid), 1)

    def initialize_from_image(self, frames, states, features):
        self.initializations += 1
        return self.delta.new_zeros(len(states), 1)

    def transition(self, previous, action, features, memory):
        self.calls.append((previous.detach().clone(), memory.detach().clone(), torch.is_grad_enabled()))
        # Consume RNG to verify the scout restores its private sampling stream.
        noise = torch.rand((), device=previous.device)*0
        addition = torch.zeros_like(previous)
        addition[:, 0] = self.delta+noise
        return {'state': previous+addition, 'memory': memory+1+self.delta*0}


def row(actor, step, cohort='contrast', *, contact=False, recovery=False):
    return {'actor': actor, 'step': step, 'cohort': cohort, 'pre_wall': contact,
        'post_spawn_recovery': recovery, 'first_source_error': step+1,
        'first_wall_step': step+3 if contact else None,
        'item': {'actions': torch.zeros(1, 16, 4)}}


class LatePrefixRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_protocol_opt_in_is_exact_and_type_strict(self):
        self.assertIsNone(normalize_late_prefix_protocol(None))
        self.assertEqual(normalize_late_prefix_protocol(LATE_PREFIX_RECOVERY_PROTOCOL), LATE_PREFIX_RECOVERY_PROTOCOL)
        for changed in ({'version': True}, {'scout_batch_size': 16}, {'no_grad_scout': False},
                        {'minimum_prefix_steps': 0}, {'new_observed_rgb_frames': False}, {'unknown': 1}):
            with self.assertRaises(ValueError):
                normalize_late_prefix_protocol({**LATE_PREFIX_RECOVERY_PROTOCOL, **changed})

    def test_scout_continues_exact_main_hz_no_initialization_no_graph_and_restores_rng(self):
        model = CountingMemory()
        current, memory = state(), torch.zeros(2, 1)
        controls, features, grid = torch.zeros(2, 450, 4), torch.zeros(2, 1), torch.zeros(2, 16, 16)
        for index in range(16):
            output = model.transition(current, controls[:, index], features, memory)
            current, memory = output['state'], output['memory']
        expected_state, expected_memory = current.detach().clone(), memory.detach().clone()
        model.calls.clear()
        rng = torch.get_rng_state().clone()
        with patch.object(model, 'initialize_from_image', side_effect=AssertionError('No reset/RGB')):
            candidates, metrics = continue_own_prefix_candidates(model, current, memory, features, grid,
                controls, start=16, late_prefix_protocol=LATE_PREFIX_RECOVERY_PROTOCOL)
        torch.testing.assert_close(model.calls[0][0], expected_state, rtol=0, atol=0)
        torch.testing.assert_close(model.calls[0][1], expected_memory, rtol=0, atol=0)
        torch.testing.assert_close(current.detach(), expected_state, rtol=0, atol=0)
        torch.testing.assert_close(memory.detach(), expected_memory, rtol=0, atol=0)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertEqual(len(model.calls), 434)
        self.assertFalse(any(grad for _, _, grad in model.calls))
        self.assertIsNone(model.delta.grad)
        self.assertEqual([item['step'] for item in candidates], list(range(16, 450, 16)))
        for item in candidates:
            self.assertEqual(float(item['memory'][0, 0]), item['step'])
            self.assertFalse(item['state'].requires_grad or item['memory'].requires_grad)
            self.assertEqual(item['memory_provenance'], 'actualownprefix')
            self.assertEqual(item['new_observed_rgb_frames'], 0)
        self.assertEqual(candidates[-1]['prediction'].shape[1], 2)
        self.assertEqual(metrics['actor_transitions'], 868)
        self.assertEqual(metrics['memory_resets'], 0)

    def test_probe_and_selection_reserve_late_slots_without_dropping_contact_spawn_or_legality(self):
        early = [row(actor, 0) for actor in range(16)]
        late = [row(actor, 16+16*actor, 'npc_motion', contact=actor < 2, recovery=2 <= actor < 4)
                for actor in range(8)]
        ordered = order_late_prefix_probes(early+late,
            late_prefix_protocol=LATE_PREFIX_RECOVERY_PROTOCOL, budget=16)
        self.assertTrue(all(item['step'] >= 16 for item in ordered[:8]))
        selected, metrics = select_late_prefix_rows(early+late, early,
            late_prefix_protocol=LATE_PREFIX_RECOVERY_PROTOCOL, max_branches=16)
        self.assertEqual(len(selected), 16)
        self.assertEqual(metrics['late_prefix_selected_count'], 8)
        self.assertEqual(metrics['late_prefix_unfilled_count'], 0)
        self.assertGreaterEqual(metrics['late_prefix_pre_contact_count'], 2)
        self.assertGreaterEqual(metrics['late_prefix_post_spawn_count'], 2)
        self.assertTrue(any(item['step'] == 0 for item in selected))
        _, missing = select_late_prefix_rows(early+late[:2], early,
            late_prefix_protocol=LATE_PREFIX_RECOVERY_PROTOCOL, max_branches=16)
        self.assertEqual(missing['late_prefix_selected_count'], 2)
        self.assertEqual(missing['late_prefix_unfilled_count'], 6)

    def test_existing_selector_keeps_late_legal_motion_after_an_earlier_source_error(self):
        early = [row(actor, 0) for actor in range(8)]
        late = [row(actor, 96, 'npc_motion') for actor in range(8)]
        protocol = {'version': 2, 'max_branches': 16,
                    'max_source_probes': 128, 'max_direct_source_queries': 256}
        with patch('turkeyworld.rssm_combat_branches._rows', return_value=(early+late, {})), \
             patch('turkeyworld.rssm_combat_branches._source_labels', side_effect=lambda item, *_: item):
            old, _ = select_combat_branches([], combat_branch_protocol=protocol)
            new, report = select_combat_branches([], combat_branch_protocol=protocol,
                late_prefix_protocol=LATE_PREFIX_RECOVERY_PROTOCOL)
        self.assertFalse(any(item['step'] >= 16 for item in old))
        self.assertEqual(sum(item['step'] >= 16 for item in new), 8)
        self.assertEqual(report['late_prefix_unfilled_count'], 0)
        self.assertLessEqual(report['source_probes'], protocol['max_source_probes'])

    def test_entire_selector_report_is_native_json_with_numpy_boolean_source_flags(self):
        early = [row(actor, 0) for actor in range(8)]
        late = [row(actor, 96, 'npc_motion', contact=np.bool_(actor < 2),
                    recovery=np.bool_(2 <= actor < 4)) for actor in range(8)]
        protocol = {'version': 2, 'max_branches': 16,
                    'max_source_probes': 128, 'max_direct_source_queries': 256}
        with patch('turkeyworld.rssm_combat_branches._rows', return_value=(early+late, {})), \
             patch('turkeyworld.rssm_combat_branches._source_labels', side_effect=lambda item, *_: item):
            _, report = select_combat_branches([], combat_branch_protocol=protocol,
                late_prefix_protocol=LATE_PREFIX_RECOVERY_PROTOCOL)
        # Exactly the production serializer: no default hook or _json_ready.
        encoded = json.dumps(report, allow_nan=False)
        self.assertEqual(json.loads(encoded), report)
        for name, value in report.items():
            if name.startswith('late_prefix_') and name.endswith('_count'):
                self.assertIs(type(value), int, name)
        self.assertIs(type(report['late_prefix_selection_active']), bool)
        self.assertTrue(all(type(step) is int for step in report['late_prefix_selected_steps']))

    def test_main_objective_continues_its_final_own_prefix_and_rejects_future_pose_inputs(self):
        from turkeyworld.rssm_objectives import rssm_dynamics_backward
        model = CountingMemory()
        source = state()[:, None].repeat(1, 17, 1)
        batch = {'states': source, 'target_states': source[:, 1:].clone(),
            'frames': torch.zeros(2, 17, 3, 2, 2), 'actions': torch.zeros(2, 16, 4),
            'map': torch.zeros(2, 16, 16)}
        extension = {'actions': torch.zeros(2, 450, 4), 'map': batch['map'].clone()}
        captured = []
        def branches(model, candidates, **kwargs):
            captured.extend(candidates)
            self.assertEqual(kwargs['late_prefix_protocol'], LATE_PREFIX_RECOVERY_PROTOCOL)
            return {'loss': 0.}
        def geometry(prediction, *args, **kwargs):
            zero = prediction.sum()*0
            return {'ar_wall_player': zero, 'ar_wall_npc': zero}, {}
        with patch('turkeyworld.rssm_objectives._posterior_backward', return_value={'loss': 0.}), \
             patch('turkeyworld.rssm_objectives.trajectory_state_losses', side_effect=geometry), \
             patch('turkeyworld.rssm_combat_branches.combat_branches_backward', side_effect=branches):
            metrics = rssm_dynamics_backward(model, batch, context=1, expected_horizon=16,
                combat_protocol={}, combat_branch_protocol={'version': 2}, numeric_state_feedback='detached',
                late_prefix_protocol=LATE_PREFIX_RECOVERY_PROTOCOL, late_prefix_batch=extension)
        self.assertEqual(model.initializations, 1)
        self.assertEqual(captured[0]['step'], 0)
        self.assertEqual(captured[1]['step'], 16)
        torch.testing.assert_close(captured[1]['state'], captured[0]['prediction'][:, -1], rtol=0, atol=0)
        self.assertEqual(float(captured[1]['memory'][0, 0]), 16.)
        self.assertTrue(all(grad for _, _, grad in model.calls[:16]))
        self.assertFalse(any(grad for _, _, grad in model.calls[16:]))
        self.assertEqual(metrics['rssm_late_prefix_scout_new_observed_rgb_frames'], 0)
        self.assertEqual(metrics['rssm_late_prefix_scout_transition_calls'], 434)
        with self.assertRaisesRegex(ValueError, 'exact first8'):
            rssm_dynamics_backward(model, batch, context=1, expected_horizon=16,
                combat_branch_protocol={'version': 2}, numeric_state_feedback='detached',
                late_prefix_protocol=LATE_PREFIX_RECOVERY_PROTOCOL,
                late_prefix_batch={**extension, 'states': source})

    def test_real_endless_spawn_heads_restore_b64_map_caches_after_b8_scout_and_exception(self):
        from test_rssm_combat_training import raw_model
        from turkeyworld.engine import Game
        from turkeyworld.rssm_combat_protocol import environment_profile
        model = raw_model().eval()
        game = Game(seed=0, endless=True)
        current = torch.tensor(game.state_vector())[None].repeat(64, 1)
        grid = torch.tensor(game.map)[None].repeat(64, 1, 1).long()
        image = torch.tensor(game.render(size=32, overlays=False)).permute(2, 0, 1)[None].float()/255
        image = image.repeat(64, 1, 1, 1)
        controls = torch.zeros(64, 450, 4)
        with environment_profile(model, navigation_only=False), torch.no_grad():
            features = model.encode_map(grid)
            memory = model.initialize_from_image(image, current, features)
            first = model.transition(current, controls[:, 0], features, memory)
            current, memory = first['state'], first['memory']
            # Exercise both named batch-shaped caches; spawn uses the real
            # backend renderer cache and map conditioning uses its own view.
            model.register_buffer('_map_condition', grid.clone(), persistent=False)
            caches = [(module, name, getattr(module, name)) for module in model.modules()
                      for name in ('_spawn_walkable', '_map_condition')
                      if torch.is_tensor(getattr(module, name, None))]
            snapshots = [value.clone() for _, _, value in caches]
            with patch.object(model, 'initialize_from_image', side_effect=AssertionError('No reset/RGB')):
                candidates, _ = continue_own_prefix_candidates(model, current[:8], memory[:8], features[:8],
                    grid[:8], controls[:8], start=432, late_prefix_protocol=LATE_PREFIX_RECOVERY_PROTOCOL)
            self.assertEqual(candidates[0]['state'].shape[0], 8)
            for (module, name, value), before in zip(caches, snapshots):
                self.assertIs(getattr(module, name), value)
                torch.testing.assert_close(value, before, rtol=0, atol=0)
            after = model.transition(current, controls[:, 1], features, memory)
            self.assertEqual(after['state'].shape[0], 64)
            with self.assertRaisesRegex(RuntimeError, 'deliberate transition failure'):
                with patch.object(model, 'transition', side_effect=RuntimeError('deliberate transition failure')):
                    continue_own_prefix_candidates(model, current[:8], memory[:8], features[:8], grid[:8],
                        controls[:8], start=432, late_prefix_protocol=LATE_PREFIX_RECOVERY_PROTOCOL)
            for module, name, value in caches:
                self.assertIs(getattr(module, name), value)
            again = model.transition(current, controls[:, 1], features, memory)
            torch.testing.assert_close(again['state'], after['state'], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
