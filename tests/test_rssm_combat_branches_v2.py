from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_rssm_combat_training import raw_model
from turkeyworld.contact_objectives import CONTACT_PROTOCOL_V2
from turkeyworld.engine import Game
from turkeyworld.rssm_combat_branches import (capture_combat_candidate,
    normalize_combat_branch_protocol, select_combat_branches, combat_branches_backward,
    _source_step_at_own_state)
from turkeyworld.rssm_combat_protocol import environment_profile
from turkeyworld.rssm_loss_balance import GroupLossAccumulator
from turkeyworld.rssm_migration import configure_rssm_training


def fixture(kind='spawn', *, move=False):
    model = raw_model(); configure_rssm_training(model); model.core.eval()
    game = Game(seed=0, endless=True, enemies_enabled=True)
    saved = game.get_state()
    if kind == 'spawn':
        saved['enemies'][0].update(hp=0., alive=False, death_timer=2.84)
    else:
        found = False
        for y in range(1, 15):
            for x in range(1, 14):
                px, py = x+.5, y+.5
                if (game._walkable(px, py, .2) and game._walkable(px+.9, py, .25)
                        and game._line_clear(px, py, px+.9, py)):
                    saved['player'].update(x=px, y=py, angle=0.)
                    saved['enemies'][0].update(x=px+.9, y=py, hp=60., alive=True,
                        death_timer=0., attack_cooldown=.13)
                    found = True; break
            if found: break
        assert found
    game.set_state(saved)
    with torch.no_grad():
        # Freeze numerical movement so exact late hit/attack labels depend on
        # cooldown/actions, not an accidental random world movement proposal.
        for head in (model.vjepa_backend.delta_head, model.vjepa_backend.spatial_motion_head):
            head[-1].weight.zero_(); head[-1].bias.zero_()
        model.vjepa_backend.attack_head[-1].weight.zero_()
        model.vjepa_backend.attack_head[-1].bias.fill_(-8)
        model.vjepa_backend.hit_target_head[-1].weight.zero_()
        model.vjepa_backend.hit_target_head[-1].bias[0] = -8
        model.vjepa_backend.hit_target_head[-1].bias[1] = 8
    state = torch.tensor(game.state_vector())[None]
    grid = torch.tensor(game.map)[None].long()
    image = torch.tensor(game.render(size=32, overlays=False)).permute(2, 0, 1)[None].float()/255
    actions = torch.zeros(1, 16, 4)
    if move: actions[:, :, 0] = .5
    if kind != 'spawn': actions[:, 4, 3] = 1
    with environment_profile(model, navigation_only=False), torch.no_grad():
        fmap = model.encode_map(grid); memory = model.initialize_from_image(image, state, fmap)
        initial_memory = memory.clone(); current = state; predictions = []
        for index in range(16):
            result = model.transition(current, actions[:, index], fmap, memory)
            current, memory = result['state'], result['memory']; predictions.append(current)
    candidate = capture_combat_candidate(0, state, initial_memory, actions, grid, torch.stack(predictions, 1))
    return model, candidate


class CombatBranchV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_protocol_v1_defaults_unchanged_and_v2_budget_covers_eight_full_branches(self):
        old = normalize_combat_branch_protocol({})
        self.assertEqual((old['version'], old['max_branches']), (1, 4))
        self.assertNotIn('max_direct_source_queries', old)
        new = normalize_combat_branch_protocol({'version': 2})
        self.assertEqual((new['max_branches'], new['horizon'], new['max_direct_source_queries']), (8, 16, 128))
        self.assertEqual(new['max_source_probes'], 64)
        for bad in ({'version': 2, 'max_branches': 9}, {'version': 2, 'max_direct_source_queries': 127},
                    {'version': 1, 'local_contact_weight': .25}):
            with self.assertRaises(ValueError): normalize_combat_branch_protocol(bad)
        scaled = {'version': 2, 'max_branches': 16, 'max_direct_source_queries': 256, 'max_source_probes': 128}
        self.assertEqual(normalize_combat_branch_protocol(scaled)['max_branches'], 16)
        for key, value in (('max_branches', 17), ('max_direct_source_queries', 257), ('max_source_probes', 129)):
            with self.assertRaises(ValueError): normalize_combat_branch_protocol({**scaled, key: value})
        with self.assertRaises(ValueError): normalize_combat_branch_protocol({**scaled, 'max_direct_source_queries': 255})

    def test_late_spawn_has_direct_ce_gradient_without_changing_any_forward_or_memory(self):
        model, candidate = fixture()
        chosen, selection = select_combat_branches([candidate], combat_branch_protocol={})
        self.assertEqual(len(chosen), 1)
        self.assertFalse(chosen[0]['labels'][1, 13] >= .5)
        self.assertTrue(chosen[0]['labels'][5, 13] >= .5)
        traces = []
        original = model.transition
        def tracked(state, action, fmap, memory):
            result = original(state, action, fmap, memory)
            traces.append((state.detach().clone(), memory.detach().clone(), result['state'].detach().clone()))
            return result
        model.transition = tracked
        before = {n: v.clone() for n, v in model.state_dict().items()}
        with environment_profile(model, navigation_only=False), \
                patch('turkeyworld.rssm_combat_branches.select_combat_branches', return_value=(chosen, selection)), \
                patch.object(model, 'initialize_from_image', side_effect=AssertionError('No new RGB/initialization')), \
                patch.object(Game, 'render', side_effect=AssertionError('No Source images')):
            old = combat_branches_backward(model, [candidate], combat_branch_protocol={})
            old_trace = traces[:]; traces.clear()
            old_gradient = model.vjepa_backend.spawn_head[-1].weight.grad
            self.assertTrue(old_gradient is None or float(old_gradient.abs().sum()) == 0)
            model.zero_grad(set_to_none=True)
            new = combat_branches_backward(model, [candidate], combat_branch_protocol={'version': 2})
        self.assertEqual(len(traces), 16)
        for old_row, new_row in zip(old_trace, traces):
            for a, b in zip(old_row, new_row): torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertGreater(float(model.vjepa_backend.spawn_head[-1].weight.grad.abs().sum()), 0)
        self.assertEqual(new['direct_per_step_counts'][0]['respawn'], 0)
        self.assertGreater(new['direct_per_step_counts'][4]['respawn'], 0)
        self.assertEqual(new['direct_source_query_attempts'], 16)
        self.assertEqual(new['direct_source_step_calls'], 16)
        self.assertEqual(new['direct_late_respawn_count'], 1)
        self.assertEqual(new['direct_target_rgb_encoded'], 0)
        self.assertEqual(new['memory_resets'], 0)
        self.assertEqual(new['optimizer_steps'], 0)
        self.assertFalse(new['direct_source_feedback'])
        self.assertTrue(all(torch.equal(v, model.state_dict()[n]) for n, v in before.items()))
        self.assertTrue(all(p.grad is None for p in model.flow_renderer.parameters()))

    def test_late_hit_attack_and_labels_use_actual_own_previous_not_continuous_source_state(self):
        model, candidate = fixture('combat')
        original = _source_step_at_own_state
        queries = []
        def source_query(state, action, grid, protocol, dt):
            queries.append(state.copy())
            return original(state, action, grid, protocol, dt)
        with environment_profile(model, navigation_only=False), \
                patch('turkeyworld.rssm_combat_branches._source_step_at_own_state', side_effect=source_query), \
                patch.object(Game, 'render', side_effect=AssertionError('No Source RGB')):
            metrics = combat_branches_backward(model, [candidate], combat_branch_protocol={'version': 2})
        self.assertEqual(len(queries), 16)
        np.testing.assert_array_equal(queries[0], candidate['state'][0].numpy())
        np.testing.assert_array_equal(np.stack(queries[1:]), candidate['prediction'][0, :-1].numpy())
        self.assertEqual(metrics['direct_per_step_counts'][0]['hit'], 0)
        self.assertEqual(metrics['direct_per_step_counts'][0]['attack'], 0)
        self.assertGreater(metrics['direct_per_step_counts'][4]['hit'], 0)
        self.assertGreater(sum(row['attack'] for row in metrics['direct_per_step_counts'][1:]), 0)
        self.assertGreater(float(model.vjepa_backend.hit_target_head[-1].weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.vjepa_backend.attack_head[-1].bias.grad.abs().sum()), 0)

    def test_component_unit_scales_preserve_actual_gradient_and_local_contact_is_causal(self):
        model, candidate = fixture(move=True)
        with environment_profile(model, navigation_only=False):
            plain = combat_branches_backward(model, [candidate], combat_branch_protocol={'version': 2},
                contact_protocol=CONTACT_PROTOCOL_V2)
            gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
            model.zero_grad(set_to_none=True)
            accumulator = GroupLossAccumulator()
            combined = combat_branches_backward(model, [candidate], combat_branch_protocol={'version': 2},
                contact_protocol=CONTACT_PROTOCOL_V2, loss_groups=accumulator)
        self.assertTrue(combined['local_contact_active'])
        self.assertAlmostEqual(plain['loss'], combined['balanced_loss'], places=6)
        self.assertAlmostEqual(combined['component_spawn_raw']+combined['component_recovery_raw'], combined['loss'], places=6)
        self.assertEqual(accumulator.calls, 2)
        for n, p in model.named_parameters():
            if n in gradients: torch.testing.assert_close(p.grad, gradients[n], rtol=2e-5, atol=2e-7)

    def test_invalid_direct_geometry_is_skipped_and_never_repaired(self):
        model, candidate = fixture()
        original = _source_step_at_own_state
        attempts = [0]
        def query(state, *args):
            attempts[0] += 1
            if attempts[0] == 5: raise ValueError('invalid_direct_own_state_geometry')
            return original(state, *args)
        with environment_profile(model, navigation_only=False), \
                patch('turkeyworld.rssm_combat_branches._source_step_at_own_state', side_effect=query):
            metrics = combat_branches_backward(model, [candidate], combat_branch_protocol={'version': 2})
        self.assertEqual(metrics['direct_source_query_attempts'], 16)
        self.assertEqual(metrics['direct_source_step_calls'], 15)
        self.assertEqual(metrics['direct_source_invalid_skip_counts'], {'invalid_direct_own_state_geometry': 1})
        self.assertEqual(metrics['direct_per_step_counts'][4]['valid'], 0)
        self.assertEqual(metrics['state_resets'], 0)

    def test_v2_selection_keeps_legal_post_spawn_recovery_prefix(self):
        model, candidate = fixture()
        with environment_profile(model, navigation_only=False), torch.no_grad():
            fmap = model.encode_map(candidate['map'])
            current, memory = candidate['state'], candidate['memory']
            for index in range(16):
                result = model.transition(current, candidate['actions'][:, index], fmap, memory)
                current, memory = result['state'], result['memory']
            initial, initial_memory = current.clone(), memory.clone()
            prediction = []
            for index in range(16):
                result = model.transition(current, candidate['actions'][:, index], fmap, memory)
                current, memory = result['state'], result['memory']; prediction.append(current)
        recovery = capture_combat_candidate(16, initial, initial_memory,
            candidate['actions'], candidate['map'], torch.stack(prediction, 1))
        selected, report = select_combat_branches([candidate, recovery], combat_branch_protocol={'version': 2})
        self.assertTrue(any(row['cohort'] == 'spawn' for row in selected))
        self.assertTrue(any(row['step'] == 16 and row['cohort'] == 'post_spawn_recovery' for row in selected))
        self.assertGreater(report['selected_cohorts']['post_spawn_recovery'], 0)
        self.assertLessEqual(report['source_probes'], 64)
        json.dumps(report, allow_nan=False)

    def test_eight_branches_cover128_queries_without_diluting_rare_spawn_loss(self):
        model, candidate = fixture()
        with environment_profile(model, navigation_only=False):
            one = combat_branches_backward(model, [candidate], combat_branch_protocol={'version': 2})
            model.zero_grad(set_to_none=True)
            expanded = {**candidate, **{name: candidate[name].repeat(8, *([1]*(candidate[name].ndim-1)))
                for name in ('state', 'memory', 'actions', 'map', 'prediction')}}
            many = combat_branches_backward(model, [expanded], combat_branch_protocol={'version': 2})
        self.assertEqual(many['branches'], 8)
        self.assertEqual(many['direct_source_query_attempts'], 128)
        self.assertEqual(many['direct_source_step_calls'], 128)
        self.assertEqual(many['direct_source_respawn_count'], 8)
        self.assertTrue(all(row['valid'] == 8 for row in many['direct_per_step_counts']))
        self.assertAlmostEqual(one['component_spawn_raw'], many['component_spawn_raw'], places=5)

    def test_explicit_b64_budget_executes256_queries_with_same_event_contact_losses_and_gradient(self):
        model, candidate = fixture(move=True)
        scaled = {'version': 2, 'max_branches': 16, 'max_direct_source_queries': 256, 'max_source_probes': 128}
        with environment_profile(model, navigation_only=False):
            one = combat_branches_backward(model, [candidate], combat_branch_protocol={'version': 2},
                contact_protocol=CONTACT_PROTOCOL_V2)
            one_gradient = model.vjepa_backend.spawn_head[-1].weight.grad.clone()
            model.zero_grad(set_to_none=True)
            expanded = {**candidate, **{name: candidate[name].repeat(16, *([1]*(candidate[name].ndim-1)))
                for name in ('state', 'memory', 'actions', 'map', 'prediction')}}
            many = combat_branches_backward(model, [expanded], combat_branch_protocol=scaled,
                contact_protocol=CONTACT_PROTOCOL_V2)
        self.assertEqual(many['branches'], 16)
        self.assertEqual(many['direct_source_query_attempts'], 256)
        self.assertEqual(many['direct_source_step_calls'], 256)
        self.assertEqual(many['direct_source_respawn_count'], 16)
        self.assertTrue(all(row['valid'] == 16 for row in many['direct_per_step_counts']))
        self.assertLessEqual(many['source_probes'], 128)
        self.assertTrue(many['local_contact_active'])
        for name in ('component_spawn_raw', 'component_recovery_raw', 'local_contact_loss', 'loss'):
            self.assertAlmostEqual(one[name], many[name], places=5)
        torch.testing.assert_close(model.vjepa_backend.spawn_head[-1].weight.grad, one_gradient,
            rtol=5e-5, atol=2e-6)


if __name__ == '__main__': unittest.main()
