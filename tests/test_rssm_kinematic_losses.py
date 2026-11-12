"""Source velocity labels retain their mass across sixteen ticks and chunks."""
import math
import unittest

import torch

from turkeyworld.rssm_kinematic_losses import (KINEMATIC_PROTOCOL_V1,
    normalize_kinematic_protocol, prepare_kinematic_supervision, kinematic_supervised_terms)
from turkeyworld.rssm_loss_balance import combat_loss_groups


def tape(batch=2, horizon=16):
    previous = torch.zeros(batch, horizon, 20)
    previous[..., [0, 1, 8, 9]] = .25
    previous[..., [3, 4, 5, 12, 13]] = 1.
    actions = torch.zeros(batch, horizon, 4)
    return previous, previous.clone(), actions


def evaluate(prediction, labels, indices=None):
    terms, metrics = {}, {}
    for index in range(labels['horizon']) if indices is None else indices:
        losses, diagnostics = kinematic_supervised_terms(prediction[:, index], supervision=labels, index=index)
        for name, value in losses.items():
            terms[name] = terms.get(name, 0)+value
        for name, value in diagnostics.items():
            metrics[name] = metrics.get(name, 0)+value
    return terms, metrics


class KinematicLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def prepare(self, previous, target, actions):
        return prepare_kinematic_supervision(previous, target, actions,
            kinematic_protocol=KINEMATIC_PROTOCOL_V1)

    def test_protocol_is_opt_in_strict_and_records_units_and_weights(self):
        self.assertIsNone(normalize_kinematic_protocol(None))
        self.assertEqual(normalize_kinematic_protocol({'version': 1}), KINEMATIC_PROTOCOL_V1)
        for value in ({'version': True}, {'version': 2}, {'unknown': 1},
                      {'npc_speed_weight': float('nan')}, {'cosine_epsilon_speed': 0},
                      {'direction_min_speed': 0}, {'source_blocked_fraction': .99}):
            with self.assertRaises(ValueError):
                normalize_kinematic_protocol(value)

    def test_source_axes_balance_free_and_blocked_and_preserve_the_legal_slide(self):
        previous, target, actions = tape()
        actions[..., :2] = 1.
        command = 2.4/math.sqrt(2)
        target[..., :2] += command/480
        target[1, :, 0] = previous[1, :, 0]  # Source X blocks; Y still slides.
        labels = self.prepare(previous, target, actions)
        prediction = previous.clone().requires_grad_()  # Incorrectly stops both axes.
        terms, metrics = evaluate(prediction, labels)
        self.assertEqual(int(metrics['combat_kinematic_player_x_free_count']), 16)
        self.assertEqual(int(metrics['combat_kinematic_player_x_blocked_count']), 16)
        self.assertEqual(int(metrics['combat_kinematic_player_y_free_count']), 32)
        self.assertEqual(int(metrics['combat_kinematic_player_sliding_count']), 16)
        self.assertAlmostEqual(float(labels['player_axis_weight'].sum()), 1., places=6)
        target_velocity = float(labels['source_player_velocity'][0, 0, 0])
        expected_huber = target_velocity-.5
        self.assertAlmostEqual(float(terms['combat_kinematic_player_axes'].detach()), .75*expected_huber, places=5)
        terms['combat_kinematic_player_axes'].backward()
        self.assertTrue(bool((prediction.grad[1, :, 1] < 0).all()))
        self.assertEqual(float(prediction.grad[1, :, 0].abs().sum()), 0.)
        self.assertAlmostEqual(float(metrics['combat_kinematic_player_y_free_velocity_mae']), command, places=4)
        self.assertEqual(float(metrics['combat_kinematic_player_x_blocked_velocity_mae']), 0.)

    def test_rare_direction_uses_whole_tape_support_and_ignores_spawn_death_and_dead_player(self):
        previous, target, actions = tape(batch=3)
        target[0, 0, 8] += .7/480  # Only one Source moving-direction label.
        previous[1, 5, [12, 13]] = 0  # Source respawn teleport is not velocity.
        target[1, 5, 8] += .3
        target[1, 15, [12, 13]] = 0  # Source death transition is excluded.
        previous[2, :, 4] = 0  # Frozen world after player death is excluded.
        previous.requires_grad_(); target.requires_grad_(); actions.requires_grad_()
        labels = self.prepare(previous, target, actions)
        prediction = previous.detach().clone()
        prediction[..., 9] += .7/480  # Orthogonal speed, including stationary labels.
        prediction.requires_grad_()
        terms, metrics = evaluate(prediction, labels)
        self.assertEqual(int(metrics['combat_kinematic_npc_speed_count']), 30)
        self.assertEqual(int(metrics['combat_kinematic_npc_direction_count']), 1)
        self.assertEqual(int(metrics['combat_kinematic_npc_stationary_count']), 29)
        self.assertAlmostEqual(float(terms['combat_kinematic_npc_direction'].detach()), .25, places=6)
        speed = float(((prediction[0, 0, 9]-previous[0, 0, 9])*480).detach())
        self.assertAlmostEqual(float(terms['combat_kinematic_npc_speed'].detach()), .5*speed**2*29/30, places=5)
        sum(terms.values()).backward()
        self.assertTrue(bool(torch.isfinite(prediction.grad).all()))
        self.assertEqual(float(prediction.grad[1, 5, 8:10].abs().sum()), 0.)
        self.assertEqual(float(prediction.grad[1, 15, 8:10].abs().sum()), 0.)
        self.assertEqual(float(prediction.grad[2, :, 8:10].abs().sum()), 0.)
        self.assertIsNone(previous.grad)
        self.assertIsNone(target.grad)
        self.assertIsNone(actions.grad)
        self.assertTrue(all(not value.requires_grad for value in metrics.values()))

    def test_stationary_direction_and_zero_prediction_have_finite_loss_and_gradients(self):
        previous, target, actions = tape()
        labels = self.prepare(previous, target, actions)
        prediction = previous.clone().requires_grad_()
        terms, metrics = evaluate(prediction, labels)
        self.assertEqual(float(terms['combat_kinematic_npc_direction'].detach()), 0.)
        self.assertEqual(int(metrics['combat_kinematic_npc_direction_count']), 0)
        sum(terms.values()).backward()
        self.assertTrue(bool(torch.isfinite(prediction.grad).all()))
        # A moving Source provides a finite direction gradient from an exact stop.
        target[0, 0, 8] += .7/480
        labels = self.prepare(previous, target, actions)
        prediction = previous.clone().requires_grad_()
        terms, _ = evaluate(prediction, labels)
        terms['combat_kinematic_npc_direction'].backward()
        self.assertLess(float(prediction.grad[0, 0, 8]), 0.)
        self.assertTrue(bool(torch.isfinite(prediction.grad).all()))

    def test_replicating_batch_or_partitioning_sixteen_ticks_preserves_losses_and_gradients(self):
        previous, target, actions = tape()
        actions[..., 0] = 1.
        target[0, :, 0] += 2.4/480
        target[1, :3, 0] += 2.4/480
        target[0, ::3, 8] += .7/480
        target[1, 7:, 9] += .7/480
        labels = self.prepare(previous, target, actions)
        prediction = previous.clone()
        prediction[..., 1] += .001
        prediction[..., 8] -= .0005
        full = prediction.clone().requires_grad_()
        full_terms, full_metrics = evaluate(full, labels)
        sum(full_terms.values()).backward()
        chunked = prediction.clone().requires_grad_()
        chunk_values = {}
        for start in range(0, 16, 4):
            losses, _ = evaluate(chunked, labels, range(start, start+4))
            sum(losses.values()).backward()
            for name, value in losses.items():
                chunk_values[name] = chunk_values.get(name, 0)+float(value.detach())
        torch.testing.assert_close(full.grad, chunked.grad)
        for name, value in full_terms.items():
            self.assertAlmostEqual(float(value.detach()), chunk_values[name], places=6)
        double_labels = self.prepare(*(value.repeat(2, 1, 1) for value in (previous, target, actions)))
        double_terms, double_metrics = evaluate(prediction.repeat(2, 1, 1), double_labels)
        for name in full_terms:
            torch.testing.assert_close(full_terms[name].detach(), double_terms[name])
        for name, value in full_metrics.items():
            if name.endswith('_count'):
                self.assertEqual(int(double_metrics[name]), int(value)*2)
            else:
                torch.testing.assert_close(double_metrics[name], value)

    def test_combat_integration_is_optional_and_groups_each_new_term_once(self):
        from test_rssm_combat_objectives import source_batch, HeadParameters
        from turkeyworld.rssm_combat_objectives import prepare_combat_supervision, combat_supervised_terms
        batch = source_batch(16)
        legacy = prepare_combat_supervision(batch)
        self.assertNotIn('kinematic_supervision', legacy)
        labels = prepare_combat_supervision(batch, kinematic_protocol=KINEMATIC_PROTOCOL_V1)
        totals, metrics = {}, {}
        model = HeadParameters()
        for index in range(16):
            old, target, action = (labels[name][:, index] for name in ('previous', 'target', 'actions'))
            terms, counts = combat_supervised_terms(model.prior(old, action), old, target, action,
                                                    supervision=labels, index=index)
            for name, value in terms.items():
                totals[name] = totals.get(name, 0)+value
            for name, value in counts.items():
                metrics[name] = metrics.get(name, 0)+value
        selected = {name: value for name, value in totals.items() if name.startswith('combat_kinematic_')}
        self.assertEqual(set(selected), {'combat_kinematic_player_axes', 'combat_kinematic_npc_speed',
                                         'combat_kinematic_npc_direction'})
        grouped = combat_loss_groups(selected)
        torch.testing.assert_close(grouped['player'], selected['combat_kinematic_player_axes'])
        torch.testing.assert_close(grouped['enemy'], selected['combat_kinematic_npc_speed']
                                   +selected['combat_kinematic_npc_direction'])
        self.assertGreater(int(metrics['combat_kinematic_npc_speed_count']), 0)
        sum(selected.values()).backward()
        self.assertTrue(bool(torch.isfinite(model.motion.grad).all()))


if __name__ == '__main__':
    unittest.main()
