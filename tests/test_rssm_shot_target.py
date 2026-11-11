"""Source-shot-only target CE: class and distance mass survive batch/H dilution."""
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from turkeyworld.engine import Game
from turkeyworld.rssm_combat_objectives import prepare_combat_supervision, combat_supervised_terms
from turkeyworld.structured_state import structured_transition


def source_tape(horizon=8, near_repeats=1, fire=True):
    rows = []
    for distance in [1.]*near_repeats+[3., 6., None]:
        game = Game(seed=0, endless=True, enemies_enabled=distance is not None)
        game.map[:] = 0; game.map[[0, -1], :] = 1; game.map[:, [0, -1]] = 1
        initial = game.get_state()
        initial['player'].update(x=2.5, y=2.5, angle=0.)
        initial['enemies'][0].update(x=2.5+(distance or 3.), y=2.5)
        game.set_state(initial)
        states, actions, events, hits, attacks = [game.state_vector()], [], [], [], []
        for _ in range(horizon):
            action = np.array([0., 0., 0., float(fire)], np.float32)
            event = game.step(action, dt=1/30)
            states.append(game.state_vector()); actions.append(action); events.append(Game.event_vector(event))
            hits.append(event['hit_id']); attacks.append([0 in event['attacker_ids']])
        rows.append(dict(states=states, actions=actions, events=events, hit_ids=hits, attack_flags=attacks))
    batch = {name: torch.from_numpy(np.asarray([row[name] for row in rows])) for name in rows[0]}
    batch['attack_flags'] = batch['attack_flags'].float()
    batch['target_states'] = batch['states'][:, 1:]
    return batch


def target_loss(batch, logits, mode=None):
    kwargs = {} if mode is None else {'target_supervision': mode}
    supervision = prepare_combat_supervision(batch, **kwargs)
    totals, diagnostics = {}, {}
    for index in range(supervision['horizon']):
        old, target, action = (supervision[name][:, index] for name in ('previous', 'target', 'actions'))
        prior = structured_transition(old, old, action, logits[:, index],
            old.new_zeros(len(old), 1), training=True, endless=True)
        terms, counts = combat_supervised_terms(prior, old, target, action, supervision=supervision, index=index)
        for name, term in terms.items(): totals[name] = totals.get(name, 0)+term
        for name, count in counts.items(): diagnostics[name] = diagnostics.get(name, 0)+count
    return totals, diagnostics, supervision


class ShotTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_real_source_hit_and_fired_miss_receive_gradients_nonshots_receive_exact_zero(self):
        batch = source_tape()
        for name, value in list(batch.items()):
            if value.is_floating_point(): batch[name] = value.detach().requires_grad_(True)
        logits = torch.zeros(len(batch['states']), 8, 2, requires_grad=True)
        with patch.object(Game, 'step', side_effect=AssertionError('No Source engine in objective')), \
                patch.object(Game, 'render', side_effect=AssertionError('No future RGB in objective')):
            terms, counts, sup = target_loss(batch, logits, 'shot_distance_balanced')
            terms['combat_target'].backward()
        self.assertTrue(torch.isfinite(terms['combat_target']))
        self.assertTrue(torch.equal(logits.grad[~sup['target_shot']], torch.zeros_like(logits.grad[~sup['target_shot']])))
        self.assertTrue((logits.grad[sup['target_hit']].abs().sum(-1) > 0).all())
        self.assertTrue((logits.grad[sup['target_fired_miss']].abs().sum(-1) > 0).all())
        self.assertTrue(all(value.grad is None for value in batch.values() if value.is_floating_point()))
        self.assertEqual(int(counts['combat_target_hit_near_count']), 2)
        self.assertEqual(int(counts['combat_target_hit_mid_count']), 2)
        self.assertEqual(int(counts['combat_target_hit_far_count']), 2)
        self.assertEqual(int(counts['combat_target_fired_miss_count']), 2)
        self.assertEqual(int(counts['combat_target_nonshot_ignored_count']), 24)

    def test_rare_far_hits_and_fired_misses_keep_equal_mass_across_source_batch_and_horizon(self):
        batch = source_tape(near_repeats=9)
        b, h = batch['actions'].shape[:2]
        logits = torch.zeros(b, h, 2)
        logits[:9, :, 0] = 2.; logits[9, :, 0] = 1.; logits[10, :, 0] = -2.; logits[11, :, 1] = 3.
        terms, counts, sup = target_loss(batch, logits, 'shot_distance_balanced')
        near, mid, far, miss = [F.cross_entropy(logits[i, :1], torch.tensor([label]))
                                for i, label in ((0, 0), (9, 0), (10, 0), (11, 1))]
        expected = ((near+mid+far)/3+miss)/2
        torch.testing.assert_close(terms['combat_target'], expected, rtol=1e-6, atol=1e-7)
        self.assertEqual(int(counts['combat_target_hit_near_count']), 18)
        self.assertEqual(int(counts['combat_target_hit_far_count']), 2)
        self.assertAlmostEqual(float(counts['combat_target_hit_component']), float((near+mid+far)/6), places=6)
        self.assertAlmostEqual(float(counts['combat_target_fired_miss_component']), float(miss/2), places=6)
        changed = logits.clone(); changed[~sup['target_shot']] = torch.tensor([30., -30.])
        other, _, _ = target_loss(batch, changed, 'shot_distance_balanced')
        self.assertTrue(torch.equal(terms['combat_target'], other['combat_target']))
        short = source_tape(horizon=4, near_repeats=9)
        shorter, _, _ = target_loss(short, logits[:, :4], 'shot_distance_balanced')
        torch.testing.assert_close(terms['combat_target'], shorter['combat_target'], rtol=0, atol=1e-7)

    def test_no_shots_or_absent_hit_distance_classes_are_finite_without_fabricated_labels(self):
        batch = source_tape(fire=False)
        logits = torch.randn(4, 8, 2, requires_grad=True)
        terms, _, _ = target_loss(batch, logits, 'shot_distance_balanced')
        self.assertEqual(float(terms['combat_target'].detach()), 0.)
        terms['combat_target'].backward()
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))
        for row, label in ((0, 0), (3, 1)):
            tape = source_tape()
            single = {name: value[row:row+1] for name, value in tape.items()}
            values = torch.tensor([[[.3, -.2]]]).repeat(1, 8, 1).requires_grad_(True)
            actual, _, _ = target_loss(single, values, 'shot_distance_balanced')
            expected = F.cross_entropy(values[:, 0], torch.tensor([label]))
            torch.testing.assert_close(actual['combat_target'], expected)
            actual['combat_target'].backward()
            self.assertTrue(torch.isfinite(values.grad).all())

    def test_legacy_arithmetic_exact_and_bad_source_hit_labels_fail_only_in_optin(self):
        batch = source_tape()
        logits = torch.linspace(-1., 1., 4*8*2).reshape(4, 8, 2)
        default, old_counts, _ = target_loss(batch, logits)
        explicit, explicit_counts, _ = target_loss(batch, logits, 'legacy')
        for name in default: self.assertTrue(torch.equal(default[name], explicit[name]), name)
        for name in old_counts: self.assertTrue(torch.equal(old_counts[name], explicit_counts[name]), name)
        expected = logits.new_zeros(())
        for index in range(8):
            labels = torch.where(batch['hit_ids'][:, index] < 0, 1, batch['hit_ids'][:, index]).long()
            expected = expected+F.cross_entropy(logits[:, index], labels,
                weight=logits.new_tensor([1., .2]), reduction='none').mean()/8
        self.assertTrue(torch.equal(default['combat_target'], expected))
        balanced, _, _ = target_loss(batch, logits, 'shot_distance_balanced')
        for name in default:
            if name != 'combat_target': self.assertTrue(torch.equal(default[name], balanced[name]), name)
        wrong = {name: value.clone() for name, value in batch.items()}
        wrong['events'][0, 0, 0] = 0.
        with self.assertRaisesRegex(ValueError, 'real shot'):
            prepare_combat_supervision(wrong, target_supervision='shot_distance_balanced')
        prepare_combat_supervision(wrong)


if __name__ == '__main__': unittest.main()
