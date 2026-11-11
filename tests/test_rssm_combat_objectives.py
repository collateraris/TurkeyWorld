import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from turkeyworld.engine import Game
from turkeyworld.rssm_combat_objectives import prepare_combat_supervision, combat_supervised_terms
from turkeyworld.structured_state import structured_transition


def source_batch(horizon=8):
    rows = []
    for kind in ('kill', 'attack', 'spawn', 'occluded'):
        game = Game(seed=0, endless=True)
        state = game.get_state()
        state['player'].update(x=2.5, y=2.5, angle=0., hp=100.)
        enemy = state['enemies'][0]
        enemy.update(x=3.3, y=2.5, hp=25. if kind == 'kill' else 60., alive=True,
                     attack_cooldown=0., death_timer=0., vx=0., vy=0.)
        if kind == 'spawn':
            enemy.update(hp=0., alive=False, death_timer=2.9)
        if kind == 'occluded':
            game.map = np.zeros((16, 16), np.uint8)
            game.map[[0, -1], :] = 1
            game.map[:, [0, -1]] = 1
            game.map[2, 3] = 1
            state['map'] = game.map.tolist()
            state['player'].update(x=2.7, y=2.7)
            enemy.update(x=3.35, y=3.29)
        game.set_state(state)
        states, actions, events, hits, attacks = [game.state_vector()], [], [], [], []
        for index in range(horizon):
            action = np.array([0, 0, 0, float(kind == 'kill')], np.float32)
            event = game.step(action, dt=1/30)
            states.append(game.state_vector())
            actions.append(action); events.append(Game.event_vector(event)); hits.append(event.get('hit_id', -1))
            attacks.append([float(0 in event.get('attacker_ids', []))])
        rows.append(dict(states=np.stack(states), actions=np.stack(actions), events=np.stack(events),
                         hit_ids=np.asarray(hits), attack_flags=np.asarray(attacks)))
    batch = {name: torch.from_numpy(np.stack([row[name] for row in rows])) for name in rows[0]}
    batch['target_states'] = batch['states'][:, 1:]
    return batch


class HeadParameters(nn.Module):
    def __init__(self):
        super().__init__()
        self.target = nn.Parameter(torch.zeros(4, 2))
        self.attack = nn.Parameter(torch.zeros(4, 1))
        self.motion = nn.Parameter(torch.full((4, 2), .0003))
        self.anchor = nn.Parameter(torch.zeros(4, 1, 9))
        self.fallback = nn.Parameter(torch.zeros(4, 1, 256))

    def prior(self, previous, action):
        candidate = previous.clone()
        candidate[:, 8:10] = candidate[:, 8:10]+self.motion
        result = structured_transition(previous, candidate, action, self.target, self.attack,
                                       training=True, endless=True)
        result.update(spawn_logits=self.fallback, spawn_anchor_logits=self.anchor,
                      spawn_fallback_logits=self.fallback)
        return result


def terms_on_tape(model, batch):
    supervision = prepare_combat_supervision(batch, context=1)
    totals, counts = {}, {}
    for index in range(supervision['horizon']):
        old, target, action = (supervision[name][:, index] for name in ('previous', 'target', 'actions'))
        terms, metrics = combat_supervised_terms(model.prior(old, action), old, target, action,
                                                 supervision=supervision, index=index)
        for name, value in terms.items():
            totals[name] = totals.get(name, 0)+value
        for name, value in metrics.items():
            counts[name] = counts.get(name, 0)+value
    return totals, counts, supervision


class RSSMCombatObjectivesTests(unittest.TestCase):
    def test_genuine_source_hit_attack_death_spawn_get_head_and_motion_gradients_without_engine_or_images(self):
        batch = source_batch()
        model = HeadParameters()
        self.assertGreater(int(batch['attack_flags'].sum()), 0)
        self.assertGreater(int((batch['hit_ids'] >= 0).sum()), 0)
        with patch('turkeyworld.engine.Game.step', side_effect=AssertionError('No engine in loss')), \
                patch('turkeyworld.engine.Game.render', side_effect=AssertionError('No image query')):
            terms, counts, supervision = terms_on_tape(model, batch)
            sum(terms.values()).backward()
        self.assertGreater(float(model.target.grad.abs().sum()), 0)
        self.assertGreater(float(model.attack.grad.abs().sum()), 0)
        self.assertGreater(float(model.motion.grad.abs().sum()), 0)
        self.assertGreater(float(model.anchor.grad.abs().sum()), 0)
        self.assertGreater(int(counts['combat_spawn_event_count']), 0)
        self.assertGreater(int(counts['combat_death_opportunity_count']), 0)
        self.assertGreater(int(supervision['attack_negative_count']), 0)
        self.assertTrue(all(not value.requires_grad for value in counts.values()))

    def test_global_attack_and_spawn_normalization_do_not_dilute_rare_events_across_horizon(self):
        short, long = source_batch(4), source_batch(8)
        short_terms, short_counts, short_sup = terms_on_tape(HeadParameters(), short)
        long_terms, long_counts, long_sup = terms_on_tape(HeadParameters(), long)
        self.assertEqual(int(short_counts['combat_spawn_event_count']), 1)
        self.assertEqual(int(long_counts['combat_spawn_event_count']), 1)
        self.assertAlmostEqual(float(short_terms['combat_spawn'].detach()), float(long_terms['combat_spawn'].detach()), places=6)
        self.assertAlmostEqual(float(short_terms['combat_attack'].detach()), float(long_terms['combat_attack'].detach()), places=6)
        # Whole-tape attack classes each carry a mean, then masses4:1.
        self.assertEqual(float(short_sup['attack_class_mass']), 5.)
        self.assertEqual(float(long_sup['attack_class_mass']), 5.)

    def test_batch_replication_preserves_every_normalized_term_and_labels_never_receive_gradients(self):
        batch = source_batch()
        for name, value in batch.items():
            if value.is_floating_point():
                batch[name] = value.detach().requires_grad_(True)
        left, _, _ = terms_on_tape(HeadParameters(), batch)
        for value in left.values():
            self.assertTrue(torch.isfinite(value))
        sum(left.values()).backward()
        self.assertTrue(all(value.grad is None for value in batch.values() if value.is_floating_point()))
        # Duplication is independent of window/chunk partitioning.
        doubled = {name: value.detach().repeat((2,)+(1,)*(value.ndim-1)) for name, value in batch.items()}
        model = HeadParameters()
        for name in ('target', 'attack', 'motion', 'anchor', 'fallback'):
            value = getattr(model, name)
            setattr(model, name, nn.Parameter(value.detach().repeat((2,)+(1,)*(value.ndim-1))))
        right, _, _ = terms_on_tape(model, doubled)
        for name in left:
            self.assertAlmostEqual(float(left[name].detach()), float(right[name].detach()), places=6, msg=name)

    def test_misaligned_source_tape_and_query_fail_before_using_labels(self):
        batch = source_batch()
        bad = {name: value.clone() for name, value in batch.items()}
        bad['target_states'][:, 0, 0] += .01
        with self.assertRaisesRegex(ValueError, 'align'):
            prepare_combat_supervision(bad)
        supervision = prepare_combat_supervision(batch)
        old, target, action = (supervision[name][:, 0] for name in ('previous', 'target', 'actions'))
        changed = old.clone(); changed[:, 0] += .01
        with self.assertRaisesRegex(ValueError, 'Source previous'):
            combat_supervised_terms(HeadParameters().prior(old, action), changed, target, action,
                                    supervision=supervision, index=0)


if __name__ == '__main__':
    unittest.main()
