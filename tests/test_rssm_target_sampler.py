"""Shot/distance sampling uses genuine TRAIN future events, never clip labels."""
from collections import Counter
from copy import deepcopy
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from turkeyworld.engine import Game
from turkeyworld.rssm_spawn_curriculum import RecoveryCombatSampler
from test_rssm_spawn_curriculum import make_source_files


def shot_files(directory):
    paths = make_source_files(directory, initial_death_timer=2.95)
    for number, kind in enumerate(('near', 'mid', 'far', 'other_miss', 'wall_miss', 'last_far'), 100):
        game = Game(seed=number, endless=True, enemies_enabled=kind != 'other_miss')
        game.map[:] = 0; game.map[[0, -1], :] = 1; game.map[:, [0, -1]] = 1
        if kind == 'wall_miss': game.map[2, 4] = 1
        initial = game.get_state(); initial['player'].update(x=2.5, y=2.5, angle=0.)
        distance = {'near': 1., 'mid': 3., 'far': 6., 'last_far': 6.}.get(kind, 3.)
        initial['enemies'][0].update(x=2.5+distance, y=2.5)
        game.set_state(initial)
        states, actions, events, hits, attacks = [game.state_vector()], [], [], [], []
        horizon = 18 if kind == 'last_far' else 32
        for tick in range(horizon):
            action = np.array([0., 0., 0., float(kind != 'last_far' or tick == 17)], np.float32)
            event = game.step(action, dt=1/30)
            states.append(game.state_vector()); actions.append(action); events.append(Game.event_vector(event))
            hits.append(event['hit_id']); attacks.append([0 in event['attacker_ids']])
        path = Path(directory)/f'episode_{number}_{kind}.npz'
        np.savez_compressed(path, states=np.asarray(states), actions=np.asarray(actions), events=np.asarray(events),
            hit_ids=np.asarray(hits), attack_flags=np.asarray(attacks, np.float32), map=game.map,
            dt=np.asarray(1/30), split='train', seed=number)
        paths.append(path)
    return paths


def data_for(paths):
    return SimpleNamespace(split='train', directory=paths[0].parent, episodes=[{'path': p} for p in paths])


class TargetSamplerTests(unittest.TestCase):
    def test_true_source32_subquotas_and_shuffled_windows_contain_their_future_event(self):
        with TemporaryDirectory() as folder:
            paths = shot_files(Path(folder)/'train'); data = data_for(paths)
            hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
            with patch.object(Game, 'step', side_effect=AssertionError('No engine during sampling')), \
                    patch.object(Game, 'render', side_effect=AssertionError('No image query during sampling')):
                sampler = RecoveryCombatSampler(data, context=2, horizon=16, target_distance_balanced=True)
                pairs = sampler.sample(32, np.random.default_rng(173))
            self.assertEqual(len(pairs), 32)
            self.assertEqual(sampler.last_sample['quota'], {'spawn': 8, 'attack_negative': 8, 'ordinary_combat': 16})
            self.assertEqual(sampler.last_sample['target_subquotas'],
                {'hit_near': 3, 'hit_mid': 3, 'hit_far': 3, 'fired_miss': 3, 'legacy': 4})
            chosen = sampler.last_sample['target_selected_windows']
            self.assertEqual(Counter(row['requested'] for row in chosen),
                {'hit_near': 3, 'hit_mid': 3, 'hit_far': 3, 'wall_miss': 2, 'other_miss': 1})
            self.assertEqual(sampler.last_sample['ordinary_sampler']['sampled_windows'], 4)
            for row in chosen:
                self.assertIn([row['episode'], row['start']], pairs.tolist())
                self.assertTrue(row['source_transition_indices'])
                with np.load(paths[row['episode']]) as source:
                    first = row['start']+1
                    for tick in row['source_transition_indices']:
                        self.assertGreaterEqual(tick, first)
                        self.assertLess(tick, first+16)
                        self.assertGreater(float(source['events'][tick, 0]), .5)
                        if row['requested'].startswith('hit_'):
                            self.assertEqual(int(source['hit_ids'][tick]), 0)
                            distance = np.linalg.norm((source['states'][tick, 8:10]-source['states'][tick+1, :2])*16)
                            if row['requested'] == 'hit_near': self.assertLess(distance, 2)
                            elif row['requested'] == 'hit_mid': self.assertTrue(2 <= distance < 5)
                            else: self.assertGreaterEqual(distance, 5)
                        else: self.assertEqual(int(source['hit_ids'][tick]), -1)
            self.assertEqual(hashes, [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths])
            self.assertGreaterEqual(sampler.last_sample['target_future_event_counts']['hit_far'], 3)
            summary = sampler.summary(batch_size=32)
            self.assertTrue(summary['target_distance_balanced_sampling'])
            self.assertGreater(summary['target_source_event_counts']['wall_miss'], 0)

    def test_last_transition_event_is_indexed_at_start1_not_lost_after_h16_or_context(self):
        with TemporaryDirectory() as folder:
            paths = shot_files(Path(folder)/'train'); sampler = RecoveryCombatSampler(
                data_for(paths), context=2, horizon=16, target_distance_balanced=True)
            index = next(i for i, p in enumerate(paths) if 'last_far' in p.name)
            selected = sampler.target_groups['hit_far']
            self.assertEqual(selected[selected[:, 0] == index].tolist(), [[index, 1]])
            self.assertEqual(np.flatnonzero(sampler.target_masks[index]['hit_far']).tolist(), [17])
            with self.assertRaises(ValueError): sampler.sample(16, np.random.default_rng(1))
            self.assertEqual(sampler.draws, 0)

    def test_saved_quarter_counters_and_caller_rng_resume_exact_from_existing_sampler_state(self):
        with TemporaryDirectory() as folder:
            paths = shot_files(Path(folder)/'train'); data = data_for(paths)
            legacy = RecoveryCombatSampler(data); rng = np.random.default_rng(173)
            legacy.sample(16, rng)
            old = deepcopy(legacy.state_dict())
            sampler = RecoveryCombatSampler(data, target_distance_balanced=True)
            sampler.restore_state(old)
            sampler.sample(32, rng)
            saved, saved_rng = deepcopy(sampler.state_dict()), deepcopy(rng.bit_generator.state)
            expected = sampler.sample(32, rng); details = deepcopy(sampler.last_sample)
            restored = RecoveryCombatSampler(data, target_distance_balanced=True)
            restored.restore_state(saved); rng2 = np.random.default_rng(); rng2.bit_generator.state = saved_rng
            np.testing.assert_array_equal(expected, restored.sample(32, rng2))
            self.assertEqual(details, restored.last_sample)
            self.assertEqual(sampler.state_dict(), restored.state_dict())
            self.assertEqual(sampler.draws, old['draws']+64)
            self.assertEqual(sampler.negative_draws, old['negative_draws']+16)
            # ExplicitFalse preserves the original same-RNG B16 sequence.
            a, b = RecoveryCombatSampler(data), RecoveryCombatSampler(data, target_distance_balanced=False)
            np.testing.assert_array_equal(a.sample(16, np.random.default_rng(42)), b.sample(16, np.random.default_rng(42)))

    def test_train_provenance_source_shot_dt_and_missing_distance_groups_fail_closed(self):
        with TemporaryDirectory() as folder:
            paths = shot_files(Path(folder)/'train'); data = data_for(paths)
            data.split = 'val'
            with self.assertRaises(ValueError): RecoveryCombatSampler(data, target_distance_balanced=True)
            data.split = 'train'
            farless = [p for p in paths if 'far' not in p.name]
            with self.assertRaisesRegex(ValueError, 'hit_far'):
                RecoveryCombatSampler(data_for(farless), target_distance_balanced=True)
            path = next(p for p in paths if '_near' in p.name)
            with np.load(path) as saved: fields = {name: saved[name] for name in saved.files}
            fields['events'][0, 0] = 0.
            np.savez_compressed(path, **fields)
            with self.assertRaisesRegex(ValueError, 'genuine'):
                RecoveryCombatSampler(data, target_distance_balanced=True)
            fields['events'][0, 0] = 1.; fields['dt'] = np.asarray(0.)
            np.savez_compressed(path, **fields)
            with self.assertRaisesRegex(ValueError, 'valid dt'):
                RecoveryCombatSampler(data, target_distance_balanced=True)


if __name__ == '__main__': unittest.main()
