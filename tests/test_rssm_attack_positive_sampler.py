"""True Source attack quotas, explicit ledger migration and exact RNG resume."""
from collections import Counter
from copy import deepcopy
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from turkeyworld.engine import Game
from turkeyworld.rssm_spawn_curriculum import (
    ATTACK_BALANCED_PROTOCOL, RecoveryCombatSampler, source_recovery_labels,
    upgrade_attack_balanced_sampler_state,
)
from test_rssm_target_sampler import shot_files, data_for


def source_files(directory):
    paths = shot_files(directory)
    for number, kind in enumerate(('open', 'near_wall', 'occluded', 'last_positive'), 400):
        game = Game(seed=number, endless=True, enemies_enabled=True)
        game.map[:] = 0; game.map[[0, -1], :] = 1; game.map[:, [0, -1]] = 1
        initial = game.get_state()
        if kind == 'occluded':
            game.map[2, 3] = 1
            initial['player'].update(x=2.7, y=2.7, angle=0.)
            initial['enemies'][0].update(x=3.35, y=3.29)
        else:
            initial['player'].update(x=6.5, y=6.5, angle=0.)
            initial['enemies'][0].update(x=7.25, y=6.5)
            if kind == 'near_wall': game.map[5, 6] = 1
        initial['enemies'][0].update(hp=60., alive=True, attack_cooldown=.6 if kind == 'last_positive' else 0.)
        if kind == 'last_positive': initial['player']['hp'] = 8.
        game.set_state(initial)
        states, actions, events, hits, attacks, clear = [game.state_vector()], [], [], [], [], []
        for tick in range(18 if kind == 'last_positive' else 32):
            clear.append([game._line_clear(game.enemies[0]['x'], game.enemies[0]['y'], game.player['x'], game.player['y'])])
            action = np.zeros(4, np.float32); event = game.step(action, 1/30)
            states.append(game.state_vector()); actions.append(action); events.append(Game.event_vector(event))
            hits.append(event['hit_id']); attacks.append([0 in event['attacker_ids']])
        states, attacks, clear = np.asarray(states), np.asarray(attacks, np.float32), np.asarray(clear, bool)
        labels = source_recovery_labels(states, attacks, 1/30)
        path = Path(directory)/f'episode_{number}_{kind}.npz'
        np.savez_compressed(path, states=states, actions=np.asarray(actions), events=np.asarray(events),
            hit_ids=np.asarray(hits), attack_flags=attacks, map=game.map, dt=np.asarray(1/30),
            source_los_clear=clear, source_hard_negative_flags=(labels['attack_negative'] & ~clear[:, 0])[:, None],
            split=np.asarray('train'), seed=number)
        paths.append(path)
    return paths


def sampler(paths, **kwargs):
    return RecoveryCombatSampler(data_for(paths), target_distance_balanced=True,
        attack_positive_balanced=True, **kwargs)


class AttackPositiveSamplerTests(unittest.TestCase):
    def test_genuine_source32_quotas_and_all_targeted_events_are_inside_future(self):
        with TemporaryDirectory() as folder:
            paths = source_files(Path(folder)/'train')
            hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
            with patch.object(Game, 'step', side_effect=AssertionError('No Source engine in sampling')), \
                    patch.object(Game, 'render', side_effect=AssertionError('No Source RGB in sampling')):
                selected = sampler(paths, context=2, horizon=16)
                pairs = selected.sample(32, np.random.default_rng(173))
            row = selected.last_sample
            self.assertEqual(row['version'], 2)
            self.assertEqual(row['quota'], {'spawn': 8, 'attack_negative': 8, 'attack_positive': 8, 'shooting_replay': 8})
            self.assertEqual(row['target_subquotas'], {'hit_near': 2, 'hit_mid': 2, 'hit_far': 2, 'fired_miss': 1, 'legacy': 1})
            self.assertEqual(row['attack_subquotas'], {'positive_near_wall': 4, 'positive_open': 4,
                'negative_occluded': 6, 'negative_other': 2})
            self.assertEqual(Counter(item['requested'] for item in row['attack_selected_windows']), row['attack_subquotas'])
            self.assertEqual(len(pairs), 32)
            self.assertEqual(row['ordinary_sampler']['sampled_windows'], 1)
            for item in row['attack_selected_windows']:
                self.assertIn([item['episode'], item['start']], pairs.tolist())
                with np.load(paths[item['episode']]) as source:
                    labels = source_recovery_labels(source['states'], source['attack_flags'], float(source['dt']))
                    ticks = item['source_transition_indices']; first = item['start']+1
                    self.assertTrue(ticks)
                    self.assertTrue(all(first <= tick < first+16 for tick in ticks))
                    mask = labels['attack_positive'] if item['requested'].startswith('positive') else labels['attack_negative']
                    self.assertTrue(mask[ticks].all())
                    if item['requested'] == 'negative_occluded':
                        self.assertTrue(source['source_hard_negative_flags'][ticks, 0].all())
                        self.assertFalse(source['source_los_clear'][ticks, 0].any())
            self.assertEqual(hashes, [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths])
            self.assertTrue(all(selected.attack_event_counts[name] > 0 for name in selected.attack_groups))

    def test_last_source_attack_is_not_lost_in_context_or_after_short_horizon(self):
        with TemporaryDirectory() as folder:
            paths = source_files(Path(folder)/'train')
            chosen = sampler(paths, context=2, horizon=16)
            episode = next(index for index, path in enumerate(paths) if 'last_positive' in path.name)
            with np.load(paths[episode]) as source:
                self.assertEqual(np.flatnonzero(source['attack_flags'][:, 0]).tolist(), [17])
                self.assertEqual(float(source['states'][-1, 4]), 0.)
            group = chosen.attack_groups['positive_open']
            self.assertEqual(group[group[:, 0] == episode].tolist(), [[episode, 1]])
            with self.assertRaises(ValueError): chosen.sample(16, np.random.default_rng(173))
            self.assertEqual(chosen.draws, 0)

    def test_explicit_upgrade_retains_old_ledger_four_micros_and_rng_resume_exact(self):
        with TemporaryDirectory() as folder:
            paths = source_files(Path(folder)/'train')
            legacy = RecoveryCombatSampler(data_for(paths), target_distance_balanced=True)
            rng = np.random.default_rng(173); legacy.sample(32, rng)
            old = deepcopy(legacy.state_dict()); rng_before = deepcopy(rng.bit_generator.state)
            new = sampler(paths)
            with self.assertRaisesRegex(ValueError, 'explicit version2'): new.restore_state(old)
            upgraded = upgrade_attack_balanced_sampler_state(old)
            self.assertEqual(old, legacy.state_dict()); self.assertEqual(rng.bit_generator.state, rng_before)
            new.restore_state(upgraded)
            misses = []
            for _ in range(4):
                new.sample(32, rng); misses.append(new.last_sample['fired_miss_subquotas'])
            state = new.state_dict()
            self.assertEqual(state['draws'], old['draws']+128)
            self.assertEqual(state['negative_draws'], old['negative_draws']+32)
            self.assertEqual(state['v2_draws'], 128); self.assertEqual(state['positive_draws'], 32)
            self.assertEqual(state['retained_legacy_ledger'], old)
            self.assertEqual(sum(state['anchor_counts'].values())-sum(old['anchor_counts'].values()), 32)
            self.assertEqual(sum(value['wall_miss'] for value in misses), 2)
            self.assertEqual(sum(value['other_miss'] for value in misses), 2)
            saved_rng = deepcopy(rng.bit_generator.state)
            expected = new.sample(32, rng); details = deepcopy(new.last_sample)
            restored = sampler(paths); restored.restore_state(state)
            rng2 = np.random.default_rng(1); rng2.bit_generator.state = saved_rng
            np.testing.assert_array_equal(expected, restored.sample(32, rng2))
            self.assertEqual(details, restored.last_sample); self.assertEqual(new.state_dict(), restored.state_dict())
            self.assertEqual(upgrade_attack_balanced_sampler_state(state), state)
            with self.assertRaisesRegex(ValueError, 'silently disable'): legacy.restore_state(state)

    def test_train_only_missing_positive_bad_labels_and_tampered_ledger_fail_closed(self):
        with TemporaryDirectory() as folder:
            paths = source_files(Path(folder)/'train'); data = data_for(paths); data.split = 'val'
            with self.assertRaises(ValueError): RecoveryCombatSampler(data, target_distance_balanced=True, attack_positive_balanced=True)
            with self.assertRaises(ValueError): RecoveryCombatSampler(data_for(paths), attack_positive_balanced=True)
            negative_only = []
            for path in paths:
                with np.load(path) as source:
                    if not source['attack_flags'].any(): negative_only.append(path)
            with self.assertRaises(ValueError): sampler(negative_only)
            valid = sampler(paths); state = valid.state_dict()
            for name in ('positive_draws', 'v2_draws', 'retained_legacy_ledger_sha256'):
                broken = deepcopy(state)
                broken[name] = 'wrong' if name.endswith('sha256') else broken[name]+1
                with self.assertRaises(ValueError): valid.restore_state(broken)
            positive = next(path for path in paths if path.name.endswith('_open.npz'))
            with np.load(positive) as source: fields = {key: source[key] for key in source.files}
            fields['states'][0, 4] = 0.
            np.savez_compressed(positive, **fields)
            with self.assertRaisesRegex(ValueError, 'eligibility'): sampler(paths)


if __name__ == '__main__': unittest.main()
