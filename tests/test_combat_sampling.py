"""Combat sampling uses Source supervision only, never predictor inputs."""

from turkeyworld.engine import STATE_DIM, GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS
import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from turkeyworld.combat_sampling import COMBAT_CASES, COMBAT_SAMPLING_VERSION, CombatSampling, classify_combat
from turkeyworld.motion_curriculum import MotionCurriculum


class CombatSamplingTests(unittest.TestCase):
    @staticmethod
    def payload(kind='idle', count=32, moving=True):
        states = np.zeros((count+1, STATE_DIM), dtype=np.float32)
        states[:, :2] = [2.5/16, 2.5/16]
        states[:, 3:6] = 1.
        states[:, 8:10] = [6.5/16, 2.5/16]
        states[:, 12:14] = 1.
        actions = np.zeros((count, 4), np.float32)
        if moving:
            actions[:, 0] = 1.
            states[:, 0] += np.arange(count+1)*(.08/16)
        events = np.zeros((count, 4), np.float32)
        hits = np.full(count, -1, np.int64)
        attacks = np.zeros((count, ENTITY_SLOTS), np.float32)
        tilemap = np.zeros((16, 16), np.int8)
        tilemap[0] = tilemap[-1] = 1
        tilemap[:, 0] = tilemap[:, -1] = 1
        index = min(4, count-1)
        if kind in ('hit', 'miss', 'wall_negative'):
            actions[index, 3] = events[index, 0] = 1.
        if kind == 'hit':
            events[index, 1] = 1.
            hits[index] = 0
        if kind == 'attack':
            attacks[index, 0] = events[index, 3] = 1.
        if kind == 'death':
            states[index+1:, 13] = 0
            states[index+1:, 12] = 0
        if kind == 'spawn':
            states[:, 12:14] = 0.
            states[index+1:, 8:10] = [13.5/16, 13.5/16]
            states[index+1:, 12:14] = [.6, 1.]
        if kind == 'wall_negative':
            tilemap[2, 4] = 1
        return {'states': states, 'actions': actions, 'events': events, 'hit_ids': hits,
            'attack_flags': attacks, 'map': tilemap, 'dt': np.asarray(1/30), 'split': np.asarray('train'),
            'seed': np.asarray(73), 'policy': np.asarray(kind), 'behavior_version': np.asarray('test_source'),
            'frames': np.zeros((count+1, 8, 8, 3), np.uint8)}

    @classmethod
    def data(cls, root, kinds=COMBAT_CASES, count=32, moving=True):
        directory = root/'train'; directory.mkdir()
        episodes = []
        for index, kind in enumerate(kinds):
            path = directory/f'episode_{index:03d}.npz'
            payload = cls.payload(kind, count, moving)
            payload['seed'] = np.asarray(73+index)
            np.savez_compressed(path, **payload)
            episodes.append({'path': str(path)})
        return SimpleNamespace(directory=directory, split='train', episodes=episodes)

    @staticmethod
    def masks(payload):
        return classify_combat(*(payload[name] for name in
            ('states', 'actions', 'events', 'hit_ids', 'attack_flags', 'map')), dt=float(payload['dt']))

    def test_recorded_labels_are_nonexclusive_and_preserve_slot_identity(self):
        for name in COMBAT_CASES:
            with self.subTest(name=name):
                payload = self.payload(name)
                masks = self.masks(payload)
                self.assertEqual(int(masks[name].sum()), 1)
                self.assertEqual(bool(masks['wall_negative'].any()), name == 'wall_negative')
                if name == 'wall_negative':
                    self.assertTrue(masks['miss'][4])
                if name == 'death':
                    self.assertFalse(masks['hit'].any())
        payload = self.payload('attack')
        payload['states'][5:, -2] = 1
        self.assertTrue(self.masks(payload)['attack'][4])

    def test_wall_negative_matches_real_source_shot_geometry_and_not_other_misses(self):
        from turkeyworld.engine import Game
        game = Game(73)
        game.map[:] = 0
        game.map[0] = game.map[-1] = 1
        game.map[:, 0] = game.map[:, -1] = 1
        for enemy in game.enemies:
            enemy.update(alive=False, hp=0., death_timer=.6)
        game.enemies[0].update(x=5.5, y=2.5, alive=True, hp=60., death_timer=0.)
        game.map[2, 4] = 1
        old = game.state_vector()
        action = np.asarray([0., 0., 0., 1.], np.float32)
        event = game.step(action)
        self.assertTrue(event['shot'])
        self.assertEqual(event['hit_id'], -1)
        states = np.stack([old, game.state_vector()])
        args = (states, action[None], game.event_vector(event)[None], np.asarray([-1]),
                np.zeros((1, ENTITY_SLOTS)), game.map.copy())
        # The classifier neither steps nor renders the engine, even for walls.
        with patch.object(Game, 'step', side_effect=AssertionError('runtime oracle')),\
                patch.object(Game, 'render', side_effect=AssertionError('RGB decode')):
            result = classify_combat(*args)
        self.assertTrue(result['wall_negative'][0])
        for mutation in ('open_wall', 'behind_player', 'off_crosshair', 'dead_slot', 'hit', 'no_shot'):
            with self.subTest(mutation=mutation):
                changed = [value.copy() for value in args]
                if mutation == 'open_wall': changed[-1][2, 4] = 0
                elif mutation == 'behind_player': changed[0][0, 8] = 1.5/16
                elif mutation == 'off_crosshair': changed[0][0, 9] = 3.5/16
                elif mutation == 'dead_slot': changed[0][0, 13] = 0
                elif mutation == 'hit': changed[3][0] = 0
                else: changed[2][0, 0] = 0
                self.assertFalse(classify_combat(*changed)['wall_negative'][0])

    def test_firing_pose_is_after_player_movement_and_turn_before_npc_movement(self):
        payload = self.payload('wall_negative', moving=False)
        # Previous player looked away; recorded shot pose faces the enemy.
        payload['states'][4, 2:4] = [1., 0.]
        # The enemy has moved off the firing ray in the next state.
        payload['states'][5, 9] = 5.5/16
        self.assertTrue(self.masks(payload)['wall_negative'][4])
        # A wall elsewhere in view does not turn a clear crosshair miss into
        # a wall negative. Missing map boundary tiles do not count as walls.
        payload['map'][2, 4] = 0
        payload['map'][3, 4] = 1
        self.assertFalse(self.masks(payload)['wall_negative'][4])

    def test_future_window_alignment_excludes_context_and_right_boundary(self):
        with TemporaryDirectory() as directory:
            data = self.data(Path(directory), ('idle',), count=8)
            path = Path(data.episodes[0]['path'])
            for transition in (0, 4, 5):
                payload = self.payload('idle', count=8)
                payload['events'][transition, 0] = 1
                np.savez(path, **payload)
                sampler = CombatSampling(data, context=4)
                first_window = any(start == 0 for episode, start in sampler._index(2)['miss'])
                self.assertEqual(first_window, transition == 4)
                if transition == 4:
                    self.assertFalse(any(start == 0 for episode, start in sampler._index(1)['miss']))
                if transition == 0:
                    self.assertEqual(sampler.plan(5)['available_windows']['miss'], 0)

    def test_no_rgb_or_engine_access_fixed_order_source_ids_and_content_fingerprint(self):
        from turkeyworld.engine import Game
        with TemporaryDirectory() as directory:
            data = self.data(Path(directory)); data.episodes.reverse()
            paths = [Path(episode['path']) for episode in data.episodes]
            hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
            with np.load(paths[0], allow_pickle=False) as saved: archive_type = type(saved)
            original = archive_type.__getitem__
            def small_fields(saved, name):
                self.assertNotIn(name, {'frames', 'hr_frames', 'initial_state', 'depth', 'visibility'})
                return original(saved, name)
            with patch.object(archive_type, '__getitem__', small_fields),\
                    patch.object(Game, 'step', side_effect=AssertionError('engine step')),\
                    patch.object(Game, 'render', side_effect=AssertionError('engine render')):
                sampler = CombatSampling(data, 4)
            summary = sampler.summary()
            self.assertEqual([record['path'] for record in sampler.records], [path.resolve() for path in paths])
            self.assertEqual(summary['transitions'], 32*len(COMBAT_CASES))
            self.assertEqual(summary['counts']['wall_negative'], 1)
            self.assertTrue(all(item['split'] == 'train' and item['dt'] == 1/30
                                and len(item['labels_sha256']) == 64 for item in summary['source_ids']))
            self.assertEqual(hashes, [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths])
            (data.directory/'not_selected.npz').write_bytes(b'Not part of the caller selection')
            self.assertEqual(sampler.summary(), summary)
            self.assertFalse(summary['runtime_engine_feedback'])

    def test_balanced_case_plan_redistributes_rare_and_all_empty_cases(self):
        with TemporaryDirectory() as directory:
            sampler = CombatSampling(self.data(Path(directory)), 4)
            self.assertEqual(sampler.plan(4)['case_weights'], {name: 1/len(COMBAT_CASES) for name in COMBAT_CASES})
            pairs = sampler.sample_indices(2000, 4, np.random.default_rng(173))
            selected = sampler.last_sample['selected_case_counts']
            self.assertEqual(sum(selected.values()), 2000)
            self.assertTrue(all(250 <= selected[name] <= 420 for name in COMBAT_CASES))
            self.assertTrue(all(sampler.last_sample['windows_with_case'][name] > 0 for name in COMBAT_CASES))
            self.assertEqual(pairs.shape, (2000, 2))
        for kind, expected in (('attack', {'attack': 1.}), ('idle', {'uniform': 1.})):
            with self.subTest(kind=kind), TemporaryDirectory() as directory:
                sampler = CombatSampling(self.data(Path(directory), (kind,)), 4)
                self.assertEqual(sampler.plan(4)['case_weights'], expected)

    def test_every_selected_window_stays_in_its_source_episode_and_contains_its_condition(self):
        with TemporaryDirectory() as directory:
            sampler = CombatSampling(self.data(Path(directory)), 4)
            for horizon in (1, 4, 8, 16, 29):
                for name, pairs in sampler._index(horizon).items():
                    for episode, start in pairs:
                        record = sampler.records[int(episode)]
                        self.assertLessEqual(start+4+horizon, record['frames'])
                        if name != 'uniform':
                            first = start+4-1
                            self.assertTrue(record['masks'][name][first:first+horizon].any())
                with self.assertRaises(ValueError): sampler.sample_report([[0, 33]], horizon)

    def test_persisted_rng_and_sequence_data_contract_unchanged(self):
        with TemporaryDirectory() as directory:
            data = self.data(Path(directory))
            sampler = CombatSampling(data, 4)
            random, resumed = np.random.default_rng(73), np.random.default_rng(999)
            sampler.sample_indices(7, 4, random)
            resumed.bit_generator.state = copy.deepcopy(random.bit_generator.state)
            global_random = np.random.get_state()
            np.testing.assert_array_equal(sampler.sample_indices(48, 29, random), sampler.sample_indices(48, 29, resumed))
            np.testing.assert_array_equal(global_random[1], np.random.get_state()[1])
            self.assertEqual(global_random[2:], np.random.get_state()[2:])
            from turkeyworld.jepa_training_utils import SequenceData
            source = SequenceData(data.directory, mode='stream', episode_paths=[Path(e['path']).name for e in data.episodes])
            pairs = sampler.sample_indices(3, 29, random)
            batch = source.batch(3, context=4, horizon=29, indices=pairs)
            self.assertEqual(tuple(batch['actions'].shape), (3, 29, 4))
            self.assertEqual(tuple(batch['context_frames'].shape), (3, 4, 3, 8, 8))
            self.assertNotIn('wall_negative', batch)
            self.assertNotIn('combat_sampling', batch)

    def test_cache_reuse_rejects_changed_labels_dt_and_episode_order(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); data = self.data(root)
            cache = root/'combat_cache.json'
            first = CombatSampling(data, 4, cache)
            self.assertFalse(first.cache_reused)
            with patch('turkeyworld.combat_sampling._wall_blocked_shots', side_effect=AssertionError('cached geometry')):
                second = CombatSampling(data, 4, cache)
            self.assertTrue(second.cache_reused)
            self.assertEqual(first.summary(), second.summary())
            data.episodes.reverse()
            reordered = CombatSampling(data, 4, cache)
            self.assertFalse(reordered.cache_reused)
            self.assertNotEqual(first.fingerprint, reordered.fingerprint)
            path = data.directory/f'episode_{COMBAT_CASES.index("wall_negative"):03d}.npz'
            payload = self.payload('idle'); payload['dt'] = np.asarray(1/60)
            np.savez(path, **payload)
            changed = CombatSampling(data, 4, cache)
            self.assertFalse(changed.cache_reused)
            self.assertNotEqual(reordered.fingerprint, changed.fingerprint)
            self.assertEqual(changed.counts['wall_negative'], 0)
            encoded = json.loads(cache.read_text(encoding='utf-8'))
            encoded['records'][0]['hit'] = []
            cache.write_text(json.dumps(encoded), encoding='utf-8')
            self.assertFalse(CombatSampling(data, 4, cache).cache_reused)

    def test_wrong_split_duplicate_outside_path_and_invalid_labels_rejected(self):
        for issue in ('split', 'declared_val', 'duplicate', 'outside', 'nan', 'hit_range', 'bad_dt'):
            with self.subTest(issue=issue), TemporaryDirectory() as directory:
                data = self.data(Path(directory), ('hit',))
                path = Path(data.episodes[0]['path'])
                payload = self.payload('hit')
                if issue == 'split': data.split = 'val'
                elif issue == 'duplicate': data.episodes.append(data.episodes[0])
                elif issue == 'outside':
                    other = Path(directory)/'val'; other.mkdir()
                    outside = other/path.name; np.savez(outside, **payload)
                    data.episodes[0] = {'path': str(outside)}
                else:
                    if issue == 'declared_val': payload['split'] = np.asarray('val')
                    elif issue == 'nan': payload['states'][0, 0] = np.nan
                    elif issue == 'hit_range': payload['hit_ids'][0] = ENTITY_SLOTS
                    else: payload['dt'] = np.asarray(0.)
                    np.savez(path, **payload)
                with self.assertRaises(ValueError): CombatSampling(data, 4)

    def test_motion_mixture_preserves_global_uniform_and_reports_actual_phase_effectiveness(self):
        with TemporaryDirectory() as directory:
            data = self.data(Path(directory))
            combat = CombatSampling(data, 4)
            curriculum = MotionCurriculum(data, 4, start_step=66, combat_sampling=combat)
            plan = curriculum.plan(66)
            self.assertEqual(plan['mixture_weights'], {'uniform': .25, 'motion': .35, 'combat': .4})
            self.assertAlmostEqual(plan['case_weights']['straight'], .35)
            self.assertAlmostEqual(sum(plan['case_weights'].values()), 1.)
            for offset in (0, 100, 300, 600, 1000):
                pairs = curriculum.sample_indices(441, 66+offset, np.random.default_rng(offset))
                updated = curriculum.plan(66+offset)
                counts = updated['sampled_effectiveness']
                self.assertEqual(counts['sampled_windows'], 441)
                self.assertEqual(sum(counts['selected_case_counts'].values()), 441)
                self.assertGreaterEqual(updated['case_weights']['uniform'], .25)
                self.assertTrue(np.all(pairs[:, 1]+4+updated['horizon'] <= 33))
                self.assertGreater(counts['combat_windows_with_case']['hit'], 0)

    def test_motion_empty_case_redistribution_and_disabled_legacy_sampling_are_exact(self):
        for moving, kind in ((True, 'idle'), (False, 'attack'), (False, 'idle')):
            with self.subTest(moving=moving, kind=kind), TemporaryDirectory() as directory:
                data = self.data(Path(directory), (kind,), moving=moving)
                combat = CombatSampling(data, 4)
                curriculum = MotionCurriculum(data, 4, 0, combat_sampling=combat)
                plan = curriculum.plan(0)
                self.assertAlmostEqual(sum(plan['case_weights'].values()), 1.)
                self.assertGreaterEqual(plan['case_weights']['uniform'], .25)
                if kind == 'attack':
                    self.assertEqual(plan['case_weights'], {'uniform': .25, 'combat:attack': .75})
                if not moving and kind == 'idle':
                    self.assertEqual(plan['case_weights'], {'uniform': 1.})
                legacy = MotionCurriculum(data, 4, 0)
                disabled = MotionCurriculum(data, 4, 0, combat_sampling=combat, combat_fraction=0.)
                self.assertEqual(legacy.summary(), disabled.summary())
                self.assertEqual(legacy.plan(300), disabled.plan(300))
                np.testing.assert_array_equal(legacy.sample_indices(48, 300, np.random.default_rng(73)),
                    disabled.sample_indices(48, 300, np.random.default_rng(73)))
                with self.assertRaises(ValueError): MotionCurriculum(data, 4, 0, combat_fraction=.76)
                with self.assertRaises(ValueError): MotionCurriculum(data, 3, 0, combat_sampling=combat)

    def test_spawn_case_matches_actual_source_respawn_of_persistent_npc_slot(self):
        from turkeyworld.engine import Game
        game = Game(73, endless=True)
        game.enemies[0].update(alive=False, hp=0., death_timer=game.RESPAWN_DELAY)
        old = game.state_vector()
        action = np.zeros(4, dtype=np.float32)
        event = game.step(action)
        new = game.state_vector()
        self.assertEqual(old[13], 0.)
        self.assertEqual(new[13], 1.)
        self.assertGreater(new[12], 0.)
        self.assertEqual(game.enemies[0]['id'], 0)
        with patch.object(Game, 'step', side_effect=AssertionError('No runtime engine')),\
                patch.object(Game, 'render', side_effect=AssertionError('No RGB')):
            masks = classify_combat(np.stack([old, new]), action[None], game.event_vector(event)[None],
                np.asarray([event['hit_id']], np.int64), np.zeros((1, ENTITY_SLOTS), np.float32), game.map)
        np.testing.assert_array_equal(masks['spawn'], [True])
        self.assertFalse(masks['death'][0])

    def test_rare_spawn_balances_only_forecast_windows_not_observed_context_or_exclusive_end(self):
        with TemporaryDirectory() as directory:
            data = self.data(Path(directory), ('idle',), count=8)
            path = Path(data.episodes[0]['path'])
            for transition in (0, 4, 5, 7):
                payload = self.payload('idle', count=8)
                payload['states'][:, 12:14] = 0.
                payload['states'][transition+1:, 12:14] = [.6, 1.]
                np.savez(path, **payload)
                sampler = CombatSampling(data, 4)
                self.assertEqual(sampler.counts['spawn'], 1)
                in_first = any(start == 0 for episode, start in sampler._index(2)['spawn'])
                self.assertEqual(in_first, transition == 4)
                for horizon in (1, 2, 5):
                    plan = sampler.plan(horizon)
                    self.assertEqual(plan['case_weights'], {'spawn': 1.} if
                        plan['available_windows']['spawn'] else {'uniform': 1.})
                    pairs = sampler.sample_indices(48, horizon, np.random.default_rng(73))
                    self.assertTrue(np.all(pairs[:, 1]+4+horizon <= 9))
                    if plan['enabled']:
                        self.assertEqual(sampler.sample_report(pairs, horizon)['windows_with_case']['spawn'], 48)
                self.assertEqual(sampler.plan(5)['available_windows']['spawn'], 0 if transition == 0 else 1)

    def test_old_cache_version_is_invalidated_without_motion_api_change(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); data = self.data(root, ('spawn',))
            cache = root/'combat_cache.json'
            with patch('turkeyworld.combat_sampling.COMBAT_SAMPLING_VERSION', 'source_combat_future_windows_v1'):
                old = CombatSampling(data, 4, cache)
            current = CombatSampling(data, 4, cache)
            self.assertEqual(COMBAT_SAMPLING_VERSION, 'combat_and_spawn_sampling_v2')
            self.assertFalse(current.cache_reused)
            self.assertNotEqual(current.fingerprint, old.fingerprint)
            self.assertEqual(current.plan(4)['case_weights'], {'spawn': 1.})
            self.assertEqual(json.loads(cache.read_text(encoding='utf-8'))['version'], COMBAT_SAMPLING_VERSION)
            curriculum = MotionCurriculum(data, 4, 0, combat_sampling=current)
            plan = curriculum.plan(0)
            self.assertEqual(plan['mixture_weights'], {'uniform': .25, 'motion': .35, 'combat': .4})
            self.assertAlmostEqual(plan['case_weights']['combat:spawn'], .4)


if __name__ == '__main__':
    unittest.main()
