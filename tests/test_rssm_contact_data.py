"""Genuine contact labels, immutable records and train-only neural poses."""
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from turkeyworld.engine import BEHAVIOR_VERSION, Game
from turkeyworld.navigation_recovery_data import collect as old_collect, digest
from turkeyworld.rssm_contact_data import (BRANCH_NAMES, CASE_NAMES, CLEARANCES,
    PROFILE, VERSION, action_for, collect, contact_cases, create_plan,
    first16_cohorts, valid_own_poses)


def config(**overrides):
    values = dict(version=VERSION, behavior_version=BEHAVIOR_VERSION, profile=PROFILE,
        seed=140_000_000, short_train=2048, short_val=256, long_train=0, long_val=0,
        size=32, own_trajectories=[], own_sha256=[], exclude_data_root=None,
        excluded_plan_sha256=None, exclusion_radius=.25, clearances_tiles=list(CLEARANCES),
        branches=list(BRANCH_NAMES), contact_cases=list(CASE_NAMES))
    values.update(overrides)
    return values


class RSSMContactDataTests(unittest.TestCase):
    def test_cardinal_heading_roundtrip_keeps_plan_and_requires_exact_source_canonical_state(self):
        from turkeyworld.rssm_contact_data import _job, _source_pose
        game = Game(seed=0, endless=True, enemies_enabled=False)
        state = game.get_state()
        state['player']['angle'] = 3*math.pi/2
        game.set_state(state)
        pose = {'state': game.state_vector().tolist(), 'kind': 'wall_clearance'}
        original = np.asarray(pose['state'], np.float32)
        canonical = _source_pose(Game(seed=0, endless=True, enemies_enabled=False), original)
        self.assertFalse(np.array_equal(original, canonical))
        self.assertLess(float(np.abs(original-canonical).max()), 1e-12)
        job = dict(stage='short', split='train', index=0, seed=173, pose_id='short/train/pose_00000',
                   branch=0, frames=32, relative_path='short/rgb/train/episode_173.npz')
        with tempfile.TemporaryDirectory() as temporary:
            arguments = (temporary, pose, job, 'test_immutable_identity')
            record = _job(arguments)
            path = Path(record['path'])
            original_file = path.read_bytes()
            self.assertTrue(_job(arguments)['reused'])
            self.assertEqual(path.read_bytes(), original_file)
            self.assertTrue(np.array_equal(np.asarray(pose['state'], np.float32), original))
            with np.load(path, allow_pickle=False) as saved:
                payload = {key: saved[key].copy() for key in saved.files}
            self.assertTrue(np.array_equal(payload['states'][0], canonical))
            # Even a one-ULP XY alteration is rejected, rather than hidden by
            # the legal heading-normalization tolerance used at collection.
            payload['states'][0, 0] = np.nextafter(payload['states'][0, 0], np.float32(1))
            np.savez_compressed(path, **payload)
            with self.assertRaisesRegex(ValueError, 'immutable'):
                _job(arguments)

    def test_full_plan_has_256_distinct_train_poses_and_all_contact_bins(self):
        plan = create_plan(config())
        train = [pose for name, pose in plan['poses'].items() if '/train/' in name]
        validation = [pose for name, pose in plan['poses'].items() if '/val/' in name]
        self.assertEqual((len(train), len(validation), len(plan['jobs'])), (256, 32, 2304))
        self.assertEqual(len({tuple(np.round(np.asarray(p['state'][:2])*16, 4)) for p in train}), 256)
        bins = {(p['requested_wall_clearance_tiles'], tuple(p['wall_normal'])) for p in train
                if 'requested_wall_clearance_tiles' in p}
        self.assertEqual(len(bins), 24)
        self.assertGreaterEqual(plan['train_val_min_initial_xy_distance_tiles'], .25-1e-6)
        for pose in validation:
            self.assertNotIn(pose['kind'], ('own_valid_train', 'default_spawn_train'))
        grouped = {}
        for job in plan['jobs']:
            grouped.setdefault(job['pose_id'], []).append(job['branch'])
        self.assertTrue(all(sorted(values) == list(range(8)) for values in grouped.values()))

    def test_genuine_source_replay_labels_cohorts_and_standard_loader(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = collect(root, short_train=16, short_val=8, long_train=1, long_val=1, workers=1)
            self.assertEqual(result['completed'], 26)
            self.assertEqual(result['groups']['short/train']['distinct_initial_poses'], 2)
            for key in ('free_axes', 'blocked_axes', 'slide_steps', 'idle_steps',
                        'positive_turn_steps', 'negative_turn_steps'):
                self.assertGreater(result['audit'][key], 0, key)
            for record in result['records']:
                self.assertEqual(digest(record['path']), record['sha256'])
                with np.load(record['path'], allow_pickle=False) as saved:
                    self.assertEqual(saved['states'].shape, (record['frames']+1, 20))
                    self.assertEqual(saved['frames'].shape, (record['frames']+1, 32, 32, 3))
                    self.assertFalse(saved['object_masks'].any())
                    self.assertFalse(saved['states'][:, 12:14].any())
                    self.assertTrue(np.array_equal(saved['background_frames'], saved['frames']))
                    _, _, _, cases, _ = contact_cases(saved['states'], saved['actions'])
                    cohort, members, counts = first16_cohorts(cases)
                    self.assertEqual(record['cohort'], cohort)
                    self.assertEqual(record['sample_cohorts'], members)
                    self.assertEqual(record['first16_case_counts'], counts)
                    self.assertTrue(np.array_equal(cases, saved['contact_cases']))
                    self.assertEqual(saved['contact_window_starts_h16'][0], 0)
                    self.assertEqual(saved['contact_window_starts_h16'][-1], record['frames']-16)
            selected = next(record for record in result['records'] if record['stage'] == 'short'
                            and record['split'] == 'train' and record['branch'] == 6)
            with np.load(selected['path'], allow_pickle=False) as saved:
                game = Game(seed=selected['seed'], endless=True, enemies_enabled=False)
                game.set_state(json.loads(str(saved['initial_state'])))
                for index, controls in enumerate(saved['actions']):
                    game.step(controls, dt=1/30)
                    self.assertTrue(np.array_equal(game.state_vector(), saved['states'][index+1]))
                self.assertTrue(np.array_equal(game.render(size=32, overlays=False), saved['frames'][-1]))
            from turkeyworld.staged_curriculum import StagedSequenceData
            loader = StagedSequenceData(root/'short/rgb/train', 'cpu', mode='stream')
            batch = loader.batch(2, context=1, horizon=16, indices=np.asarray([[0, 0], [6, 16]]))
            self.assertEqual(tuple(batch['states'].shape), (2, 17, 20))
            self.assertEqual(tuple(batch['target_states'].shape), (2, 16, 20))
            self.assertEqual(tuple(batch['target_object_masks'].shape), (2, 16, 1, 32, 32))

    def test_train_own_batched_states_reject_invalid_without_repair_or_source_steps(self):
        game = Game(seed=0, endless=True, enemies_enabled=False)
        good = game.state_vector()
        wall, nonfinite, hp, enemy = [good.copy() for _ in range(4)]
        wall[:2], nonfinite[0], hp[4], enemy[12:14] = 0, np.nan, 1.5, [.6, 1.]
        values = np.stack([good, good, wall, nonfinite, hp, enemy]).reshape(2, 3, 20)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'own.npz'
            np.savez(path, predicted=values, split=np.asarray('train'),
                     panel=np.asarray(json.dumps({'split': 'train', 'source_optimizer_step': 339})))
            before = path.read_bytes()
            with (patch.object(Game, 'step', side_effect=AssertionError('No source trajectory during pose validation')),
                  patch.object(Game, 'render', side_effect=AssertionError('No rendering during pose validation'))):
                poses, audit = valid_own_poses(path)
            self.assertEqual((audit['accepted_states'], audit['rejected_states'], audit['duplicate_xy_states']), (2, 4, 1))
            self.assertEqual(len(poses), 1)
            self.assertTrue(np.array_equal(poses[0]['state'], good))
            self.assertEqual(path.read_bytes(), before)
            for fields in ({}, {'split': np.asarray('val')},
                           {'split': np.asarray('train'), 'panel': np.asarray('{"split":"val"}')}):
                np.savez(path, predictions=values, **fields)
                with self.assertRaisesRegex(ValueError, 'TRAIN'):
                    valid_own_poses(path)

    def test_old_holdout_and_new_holdout_neighborhoods_are_excluded_and_old_files_immutable(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            old = base/'old'
            old_collect(old, short_train=16, short_val=16, long_train=0, long_val=0)
            original = {path: (path.stat().st_mtime_ns, digest(path)) for path in old.rglob('*') if path.is_file()}
            game = Game(seed=0, endless=True, enemies_enabled=False)
            saved = game.get_state()
            saved['player']['x'] = 2.8
            game.set_state(saved)
            own = base/'own.npz'
            np.savez(own, predicted=game.state_vector()[None], split=np.asarray('train'))
            root = base/'new'
            report = collect(root, short_train=40, short_val=16, workers=1,
                             own_trajectories=[own], exclude_data_root=old)
            self.assertEqual(report['own_training_poses_used'], 1)
            old_plan = json.loads((old/'plan.json').read_text(encoding='utf-8'))
            new_plan = json.loads((root/'plan.json').read_text(encoding='utf-8'))
            for name, pose in new_plan['poses'].items():
                split = name.split('/')[1]
                if pose['kind'] in ('own_valid_train', 'default_spawn_train'):
                    self.assertEqual(split, 'train')
                opposite = 'val' if split == 'train' else 'train'
                point = np.asarray(pose['state'][:2])*16
                for old_name, old_pose in old_plan['poses'].items():
                    if '/'+opposite+'/' in old_name:
                        self.assertGreaterEqual(np.linalg.norm(point-np.asarray(old_pose['state'][:2])*16), .25-1e-6)
            self.assertEqual(original, {path: (path.stat().st_mtime_ns, digest(path)) for path in original})

    def test_resume_never_renders_and_changed_plan_is_rejected_without_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collect(root, short_train=8, short_val=8, workers=1)
            original = {path: (path.stat().st_mtime_ns, digest(path)) for path in root.rglob('*.npz')}
            with (patch.object(Game, 'step', side_effect=AssertionError('No repeated source step')),
                  patch.object(Game, 'render', side_effect=AssertionError('No repeated source RGB'))):
                resumed = collect(root, short_train=8, short_val=8, workers=1)
            self.assertTrue(all(record['reused'] for record in resumed['records']))
            self.assertEqual(original, {path: (path.stat().st_mtime_ns, digest(path)) for path in original})
            plan_bytes = (root/'plan.json').read_bytes()
            for values in ({'short_train': 16}, {'seed': 140000001}, {'exclusion_radius': .1}):
                args = dict(short_train=8, short_val=8, workers=1)
                args.update(values)
                with self.assertRaisesRegex(ValueError, 'immutable'):
                    collect(root, **args)
                self.assertEqual((root/'plan.json').read_bytes(), plan_bytes)

    def test_back_away_idle_return_and_first16_membership_are_real_future_labels(self):
        values = np.stack([action_for(7, index) for index in range(32)])
        np.testing.assert_array_equal(values[:8, 0], -np.ones(8))
        np.testing.assert_array_equal(values[8:12], np.zeros((4, 4)))
        np.testing.assert_array_equal(values[12:28, 0], np.ones(16))
        cases = np.zeros((32, len(CASE_NAMES)), bool)
        cases[20, CASE_NAMES.index('blocked')] = True
        cases[4, CASE_NAMES.index('free')] = True
        self.assertEqual(first16_cohorts(cases)[1], ['free'])
        cases[5, CASE_NAMES.index('blocked')] = True
        cases[5, CASE_NAMES.index('slide')] = True
        self.assertEqual(first16_cohorts(cases)[0], 'slide')

    def test_windows_spawn_cli_parallel_collection_has_no_import_recursion(self):
        with tempfile.TemporaryDirectory() as temporary:
            completed = subprocess.run([sys.executable, '-X', 'utf8', '-m', 'turkeyworld.rssm_contact_data',
                '--root', temporary, '--short-train', '8', '--short-val', '0', '--workers', '2'],
                capture_output=True, text=True, encoding='utf-8', timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            manifest = json.loads((Path(temporary)/'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual((manifest['status'], manifest['completed']), ('complete', 8))


if __name__ == '__main__':
    unittest.main()
