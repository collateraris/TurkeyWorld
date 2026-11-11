"""Paired Source wall labels, strict split isolation and immutable byte replay."""
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from turkeyworld.engine import Game
from turkeyworld.navigation_recovery_data import collect as recovery_collect, digest
from turkeyworld.rssm_contact_data import collect as contact_collect
from turkeyworld.contact_refresh_data import (BRANCH_NAMES, _job, action_for,
    collect, create_plan, record_episode, wall_pair_geometry)


class ContactRefreshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = tempfile.TemporaryDirectory()
        cls.base = Path(cls.fixture.name)
        cls.recovery = cls.base/'recovery'
        cls.contact = cls.base/'contact_v1'
        recovery_collect(cls.recovery, short_train=16, short_val=8, long_train=0, long_val=0)
        contact_collect(cls.contact, short_train=16, short_val=8,
                        exclude_data_root=cls.recovery, workers=1)
        game = Game(seed=0, endless=True, enemies_enabled=False)
        values = []
        for x, angle in ((2.5, math.pi-.005), (3.5, -math.pi+.005)):
            state = game.get_state()
            state['player'].update(x=x, y=1.205, angle=angle)
            game.set_state(state)
            values.append(game.state_vector().copy())
        cls.own = cls.base/'own_train.npz'
        np.savez(cls.own, predicted=np.asarray(values)[:, None], maps=game.map,
                 split=np.asarray('train'), panel=np.asarray(json.dumps({
                     'split': 'train', 'source_optimizer_step': 2000,
                     'future_source_feedback': False, 'state_resets': 0})))

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    def arguments(self, **values):
        args = dict(own_trajectory=self.own, source_contact_root=self.contact,
            train_poses=2, replay_clips=4, workers=1,
            exclude_data_roots=[self.recovery, self.contact])
        args.update(values)
        return args

    def test_exact_paired_approach_tangents_and_away_are_real_source_labels(self):
        game = Game(seed=0, endless=True, enemies_enabled=False)
        with np.load(self.own, allow_pickle=False) as data:
            vector = data['predicted'][0, 0]
        pair = wall_pair_geometry(game, vector)
        self.assertIsNotNone(pair)
        self.assertEqual(pair['wall_normal'], [0., -1.])
        pose = {'state': vector.tolist(), 'kind': 'unchanged_valid_TRAIN_own', **pair}
        initial_states = []
        for branch in range(4):
            payload, _ = record_episode({'seed': 173, 'branch': branch}, pose)
            initial_states.append(payload['states'][0])
            active = payload['axis_command_active'][:8]
            actual_fraction = payload['axis_pass_targets'][:8][active]
            self.assertTrue((actual_fraction <= .05).all() if branch == 0
                            else (actual_fraction >= .95).all(), BRANCH_NAMES[branch])
            self.assertEqual(payload['frames'].shape, (33, 32, 32, 3))
            self.assertFalse(payload['object_masks'].any())
            self.assertTrue(np.array_equal(payload['frames'], payload['background_frames']))
            replay = Game(seed=173, endless=True, enemies_enabled=False)
            replay.set_state(json.loads(str(payload['initial_state'])))
            for step, action in enumerate(payload['actions']):
                replay.step(action)
                self.assertTrue(np.array_equal(replay.state_vector(), payload['states'][step+1]))
            self.assertTrue(np.array_equal(replay.render(size=32, overlays=False), payload['frames'][-1]))
        self.assertTrue(all(np.array_equal(initial_states[0], state) for state in initial_states))

    def test_world_relative_controls_use_post_turn_heading_across_both_pi_wraps(self):
        for angle, branch in ((math.pi-.005, 6), (-math.pi+.005, 7)):
            normal = np.asarray((0., -1.))
            sign = 1 if branch == 6 else -1
            tangent = np.asarray((1., 0.))
            expected = normal+.35*sign*tangent
            expected /= np.linalg.norm(expected)
            action = action_for(branch, 0, normal, angle)
            post_angle = angle+float(action[2])*2.2/30
            realized = np.asarray((math.cos(post_angle)*action[0]-math.sin(post_angle)*action[1],
                                   math.sin(post_angle)*action[0]+math.cos(post_angle)*action[1]))
            np.testing.assert_allclose(realized, expected, atol=1e-7, rtol=0)
            self.assertEqual(np.sign(action[2]), sign)
        np.testing.assert_array_equal(action_for(9, 0, (0, -1), 0), np.zeros(4))
        self.assertLess(action_for(8, 0, (0, -1), 0)[1], 0)
        self.assertGreater(action_for(8, 8, (0, -1), 0)[1], 0)
        np.testing.assert_array_equal(action_for(8, 24, (0, -1), 0), np.zeros(4))

    def test_immutable_refresh_replays_exact_train_bytes_and_keeps_both_old_holdouts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/'fresh'
            originals = {path: (path.stat().st_mtime_ns, digest(path))
                for old in (self.recovery, self.contact) for path in old.rglob('*') if path.is_file()}
            report = collect(root, **self.arguments(val_poses=1))
            self.assertEqual((report['completed'], report['groups']['short/train']['new_Source_clips'],
                              report['groups']['short/train']['exact_replay_clips']), (34, 20, 4))
            self.assertEqual(report['old_validation_neighborhoods_excluded'], 2)
            self.assertFalse(report['heldout_inputs_used_for_train'])
            self.assertGreaterEqual(report['train_val_min_initial_xy_distance_tiles'], .25-1e-6)
            plan = json.loads((root/'plan.json').read_text(encoding='utf-8'))
            for name, pose in plan['poses'].items():
                if '/train/' not in name:
                    continue
                point = np.asarray(pose['state'][:2])*16
                for old in (self.recovery, self.contact):
                    old_plan = json.loads((old/'plan.json').read_text(encoding='utf-8'))
                    for old_name, old_pose in old_plan['poses'].items():
                        if '/val/' in old_name:
                            self.assertGreaterEqual(np.linalg.norm(point-np.asarray(old_pose['state'][:2])*16), .25-1e-6)
            for record in report['records']:
                self.assertEqual(digest(record['path']), record['sha256'])
                if record['origin'] == 'replay_v1_TRAIN':
                    source = self.contact/record['source_relative_path']
                    self.assertEqual(Path(record['path']).read_bytes(), source.read_bytes())
                    self.assertEqual(record['source_sha256'], record['sha256'])
                    self.assertEqual(record['split'], 'train')
            self.assertEqual(originals, {path: (path.stat().st_mtime_ns, digest(path)) for path in originals})
            from turkeyworld.staged_curriculum import StagedSequenceData
            loader = StagedSequenceData(root/'short/rgb/train', 'cpu', mode='stream')
            batch = loader.batch(2, context=1, horizon=16, indices=np.asarray([[0, 0], [1, 16]]))
            self.assertEqual(tuple(batch['target_states'].shape), (2, 16, 20))

    def test_completed_resume_preserves_manifest_and_files_without_engine_calls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/'fresh'
            collect(root, **self.arguments())
            original = {path: (path.stat().st_mtime_ns, digest(path)) for path in root.rglob('*') if path.is_file()}
            with (patch.object(Game, 'step', side_effect=AssertionError('No Source on completed resume')),
                  patch.object(Game, 'render', side_effect=AssertionError('No RGB on completed resume'))):
                resumed = collect(root, **self.arguments())
            self.assertTrue(resumed['verified_existing'])
            self.assertEqual(original, {path: (path.stat().st_mtime_ns, digest(path)) for path in original})
            for change in ({'train_poses': 3}, {'seed': 150000001}, {'exclusion_radius': .1}):
                with self.assertRaisesRegex(ValueError, 'immutable'):
                    collect(root, **self.arguments(**change))
            self.assertEqual(original, {path: (path.stat().st_mtime_ns, digest(path)) for path in original})

    def test_explicit_augmentation_fills384_with_all_normals_and_both_pi_yaws(self):
        with tempfile.TemporaryDirectory() as temporary:
            # Obtain the exact public configuration without generating a large
            # image corpus. Planning itself uses no raster rendering.
            tiny = collect(Path(temporary)/'tiny', **self.arguments(train_poses=0, replay_clips=0))
            values = {**tiny['config'], 'train_poses': 384, 'allow_source_augmentation': True}
            with patch.object(Game, 'render', side_effect=AssertionError('Planning does not render')):
                plan = create_plan(values)
            self.assertEqual((len(plan['poses']), len(plan['jobs'])), (384, 3840))
            self.assertEqual(sum(plan['new_train_pose_kinds'].values()), 384)
            self.assertEqual(plan['new_train_pose_kinds']['unchanged_valid_TRAIN_own'], 2)
            normals = {tuple(pose['wall_normal']) for pose in plan['poses'].values()}
            self.assertEqual(normals, {(1., 0.), (-1., 0.), (0., 1.), (0., -1.)})
            yaws = [pose['yaw_target_radians'] for pose in plan['poses'].values() if 'yaw_target_radians' in pose]
            self.assertTrue(any(angle > 3.1 for angle in yaws))
            self.assertTrue(any(angle < -3.1 for angle in yaws))
            self.assertTrue(all(pose['kind'] != 'repaired_own' for pose in plan['poses'].values()))

    def test_invalid_or_heldout_own_inputs_never_become_repaired_train_seeds(self):
        game = Game(seed=0, endless=True, enemies_enabled=False)
        invalid = game.state_vector()
        invalid[:2] = 0
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            own = base/'bad.npz'
            np.savez(own, predicted=invalid[None], split=np.asarray('train'))
            with self.assertRaisesRegex(ValueError, 'legal paired own poses'):
                collect(base/'strict', **self.arguments(own_trajectory=own))
            self.assertFalse((base/'strict').exists())
            augmented = collect(base/'augmented', **self.arguments(own_trajectory=own,
                                train_poses=1, replay_clips=0, allow_source_augmentation=True))
            self.assertEqual(augmented['own_pose_audit']['accepted_states'], 0)
            self.assertEqual(augmented['new_train_pose_kinds'], {'independent_Source_wall_augmentation': 1})
            np.savez(own, predicted=game.state_vector()[None], split=np.asarray('val'))
            with self.assertRaisesRegex(ValueError, 'TRAIN'):
                collect(base/'heldout', **self.arguments(own_trajectory=own, allow_source_augmentation=True))
            self.assertFalse((base/'heldout').exists())

    def test_replay_copy_fallback_is_atomic_and_recovers_without_source_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            tiny = collect(base/'tiny', **self.arguments(train_poses=0, replay_clips=0))
            values = {**tiny['config'], 'replay_clips': 1}
            plan = create_plan(values)
            job = plan['jobs'][0]
            source = self.contact/job['source_relative_path']
            original = source.read_bytes()
            destination = base/'replay'
            arguments = (str(destination), values, plan['fingerprint'], None, job)
            def interrupted_copy(source_path, temporary_path):
                Path(temporary_path).write_bytes(b'incomplete')
                raise OSError('test interruption during cross-volume copy')
            with (patch('turkeyworld.contact_refresh_data.os.link', side_effect=OSError('No hardlinks')),
                  patch('turkeyworld.contact_refresh_data.shutil.copy2', side_effect=interrupted_copy)):
                with self.assertRaises(OSError):
                    _job(arguments)
            self.assertFalse((destination/job['relative_path']).exists())
            self.assertEqual(list(destination.rglob('*.tmp')), [])
            with patch('turkeyworld.contact_refresh_data.os.link', side_effect=OSError('No hardlinks')):
                report = _job(arguments)
            self.assertEqual(report['storage'], 'copy')
            self.assertEqual(Path(report['path']).read_bytes(), original)
            self.assertEqual(source.read_bytes(), original)

    def test_parallel_cli_collects_source_with_safe_windows_spawn(self):
        with tempfile.TemporaryDirectory() as temporary:
            completed = subprocess.run([sys.executable, '-X', 'utf8', '-m', 'turkeyworld.contact_refresh_data',
                '--root', str(Path(temporary)/'fresh'), '--own-trajectory', str(self.own),
                '--source-contact-root', str(self.contact), '--exclude-data-root', str(self.recovery),
                '--train-poses', '1', '--replay-clips', '1', '--workers', '2'],
                capture_output=True, text=True, encoding='utf-8', timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads((Path(temporary)/'fresh/manifest.json').read_text(encoding='utf-8'))
            self.assertEqual((result['status'], result['completed']), ('complete', 11))


if __name__ == '__main__':
    unittest.main()
