"""Actual Source corner outcomes, heldout exclusion and immutable replay."""
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
from turkeyworld.navigation_recovery_data import digest
from turkeyworld.rssm_contact_data import collect as contact_collect
from turkeyworld.rssm_corner_data import (AXIS_OUTCOMES, action_for, axis_outcomes,
    collect, corner_geometry, create_plan, record_episode, _job)


def corner_pose(x=14.79, y=14.79, angle=-math.pi+.005):
    game = Game(seed=0, endless=True, enemies_enabled=False)
    saved = game.get_state()
    saved['player'].update(x=x, y=y, angle=angle)
    game.set_state(saved)
    state = game.state_vector()
    return {'state': state.tolist(), 'kind': 'test_Source_corner', **corner_geometry(game, state)}


def config(**overrides):
    result = dict(seed=160_000_000, train_poses=256, exclude_data_roots=[],
        reuse_train_roots=[], exclusion_radius=.25, replay_root=None)
    result.update(overrides)
    return result


class RSSMCornerDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temporary.name)
        cls.replay = cls.base/'replay'
        contact_collect(cls.replay, short_train=8, short_val=0, workers=1)
        cls.excluded = cls.base/'excluded'
        cls.excluded.mkdir()
        # TRAIN and VAL may share a historical corner location. New collection
        # must reject that TRAIN pose too, rather than trust its old split tag.
        train = corner_pose()
        (cls.excluded/'plan.json').write_text(json.dumps({'poses':{
            'short/train/pose_same':{**train,'kind':'corner'},
            'short/val/pose_heldout':{**train,'kind':'corner'},
            'short/train/pose_allowed':{**corner_pose(1.21,14.79),'kind':'corner'}}}),encoding='utf-8')
        (cls.excluded/'heldout_bytes.npz').write_bytes(b'heldout sentinel never read as training')

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_balanced_plan_has_real_corners_four_quadrants_and_strict_old_val_distance(self):
        old_bytes = {p:p.read_bytes() for p in self.excluded.iterdir()}
        plan = create_plan(config(exclude_data_roots=[str(self.excluded)],reuse_train_roots=[str(self.excluded)]))
        self.assertEqual(len(plan['poses']),256)
        self.assertEqual(len(plan['jobs']),2048)
        self.assertEqual(set(plan['selected_corner_quadrants'].values()),{64})
        self.assertEqual(len({tuple(np.round(np.asarray(p['state'][:2])*16,5)) for p in plan['poses'].values()}),256)
        self.assertGreaterEqual(plan['minimum_initial_xy_distance_to_old_val_tiles'],.25-1e-7)
        game = Game(seed=0,endless=True,enemies_enabled=False)
        for pose in plan['poses'].values():
            geometry = corner_geometry(game,pose['state'])
            self.assertIsNotNone(geometry)
            self.assertLessEqual(max(geometry['corner_clearances_tiles']),.080002)
            self.assertTrue(game._walkable(*(np.asarray(pose['state'][:2])*16),.2))
        reused = [p for p in plan['poses'].values() if p['kind']=='reused_Source_TRAIN_corner']
        self.assertTrue(reused)
        self.assertTrue(all(p['source_train_pose_id']!='short/train/pose_same' for p in reused))
        yaw = [math.atan2(p['state'][2],p['state'][3]) for p in plan['poses'].values()]
        self.assertTrue(any(a>math.pi-.01 for a in yaw))
        self.assertTrue(any(a<-math.pi+.01 for a in yaw))
        self.assertEqual(old_bytes,{p:p.read_bytes() for p in old_bytes})

    def test_real_source_x_then_y_covers_all_axis_outcomes_and_exact_rgb_replay(self):
        pose = corner_pose()
        expected_first = ('blocked_blocked','blocked_free','free_blocked','free_free')
        memberships=set()
        for branch in range(8):
            job={'seed':173+branch,'branch':branch}
            payload,audit=record_episode(job,pose)
            memberships.update(payload['corner_axis_outcomes'].tolist())
            if branch<4:
                first,_ = axis_outcomes(payload['axis_pass_targets'],payload['axis_command_active'],1)
                self.assertEqual(first,[expected_first[branch]])
                self.assertTrue(np.allclose(np.linalg.norm(payload['actions'][:,:2],axis=-1),1,atol=1e-6))
            self.assertEqual(payload['states'].shape,(33,20))
            self.assertEqual(payload['frames'].shape,(33,32,32,3))
            self.assertFalse(payload['object_masks'].any())
            self.assertTrue(np.array_equal(payload['frames'],payload['background_frames']))
            game=Game(seed=job['seed'],endless=True,enemies_enabled=False)
            game.set_state(json.loads(str(payload['initial_state'])))
            for index,control in enumerate(payload['actions']):
                game.step(control,dt=1/30)
                self.assertTrue(np.array_equal(game.state_vector(),payload['states'][index+1]))
            self.assertTrue(np.array_equal(game.render(size=32,overlays=False),payload['frames'][-1]))
        self.assertEqual(memberships,set(AXIS_OUTCOMES))

    def test_postturn_fullspeed_controls_are_continuous_across_signed_pi(self):
        for angle in (-math.pi+.0001,math.pi-.0001,0.,math.pi/2):
            for branch in (0,1,2,3,4,5):
                action=action_for(branch,0,[1,-1],angle)
                theta=angle+float(action[2])*2.2/30
                actual=np.asarray((math.cos(theta)*action[0]-math.sin(theta)*action[1],
                                   math.sin(theta)*action[0]+math.cos(theta)*action[1]))
                self.assertAlmostEqual(float(np.linalg.norm(action[:2])),1.,places=6)
                self.assertAlmostEqual(float(np.linalg.norm(actual)),1.,places=6)
                desired=((1,-1),(1,1),(-1,-1),(-1,1))[branch] if branch<4 else ((-1,-1) if branch==4 else (1,1))
                np.testing.assert_allclose(actual,np.asarray(desired)/math.sqrt(2),atol=2e-7)

    def test_partial_or_inactive_axes_never_get_fabricated_binary_membership(self):
        fractions=np.asarray([[1,0],[0,1],[1,1],[0,0],[.4,1],[0,1],[0,0]],np.float32)
        active=np.asarray([[1,1]]*5+[[0,1],[0,0]],bool)
        members,counts=axis_outcomes(fractions,active)
        self.assertEqual(set(members),set(AXIS_OUTCOMES))
        self.assertEqual([counts[k] for k in AXIS_OUTCOMES],[1,1,1,1])
        self.assertEqual(counts['partial_or_unclassified_both_axes'],1)
        self.assertEqual(counts['single_or_inactive_axes'],2)

    def test_composite_exact_replay_loader_and_idempotent_completion(self):
        from turkeyworld.staged_curriculum import StagedSequenceData
        before={p:(p.stat().st_mtime_ns,digest(p)) for p in self.replay.rglob('*') if p.is_file()}
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'new'
            report=collect(root,train_poses=4,workers=1,replay_root=self.replay)
            self.assertEqual(report['completed'],40)
            self.assertEqual((len(report['sampling_pools']['corner']),len(report['sampling_pools']['singlewall'])),(32,8))
            for r in report['records']:
                self.assertEqual(digest(root/r['relative_path']),r['sha256'])
                if r['pool']=='singlewall':
                    self.assertEqual((root/r['relative_path']).read_bytes(),(self.replay/r['source_relative_path']).read_bytes())
                else:
                    with np.load(root/r['relative_path']) as z:
                        self.assertEqual(r['axis_outcomes_first16'],z['corner_axis_outcomes'].tolist())
                        self.assertEqual(r['cohort'],str(z['contact_first16_cohort']))
            original={p:(p.stat().st_mtime_ns,digest(p)) for p in root.rglob('*') if p.is_file()}
            with (patch.object(Game,'render',side_effect=AssertionError('No repeated Source RGB')),
                  patch.object(Game,'step',side_effect=AssertionError('No repeated Source actions'))):
                repeated=collect(root,train_poses=4,workers=1,replay_root=self.replay)
            self.assertTrue(repeated['verified_existing'])
            self.assertEqual(original,{p:(p.stat().st_mtime_ns,digest(p)) for p in original})
            loader=StagedSequenceData(root/'short/rgb/train','cpu',mode='stream')
            batch=loader.batch(2,context=1,horizon=16,indices=np.asarray([[0,0],[7,16]]))
            self.assertEqual(tuple(batch['states'].shape),(2,17,20))
            self.assertEqual(tuple(batch['target_states'].shape),(2,16,20))
        self.assertEqual(before,{p:(p.stat().st_mtime_ns,digest(p)) for p in before})

    def test_resume_rejects_one_ulp_initial_xy_corruption_and_changed_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            collect(root,train_poses=4,workers=1)
            plan=json.loads((root/'plan.json').read_text())
            old=(root/'plan.json').read_bytes()
            for kwargs in ({'train_poses':8},{'seed':160000001},{'exclusion_radius':.1}):
                settings=dict(train_poses=4,workers=1);settings.update(kwargs)
                with self.assertRaises(ValueError): collect(root,**settings)
                self.assertEqual((root/'plan.json').read_bytes(),old)
            job=plan['jobs'][0];path=root/job['relative_path']
            with np.load(path) as z: payload={k:z[k].copy() for k in z.files}
            payload['states'][0,0]=np.nextafter(payload['states'][0,0],np.float32(1))
            np.savez_compressed(path,**payload)
            with self.assertRaisesRegex(ValueError,'immutable'):
                _job((str(root),plan['config'],plan['fingerprint'],plan['poses'][job['pose_id']],job))

    def test_impossible_holdout_exclusion_is_rejected_without_relaxation_or_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'new'
            with self.assertRaisesRegex(ValueError,'quadrant'):
                collect(root,train_poses=4,workers=1,exclude_data_roots=[self.excluded],exclusion_radius=100.)
            self.assertFalse(root.exists())

    def test_parallel_cli_uses_real_source_and_safe_process_spawn(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'parallel'
            completed=subprocess.run([sys.executable,'-X','utf8','-m','turkeyworld.rssm_corner_data',
                '--root',str(root),'--train-poses','4','--workers','2'],capture_output=True,
                text=True,encoding='utf-8',timeout=60)
            self.assertEqual(completed.returncode,0,completed.stderr)
            manifest=json.loads((root/'manifest.json').read_text())
            self.assertEqual((manifest['status'],manifest['completed']),('complete',32))
            self.assertTrue(all(manifest['corner_axis_outcome_clip_memberships'].values()))


if __name__=='__main__': unittest.main()
