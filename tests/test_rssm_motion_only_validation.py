"""A fixed VAL identity covers files, schema and the actual evaluator panel."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

from turkeyworld.behavior_trajectory import evaluate_state_trajectories
from turkeyworld.jepa_training_utils import SequenceData
from turkeyworld.rssm_motion_only_validation import (
    PROTOCOL,bind_motion_validation_plan,verify_motion_validation_plan,file_sha,
)
from turkeyworld.structured_state import STATE_DIM,ENTITY_SLOTS,STATE_SCHEMA_VERSION


class StationaryWorld(nn.Module):
    def __init__(self):
        super().__init__();self.bias=nn.Parameter(torch.zeros(()));self.config=SimpleNamespace(simulation_hz=30)
    def encode_map(self,grid):return torch.zeros((len(grid),1),device=grid.device)
    def initialize_from_image(self,frames,states,features):return torch.zeros((len(states),1),device=states.device)
    def transition(self,current,action,features,memory):return dict(state=current,memory=memory)


class MotionOnlyValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.nav=self.root/'navigation'/'val';self.enemy=self.root/'quiet'/'val'
        for directory,npc in((self.nav,False),(self.enemy,True)):
            directory.mkdir(parents=True)
            for index in range(9):self.write(directory/f'episode_{index:03d}.npz',index=index,npc=npc)
        self.path=self.root/'motion_validation_plan.json'

    def write(self,path,*,index=0,npc=False,split='val',change=None):
        states=np.zeros((181,STATE_DIM),np.float32);states[:,:2]=(.25,.25);states[:,3:6]=1
        if npc:states[:,8:10]=(.65,.75);states[:,11:14]=1
        values=dict(states=states,frames=np.full((181,32,32,3),index,np.uint8),
            actions=np.zeros((180,4),np.float32),events=np.zeros((180,4),np.float32),
            hit_ids=np.full(180,-1,np.int64),attack_flags=np.zeros((180,ENTITY_SLOTS),np.float32),
            respawn_flags=np.zeros((180,ENTITY_SLOTS),np.float32),map=np.zeros((16,16),np.int8),
            split=np.asarray(split),dt=np.asarray(1/30),state_schema_version=np.asarray(STATE_SCHEMA_VERSION),
            state_dim=np.asarray(STATE_DIM),entity_slots=np.asarray(ENTITY_SLOTS))
        if change is not None:change(values)
        np.savez_compressed(path,**values)

    def bind(self,**kwargs):return bind_motion_validation_plan(self.nav,self.enemy,self.path,**kwargs)

    def test_atomic_create_then_identical_bind_preserve_bytes_and_do_not_touch_rng(self):
        nrng=deepcopy(np.random.get_state());trng=torch.get_rng_state().clone()
        plan=self.bind();before=self.path.read_bytes();stat=self.path.stat()
        self.assertTrue(verify_motion_validation_plan(plan));self.assertEqual(self.bind(),plan)
        self.assertEqual(self.path.read_bytes(),before);self.assertEqual(self.path.stat().st_mtime_ns,stat.st_mtime_ns)
        self.assertEqual(plan['validation_plan_sha256'],file_sha(self.path));self.assertEqual(plan['kind'],PROTOCOL)
        self.assertEqual(plan['settings']['horizon'],180);self.assertEqual(plan['settings']['count'],8)
        self.assertTrue(np.array_equal(nrng[1],np.random.get_state()[1]));self.assertTrue(torch.equal(trng,torch.get_rng_state()))
        self.assertFalse(list(self.root.glob('*.tmp')))

    def test_exact_panel_and_fingerprint_match_real_state_evaluator_including_int8_map_conversion(self):
        plan=self.bind();model=StationaryWorld()
        for key in('navigation','enemy_motion'):
            cohort=plan[key]
            data=SequenceData(cohort['directory'],split='val',image_size=32,mode='stream',
                episode_paths=cohort['episode_paths'])
            report=evaluate_state_trajectories(model,data,self.root/'results'/key,
                context=1,horizon=180,count=8,batch_size=4,seed=173,include_respawned_slots=False)
            self.assertEqual(report['panel'],cohort['panel'])
            self.assertEqual(report['fingerprint'],cohort['expected_fingerprint'])
            self.assertEqual(len(cohort['files']),9);self.assertEqual(len(cohort['panel']['windows']),8)
            self.assertEqual(cohort['files'][0]['array_schema']['map']['dtype'],'|i1')

    def test_even_unselected_file_content_change_blocks_verify_and_never_rewrites_plan(self):
        plan=self.bind();before=self.path.read_bytes()
        selected={row['episode']for row in plan['navigation']['panel']['windows']}
        unselected=next(row for row in plan['navigation']['files']if row['episode']not in selected)
        self.write(Path(unselected['path']),index=79)
        with self.assertRaises(ValueError):verify_motion_validation_plan(plan)
        with self.assertRaises(ValueError):self.bind()
        self.assertEqual(self.path.read_bytes(),before)

    def test_added_removed_or_renamed_files_cannot_change_ordered_val_identity(self):
        plan=self.bind();before=self.path.read_bytes()
        extra=self.nav/'episode_extra.npz';self.write(extra)
        with self.assertRaises(ValueError):verify_motion_validation_plan(plan)
        extra.unlink()
        original=self.nav/'episode_000.npz';renamed=self.nav/'renamed.npz';original.rename(renamed)
        with self.assertRaises(ValueError):verify_motion_validation_plan(plan)
        renamed.rename(original);original.unlink()
        with self.assertRaises(ValueError):self.bind()
        self.assertEqual(self.path.read_bytes(),before)

    def test_train_path_or_train_declaration_is_rejected_before_any_plan_is_written(self):
        train=self.root/'train';train.mkdir();self.write(train/'episode.npz',split='train')
        with self.assertRaises(ValueError):bind_motion_validation_plan(train,self.enemy,self.path)
        self.assertFalse(self.path.exists())
        self.write(self.nav/'episode_000.npz',split='train')
        with self.assertRaises(ValueError):self.bind()
        self.assertFalse(self.path.exists())

    def test_wrong_state_schema_shape_or_nonfinite_labels_are_rejected(self):
        target=self.nav/'episode_000.npz'
        for change in(
                lambda v:v.update(state_schema_version=np.asarray('other')),
                lambda v:v.update(states=v['states'][:,:-1]),
                lambda v:v['states'].__setitem__((0,0),np.nan)):
            self.write(target,change=change)
            with self.assertRaises(ValueError):self.bind()
            self.assertFalse(self.path.exists())

    def test_seed_and_file_plan_tampering_are_detected_without_silent_repin(self):
        plan=self.bind();before=self.path.read_bytes()
        with self.assertRaises(ValueError):self.bind(seed=174)
        changed=deepcopy(plan);changed['navigation']['panel']['windows'][0]['start']=1
        with self.assertRaises(ValueError):verify_motion_validation_plan(changed)
        raw=json.loads(before);raw['settings']['count']=7;self.path.write_text(json.dumps(raw),encoding='utf-8')
        with self.assertRaises(ValueError):verify_motion_validation_plan(plan)
        with self.assertRaises(ValueError):self.bind()
        self.assertNotEqual(self.path.read_bytes(),before)

    def test_incomplete_episode_support_and_same_cohort_are_rejected(self):
        for path in list(self.nav.glob('*.npz'))[7:]:path.unlink()
        with self.assertRaises(ValueError):self.bind()
        with self.assertRaises(ValueError):bind_motion_validation_plan(self.enemy,self.enemy,self.path)
        self.assertFalse(self.path.exists())


if __name__=='__main__':unittest.main()
