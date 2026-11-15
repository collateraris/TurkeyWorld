"""Move real replay/panel/checkpoint fixtures without changing saved identities."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import test_dreamer_policy_validation as source_fixture
from turkeyworld.dreamer_policy_data import PolicyReplay,file_sha
from turkeyworld.dreamer_policy_cycle_data import PolicyCycleData
from turkeyworld.dreamer_policy_validation import build_policy_validation_panel,validate_panel
from turkeyworld.train_dreamer_policy import PolicyTraining,pinned_world
from turkeyworld.structured_state import STATE_DIM,ENTITY_SLOTS
from turkeyworld.workspace_paths import identity_path
from turkeyworld.dreamer_motion_cycle import MotionCycle


class PortablePolicyPathsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.parent=Path(self.temp.name)
        self.original=self.parent/'first_mount';self.original.mkdir();self.moved=self.parent/'ubuntu_mount'
        self.origin=r'Z:\original\TurkeyWorld'
        (self.original/'workspace_portability.json').write_text(json.dumps(dict(
            protocol='workspace_paths_v1',origin_root=self.origin)),encoding='utf-8')

    def mount(self,path):return patch('turkeyworld.workspace_paths.ROOT',path)

    def move(self):shutil.copytree(self.original,self.moved,copy_function=shutil.copy2)

    def replay_pool(self,name,marker,npc=False):
        root=self.original/name/'train';root.mkdir(parents=True)
        states=np.zeros((33,STATE_DIM),np.float32);states[:,:2]=.3;states[:,3:6]=1;states[:,6]=marker
        if npc:states[:,8:10]=.7;states[:,11:14]=1
        np.savez_compressed(root/'episode.npz',states=states,frames=np.full((33,32,32,3),marker,np.uint8),
            actions=np.zeros((32,4),np.float32),events=np.zeros((32,4),np.float32),hit_ids=np.full(32,-1,np.int64),
            attack_flags=np.zeros((32,ENTITY_SLOTS),np.float32),map=np.zeros((16,16),np.int8),split='train',dt=1/30)
        return root

    def test_relocated_real_policy_replay_accepts_old_plan_and_restores_exact_private_rng(self):
        directories=[self.replay_pool('nav',10),self.replay_pool('quiet',20,True)]
        with self.mount(self.original):
            first=PolicyReplay(directories,horizon=16,seed=173);first.batch(3);saved=deepcopy(first.state_dict())
            identities=deepcopy(first.identities);old_directories=[identity_path(value)for value in directories]
            expected=first.batch(3)
        self.move()
        with self.mount(self.moved):
            resumed=PolicyReplay(old_directories,horizon=16,seed=173,identities=identities)
            resumed.load_state_dict(saved);actual=resumed.batch(3)
        self.assertEqual(resumed.identities,identities);self.assertEqual(resumed.state_dict(),first.state_dict())
        self.assertTrue(all(row['directory'].startswith(self.origin)for row in resumed.identities))
        for key in expected:self.assertTrue(torch.equal(expected[key],actual[key]),key)

    def test_relocated_cycle_receipt_bytes_and_private_sampler_identity_remain_unchanged(self):
        nav=self.replay_pool('latest/nav',40);enemy=self.replay_pool('latest/quiet',50,True)
        receipt=self.original/'collection_receipt.json'
        with self.mount(self.original):
            value=dict(split='train',validation_unchanged=True,nav_train_directory=identity_path(nav),
                enemy_train_directory=identity_path(enemy),episodes=[dict(path=identity_path(path),sha256=file_sha(path),
                    split='train')for directory in(nav,enemy)for path in directory.glob('*.npz')])
            receipt.write_text(json.dumps(value),encoding='utf-8');raw=receipt.read_bytes();old_receipt=identity_path(receipt)
            first=PolicyCycleData(receipt,device='cpu',horizon=16,seed=173);first.data.batch(3,balanced=False)
            saved=deepcopy(first.state_dict());expected=first.data.batch(3,balanced=False);expected_state=first.state_dict()
        self.move()
        with self.mount(self.moved):
            resumed=PolicyCycleData(old_receipt,device='cpu',horizon=16,seed=173,previous=saved)
            actual=resumed.data.batch(3,balanced=False);actual_state=resumed.state_dict()
        self.assertEqual(actual_state,expected_state);self.assertEqual(resumed.receipt,value)
        self.assertEqual((self.moved/'collection_receipt.json').read_bytes(),raw)
        for key in expected:self.assertTrue(torch.equal(actual[key],expected[key]),key)

    def test_relocated_genuine_source_val_panel_keeps_old_paths_and_exact_fingerprint(self):
        data=self.original/'combat_source';records=[source_fixture.write_source(data,'navigation',100),
            source_fixture.write_source(data,'full_game',101),source_fixture.write_source(data,'moving1',102,terminal=True)]
        # Old Windows manifests can contain relative backslashes even though the
        # stored files have ordinary native directory components on Linux.
        for row in records:row['relative_path']=row['relative_path'].replace('/','\\')
        (data/'manifest.json').write_text(json.dumps(dict(records=records)),encoding='utf-8')
        with self.mount(self.original):
            old_root=identity_path(data);panel=build_policy_validation_panel(data,count=3,seed=174,include_terminal=True)
            validate_panel(panel);before=deepcopy(panel)
        self.move()
        with self.mount(self.moved):
            restored=build_policy_validation_panel(old_root,count=3,seed=174,include_terminal=True)
            validate_panel(before)
        self.assertEqual(restored,before);self.assertEqual(panel,before)
        self.assertEqual(restored['recorded_terminal_starts'],1)
        self.assertTrue(all(row['episode_path'].startswith(self.origin)for row in restored['starts']))

    def test_pinned_world_read_uses_relocated_file_but_returns_identical_old_identity(self):
        source=self.original/'world.pt'
        torch.save(dict(stage='world',training_steps=8111,global_training_step=8111,
            run_state=dict(global_training_step=8111),config=dict(architecture='dreamer_rssm'),
            model=dict(weight=torch.ones(2))),source)
        with self.mount(self.original):
            identity=pinned_world(source,self.original/'world_snapshots/paused.pt')
            old_source=identity_path(source);before=deepcopy(identity)
        self.move()
        with self.mount(self.moved):
            after=pinned_world(old_source,identity['path'])
        self.assertEqual(after,before);self.assertEqual(identity,before)
        self.assertEqual((self.moved/'world.pt').read_bytes(),source.read_bytes())

    def test_policy_world_loader_resolves_native_io_without_mutating_saved_world_identity(self):
        snapshot=self.original/'world_snapshots/paused.pt';snapshot.parent.mkdir();snapshot.write_bytes(b'pinned')
        with self.mount(self.original):old_path=identity_path(snapshot)
        self.move();identity=dict(path=old_path,sha256='a'*64,step=8111);before=deepcopy(identity)
        learner=PolicyTraining.__new__(PolicyTraining);learner.device=torch.device('cpu');learner.state={}
        model=nn.Linear(1,1);model.config=SimpleNamespace(architecture='dreamer_rssm')
        with self.mount(self.moved),patch('turkeyworld.train_dreamer_policy.load_world_checkpoint',
                return_value=(model,{}))as load,patch('turkeyworld.train_dreamer_policy.RSSMImaginationAdapter',
                return_value=SimpleNamespace()):
            learner.load_world(identity)
        load.assert_called_once_with(self.moved/'world_snapshots/paused.pt',torch.device('cpu'))
        self.assertEqual(identity,before);self.assertEqual(learner.world_identity,before)
        self.assertEqual(learner.state['world_checkpoint'],old_path)

    def test_relocated_receipt_still_rejects_modified_episode_bytes(self):
        nav=self.replay_pool('latest/nav',40);enemy=self.replay_pool('latest/quiet',50,True)
        with self.mount(self.original):
            value=dict(split='train',validation_unchanged=True,nav_train_directory=identity_path(nav),
                enemy_train_directory=identity_path(enemy),episodes=[dict(path=identity_path(path),sha256=file_sha(path),
                    split='train')for directory in(nav,enemy)for path in directory.glob('*.npz')])
        receipt=self.original/'collection_receipt.json';receipt.write_text(json.dumps(value));self.move()
        (self.moved/'latest/nav/train/episode.npz').write_bytes(b'changed')
        with self.mount(self.moved),self.assertRaises(ValueError):
            PolicyCycleData(self.origin+r'\collection_receipt.json',device='cpu',horizon=16,seed=173)

    def test_real_cycle_constructor_accepts_unchanged_old_plan_after_workspace_move(self):
        def arguments(root):
            return SimpleNamespace(run_dir=root/'cycle',world_run=root/'world',policy_run=root/'policy',
                motion_plan=root/'world/motion_only_plan.json',initial_world=root/'world/checkpoints/world.pt',
                episodes_per_stream=16,policy_updates=50,cycles=1)
        world=self.original/'world';world.mkdir();(world/'motion_only_plan.json').write_bytes(b'{"same":"plan"}')
        with self.mount(self.original):
            first=MotionCycle(arguments(self.original));first.publish()
            raw=(first.root/'cycle_plan.json').read_bytes();plan=json.loads(raw)
            self.assertEqual(plan['world_run'],self.origin+r'\world')
            self.assertEqual(plan['policy_run'],self.origin+r'\policy')
        self.move()
        with self.mount(self.moved):resumed=MotionCycle(arguments(self.moved))
        self.assertEqual((resumed.root/'cycle_plan.json').read_bytes(),raw)
        self.assertEqual(resumed.state,first.state)


if __name__=='__main__':unittest.main()
