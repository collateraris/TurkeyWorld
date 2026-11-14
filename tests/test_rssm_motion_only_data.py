"""Real NPZ/SequenceData checks for the new private eight-actor replay streams."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from turkeyworld.rssm_motion_only_data import MotionReplay
from turkeyworld.structured_state import STATE_DIM,ENTITY_SLOTS


class MotionOnlyDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.initial=dict(protocol='rssm_navigation_replay_v1',draws=137,
            rng=np.random.default_rng(173).bit_generator.state)
        self.nav=self.pool('original/navigation',stream='navigation',markers=(10,11,12))
        self.enemy=self.pool('original/enemy_motion',stream='enemy_motion',markers=(20,21,22))

    def write(self,path,*,stream='navigation',marker=10,transitions=180,change=None,split='train'):
        path.parent.mkdir(parents=True,exist_ok=True)
        states=np.zeros((transitions+1,STATE_DIM),np.float32)
        states[:,0]=marker/100+np.arange(transitions+1,dtype=np.float32)*.0001
        states[:,1]=.25;states[:,3]=1;states[:,4:6]=1
        if stream=='enemy_motion':
            states[:,8:10]=(.65,.75);states[:,11:14]=1
        frames=np.empty((transitions+1,32,32,3),np.uint8)
        frames[...,0]=marker;frames[...,1]=np.arange(transitions+1,dtype=np.uint8)[:,None,None];frames[...,2]=42
        values=dict(states=states,frames=frames,actions=np.zeros((transitions,4),np.float32),
            events=np.zeros((transitions,4),np.float32),hit_ids=np.full(transitions,-1,np.int64),
            attack_flags=np.zeros((transitions,ENTITY_SLOTS),np.float32),
            respawn_flags=np.zeros((transitions,ENTITY_SLOTS),np.float32),
            map=np.zeros((16,16),np.int64),split=np.asarray(split),dt=np.asarray(1/30))
        if change is not None:change(values)
        np.savez(path,**values)
        return path

    def pool(self,name,*,stream,markers=(1,),transitions=180,change=None,split='train'):
        directory=self.root/name/'train'
        for marker in markers:
            self.write(directory/f'episode_{marker:03d}.npz',stream=stream,marker=marker,
                transitions=transitions,change=change,split=split)
        return directory

    def replay(self,*,directory=None,stream='navigation',state=None,corpora=()):
        return MotionReplay((self.nav if stream=='navigation'else self.enemy)if directory is None else directory,
            deepcopy(self.initial if state is None else state),stream=stream,device='cpu',corpora=corpora)

    def receipt(self,name,navmarkers=(30,31),enemymarkers=(40,41)):
        nav=self.pool(name+'/navigation',stream='navigation',markers=navmarkers)
        enemy=self.pool(name+'/enemy_motion',stream='enemy_motion',markers=enemymarkers)
        return dict(nav_train_directory=str(nav),enemy_train_directory=str(enemy),
            episodes=[dict(path=str(path),split='train')for directory in(nav,enemy)for path in directory.glob('*.npz')])

    def assert_batches_equal(self,a,b):
        self.assertEqual(a.keys(),b.keys())
        for key in a:self.assertTrue(torch.equal(a[key],b[key]),key)

    def test_actual_npz_sequence_batches_are_eight_plus_eight_at_state0_action0(self):
        nav=self.replay();enemy=self.replay(stream='enemy_motion')
        a,ar=nav.batch(horizon=180);b,br=enemy.batch(horizon=180)
        for batch,receipt in((a,ar),(b,br)):
            self.assertEqual(batch['states'].shape,(8,181,STATE_DIM))
            self.assertEqual(batch['actions'].shape,(8,180,4))
            self.assertEqual(batch['context_frames'].shape,(8,1,3,32,32))
            self.assertEqual(batch['target_frames'].shape,(8,180,3,32,32))
            self.assertTrue(torch.equal(batch['target_states'],batch['states'][:,1:]))
            self.assertEqual(receipt['start_indices'],[0]*8)
            self.assertEqual(receipt['private_pcg_draws'],145)
            self.assertEqual((receipt['actors'],receipt['context'],receipt['horizon']),(8,1,180))
            self.assertFalse(receipt['Source_future_feedback'])
            self.assertTrue(bool(batch['actions'][...,3].eq(0).all()))
        self.assertTrue(bool(a['states'][...,12:14].eq(0).all()))
        self.assertTrue(bool(b['states'][...,12:14].eq(1).all()))

    def test_private_pcg_consumes_only_eight_episode_indices_without_global_rng_change(self):
        replay=self.replay();before=deepcopy(self.initial)
        rng=np.random.default_rng();rng.bit_generator.state=deepcopy(before['rng'])
        indices=rng.integers(3,size=8)
        global_numpy=deepcopy(np.random.get_state());global_torch=torch.get_rng_state().clone()
        batch,receipt=replay.batch(horizon=16)
        files=sorted(self.nav.glob('*.npz'))
        self.assertEqual(receipt['episode_files'],[str(files[i])for i in indices])
        self.assertEqual(replay.state_dict(),dict(protocol=before['protocol'],draws=before['draws']+8,
            rng=rng.bit_generator.state))
        self.assertEqual(before,self.initial)
        self.assertTrue(np.array_equal(global_numpy[1],np.random.get_state()[1]))
        self.assertTrue(torch.equal(global_torch,torch.get_rng_state()))
        for index,path in enumerate(receipt['episode_files']):
            marker=int(Path(path).stem.split('_')[-1])
            self.assertAlmostEqual(float(batch['states'][index,0,0]),marker/100,places=6)

    def test_restore_replays_exact_future_files_arrays_and_rng_with_inherited_draws(self):
        first=self.replay();first.batch(horizon=16);saved=first.state_dict()
        second=self.replay(state=saved)
        for horizon in(48,80,112,144,180):
            a,ar=first.batch(horizon=horizon);b,br=second.batch(horizon=horizon)
            self.assertEqual(ar,br);self.assert_batches_equal(a,b);self.assertEqual(first.state_dict(),second.state_dict())
        second.restore_state(saved)
        third=self.replay(state=saved);self.assert_batches_equal(second.batch(horizon=16)[0],third.batch(horizon=16)[0])
        self.assertEqual(set(saved),{'protocol','draws','rng'})

    def test_latest_corpus_supplies_exactly_four_and_original_supplies_four_each_stream(self):
        receipt=self.receipt('cycle1')
        for stream,original,key in(('navigation',self.nav,'nav_train_directory'),
                ('enemy_motion',self.enemy,'enemy_train_directory')):
            replay=self.replay(stream=stream,corpora=[receipt]);_,row=replay.batch(horizon=180)
            parents=[Path(path).parent for path in row['episode_files']]
            self.assertEqual(parents[:4],[original]*4);self.assertEqual(parents[4:],[Path(receipt[key])]*4)
            self.assertEqual((row['latest_policy_actors'],row['previous_replay_actors']),(4,4))
            self.assertEqual(replay.state_dict()['draws'],self.initial['draws']+8)

    def test_second_cycle_still_replays_original_four_instead_of_prior_policy_corpus(self):
        previous=self.receipt('cycle1');latest=self.receipt('cycle2',navmarkers=(50,51),enemymarkers=(60,61))
        replay=self.replay(corpora=[previous,latest])
        for _ in range(4):
            _,row=replay.batch(horizon=16)
            self.assertEqual([Path(path).parent for path in row['episode_files'][:4]],[self.nav]*4)
            self.assertEqual([Path(path).parent for path in row['episode_files'][4:]],
                [Path(latest['nav_train_directory'])]*4)

    def test_val_directory_and_val_declared_inside_train_are_rejected(self):
        val=self.root/'heldout'/'val';self.write(val/'episode.npz',split='val')
        with self.assertRaises(ValueError):self.replay(directory=val)
        renamed=self.pool('relabeled_val',stream='navigation',split='val')
        with self.assertRaises(ValueError):self.replay(directory=renamed)
        self.assertEqual(self.initial['draws'],137)

    def test_firing_or_enemy_in_navigation_rolls_back_private_rng_and_counter(self):
        for name,change in(
                ('fire',lambda v:v['actions'].__setitem__((5,3),1)),
                ('npc',lambda v:v['states'].__setitem__((slice(None),slice(12,14)),1))):
            directory=self.pool(name,stream='navigation',change=change)
            replay=self.replay(directory=directory);before=replay.state_dict()
            with self.subTest(name=name),self.assertRaises(ValueError):replay.batch(horizon=180)
            self.assertEqual(before,replay.state_dict())

    def test_quiet_enemy_hp_damage_death_or_absence_rolls_back_rng_and_counter(self):
        for name,change in(
                ('player_damage',lambda v:v['states'].__setitem__((90,4),.5)),
                ('npc_damage',lambda v:v['states'].__setitem__((90,12),.5)),
                ('npc_dead',lambda v:v['states'].__setitem__((90,13),0)),
                ('npc_absent',lambda v:v['states'].__setitem__((slice(None),slice(12,14)),0))):
            directory=self.pool(name,stream='enemy_motion',change=change)
            replay=self.replay(directory=directory,stream='enemy_motion');before=replay.state_dict()
            with self.subTest(name=name),self.assertRaises(ValueError):replay.batch(horizon=180)
            self.assertEqual(before,replay.state_dict())

    def test_incomplete_latest_corpus_and_invalid_horizon_leave_rng_unchanged(self):
        receipt=self.receipt('short');short=Path(receipt['nav_train_directory'])
        for path in short.glob('*.npz'):path.unlink()
        self.write(short/'short.npz',transitions=179)
        receipt['episodes']=[dict(path=str(path),split='train')for directory in(
            receipt['nav_train_directory'],receipt['enemy_train_directory'])for path in Path(directory).glob('*.npz')]
        replay=self.replay(corpora=[receipt]);before=replay.state_dict()
        with self.assertRaises(ValueError):replay.batch(horizon=180)
        self.assertEqual(replay.state_dict(),before)
        for invalid in(4,176,208,240,450):
            with self.assertRaises(ValueError):replay.batch(horizon=invalid)
            self.assertEqual(replay.state_dict(),before)

    def test_fixed_episode_selection_does_not_silently_discover_unpinned_npz(self):
        replay=self.replay();self.write(self.nav/'episode_999.npz',marker=99)
        _,row=replay.batch(horizon=16)
        self.assertTrue(all(Path(path).name!='episode_999.npz'for path in row['episode_files']))
        before=replay.state_dict()
        for bad in(dict(before,protocol='wrong'),dict(before,draws=-1)):
            with self.assertRaises(ValueError):replay.restore_state(bad)
            self.assertEqual(replay.state_dict(),before)

    def test_new_corpus_rejects_unlisted_additional_npz_before_sampling(self):
        receipt=self.receipt('pinned')
        self.write(Path(receipt['nav_train_directory'])/'unlisted.npz',marker=99)
        with self.assertRaises(ValueError):self.replay(corpora=[receipt])


if __name__=='__main__':unittest.main()
