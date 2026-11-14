"""Serial handoffs and recovery, with no real subprocess, CUDA or live files."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import numpy as np
import torch

from turkeyworld.dreamer_motion_cycle import MotionCycle,atomic,read,sha
from turkeyworld.dreamer_policy_cycle_data import PolicyCycleData
from turkeyworld.dreamer_policy_data import PolicyReplay
from turkeyworld.train_dreamer_policy import PolicyTraining,pinned_world
from turkeyworld.structured_state import STATE_DIM,ENTITY_SLOTS


class MotionCycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        world=self.root/'world';policy=self.root/'policy'
        for directory in(world,policy):
            (directory/'checkpoints').mkdir(parents=True)
        (world/'checkpoints/rssm_world.pt').write_bytes(b'complete world boundary')
        (policy/'checkpoints/policy.pt').write_bytes(b'complete actor boundary')
        atomic(world/'status.json',dict(status='stopped',global_training_step=6873))
        atomic(policy/'status.json',dict(status='stopped',actor_updates=100,step=200))
        plan=world/'motion_plan.json';atomic(plan,dict(kind='motion test plan'))
        self.args=SimpleNamespace(run_dir=self.root/'cycle',world_run=world,policy_run=policy,
            motion_plan=plan,initial_world=world/'checkpoints/rssm_world.pt',episodes_per_stream=2,
            policy_updates=50,cycles=1)
        self.popen=self.enterContext(patch('turkeyworld.dreamer_motion_cycle.subprocess.Popen',
            side_effect=AssertionError('No actual process launch is allowed in CPU tests')))
        self.alive=self.enterContext(patch('turkeyworld.dreamer_motion_cycle.alive',return_value=False))
        self.enterContext(patch('turkeyworld.dreamer_motion_cycle.time.sleep',return_value=None))

    def controller(self):return MotionCycle(self.args)

    def test_exact_stage_order_and_finished_round_is_not_quality_pass(self):
        cycle=self.controller();order=[]
        cycle.collect=Mock(side_effect=lambda directory:order.append('collect_source'))
        cycle.world=Mock(side_effect=lambda directory:order.append('world_training'))
        cycle.policy=Mock(side_effect=lambda directory:order.append('policy_training'))
        cycle.run()
        self.assertEqual(order,['collect_source','world_training','policy_training'])
        self.assertEqual((cycle.state['completed_rounds'],cycle.state['round_index']),(1,1))
        self.assertEqual((cycle.state['status'],cycle.state['phase']),('completed','collect_source'))
        plan=read(cycle.root/'cycle_plan.json')
        self.assertEqual(plan['horizons'],[16,48,80,112,144,180]);self.assertEqual(plan['updates_per_horizon'],30)
        self.assertEqual(plan['world_updates_per_round'],180);self.assertFalse(plan['combat_loss'])
        self.popen.assert_not_called()

    def test_resume_at_policy_phase_skips_finished_collection_and_world(self):
        cycle=self.controller();cycle.state.update(phase='policy_training',initial_flags_retired=True)
        cycle.publish();restored=self.controller()
        restored.collect=Mock(side_effect=AssertionError('Source stage must not repeat'))
        restored.world=Mock(side_effect=AssertionError('World round must not repeat'))
        restored.policy=Mock();restored.run()
        restored.collect.assert_not_called();restored.world.assert_not_called();restored.policy.assert_called_once()
        self.assertEqual(restored.state['completed_rounds'],1)

    def test_initial_authorized_stop_retirement_never_removes_later_user_stops(self):
        cycle=self.controller()
        world_stop=self.args.world_run/'stop_training';policy_stop=self.args.policy_run/'stop.request'
        world_stop.write_text('old initial stop');policy_stop.write_text('old initial stop')
        cycle.retire_initial_flags()
        self.assertFalse(world_stop.exists());self.assertFalse(policy_stop.exists())
        world_stop.write_text('NEW user stop');policy_stop.write_text('NEW user stop')
        restored=self.controller();restored.retire_initial_flags()
        self.assertEqual(world_stop.read_text(),'NEW user stop');self.assertEqual(policy_stop.read_text(),'NEW user stop')

    def test_controller_stop_marker_blocks_stage_relaunch_after_restart(self):
        cycle=self.controller();cycle.state.update(status='stopped',initial_flags_retired=True);cycle.publish()
        (cycle.root/'stop.request').write_text('user stopped')
        restored=self.controller();restored.collect=Mock();restored.world=Mock();restored.policy=Mock()
        with self.assertRaises(InterruptedError):restored.run()
        restored.collect.assert_not_called();restored.world.assert_not_called();restored.policy.assert_not_called()
        self.popen.assert_not_called()

    def test_execute_checks_existing_stop_before_any_child_launch(self):
        cycle=self.controller();directory=cycle.root/'one';directory.mkdir()
        (cycle.root/'stop.request').write_text('stop before launch')
        with self.assertRaises(InterruptedError):
            cycle.execute(directory,'world',['fake_world'],finished=lambda:False,progress=lambda:None)
        self.popen.assert_not_called()

    def test_existing_world_or_policy_user_stop_is_not_automatically_restarted(self):
        cycle=self.controller();directory=cycle.root/'one';directory.mkdir()
        for name,flag in(('world',self.args.world_run/'stop_training'),('policy',self.args.policy_run/'stop.request')):
            flag.write_text('a new user stop')
            with self.subTest(name=name),self.assertRaises(InterruptedError):
                cycle.execute(directory,name,['fake_child'],finished=lambda:False,progress=lambda:None)
            flag.unlink()
        self.popen.assert_not_called()

    def test_child_incomplete_exit_is_reported_once_without_retry_loop(self):
        cycle=self.controller();directory=cycle.root/'one';directory.mkdir()
        fake=SimpleNamespace(pid=12345,wait=Mock(return_value=1))
        self.popen.side_effect=None;self.popen.return_value=fake
        with self.assertRaises(RuntimeError):
            cycle.execute(directory,'world',['fake_world'],finished=lambda:False,progress=lambda:None)
        self.assertEqual(self.popen.call_count,1);fake.wait.assert_called_once()
        self.assertIsNone(cycle.state['active_pid'])

    def test_finished_handoff_shortcircuits_stale_pid_without_query_or_relaunch(self):
        cycle=self.controller();directory=cycle.root/'one';directory.mkdir()
        atomic(directory/'world_launch.json',dict(pid=12345));progress=Mock()
        self.alive.side_effect=AssertionError('A completed stage must not adopt a reused PID')
        cycle.execute(directory,'world',['fake_world'],finished=lambda:True,progress=progress)
        progress.assert_called_once();self.popen.assert_not_called();self.alive.assert_not_called()

    def test_policy_resume_uses_persisted_absolute_actor_target_not_another_fifty(self):
        cycle=self.controller();cycle.state.update(phase='policy_training',source_receipt='immutable_receipt.json',
            trained_world_sha256=sha(self.args.world_run/'checkpoints/rssm_world.pt'))
        directory=cycle.root/'one';directory.mkdir();commands=[]
        def execute_once(directory,name,command,*,finished,progress):
            commands.append(command)
            atomic(self.args.policy_run/'status.json',dict(status='completed',actor_updates=150,step=270,
                world_sha256=cycle.state['trained_world_sha256']))
            progress();self.assertTrue(finished())
        cycle.execute=Mock(side_effect=execute_once);cycle.policy(directory)
        self.assertEqual((cycle.state['policy_actor_start'],cycle.state['policy_actor_target']),(100,150))
        self.assertIn('--refresh-world-now',commands[0]);self.assertIn('--cycle-corpus',commands[0])
        self.assertEqual(commands[0][commands[0].index('--until-actor-updates')+1],'150')
        cycle.publish();restored=self.controller()
        def execute_completed(directory,name,command,*,finished,progress):
            self.assertTrue(finished());self.assertEqual(command[command.index('--until-actor-updates')+1],'150')
        restored.execute=Mock(side_effect=execute_completed);restored.policy(directory)
        self.assertEqual(restored.state['policy_actor_target'],150)

    def test_cached_source_receipt_is_verified_and_does_not_draw_or_launch_again(self):
        cycle=self.controller();directory=cycle.root/'cycles/cycle_000000';source=directory/'source';source.mkdir(parents=True)
        assets=cycle.pin_assets(directory);episodes=[]
        policy_sha=sha(assets/'policy.pt');world_sha=sha(assets/'world.pt')
        for stream in('navigation','enemy_motion'):
            train=source/stream/'train';train.mkdir(parents=True)
            for index in range(2):
                path=train/f'episode_{index}.npz';path.write_bytes(f'{stream}:{index}'.encode())
                episodes.append(dict(path=str(path),sha256=sha(path),split='train',stream=stream,transitions=180,
                    policy_checkpoint_sha256=policy_sha,world_checkpoint_sha256=world_sha))
        receipt=dict(split='train',validation_unchanged=True,cycle_index=0,episodes=episodes,
            total_real_Source_transitions=720,transitions_per_episode=180,
            policy_checkpoint_sha256=policy_sha,world_checkpoint_sha256=world_sha)
        path=source/'collection_receipt.json';atomic(path,receipt)
        self.assertEqual(cycle.collect(directory),path);self.assertEqual(cycle.collect(directory),path)
        self.popen.assert_not_called();self.assertEqual(cycle.state['source_transitions'],720)
        wrong=deepcopy(receipt);wrong['world_checkpoint_sha256']='0'*64;atomic(path,wrong)
        with self.assertRaises(ValueError):cycle.collect(directory)
        atomic(path,receipt)
        Path(episodes[0]['path']).write_bytes(b'modified Source')
        with self.assertRaises(ValueError):cycle.collect(directory)


class PolicyCycleDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.old=[self.pool('old/'+name,marker,npc)for name,marker,npc in(
            ('nav',10,False),('quiet',20,True),('combat',30,True))]
        nav=self.pool('new/nav',40,False);enemy=self.pool('new/quiet',50,True)
        episodes=[dict(path=str(path),sha256=sha(path),split='train')for directory in(nav,enemy)for path in directory.glob('*.npz')]
        self.receipt=self.root/'source_receipt.json';atomic(self.receipt,dict(split='train',validation_unchanged=True,
            nav_train_directory=str(nav),enemy_train_directory=str(enemy),episodes=episodes))

    def pool(self,name,marker,npc):
        directory=self.root/name/'train';directory.mkdir(parents=True)
        states=np.zeros((181,STATE_DIM),np.float32);states[:,0:2]=.25;states[:,3:6]=1;states[:,6]=marker
        if npc:states[:,8:10]=.65;states[:,11:14]=1
        np.savez_compressed(directory/'episode.npz',states=states,frames=np.full((181,32,32,3),marker,np.uint8),
            actions=np.zeros((180,4),np.float32),events=np.zeros((180,4),np.float32),
            hit_ids=np.full(180,-1,np.int64),attack_flags=np.zeros((180,ENTITY_SLOTS),np.float32),
            map=np.zeros((16,16),np.int8),split=np.asarray('train'),dt=np.asarray(1/30))
        return directory

    def test_actual_npz_merge_replaces_three_nav_three_quiet_and_leaves_combat_six_bitexact(self):
        old=PolicyReplay(self.old,device='cpu',horizon=16,seed=13);batch=old.batch(6);before={k:v.clone()for k,v in batch.items()}
        new=PolicyCycleData(self.receipt,device='cpu',horizon=16,seed=173);result=new.merge(batch,6)
        self.assertIs(result,batch)
        self.assertEqual(result['states'][:,0,6].tolist(),[40]*3+[10]*3+[50]*3+[20]*3+[30]*6)
        for key in result:self.assertTrue(torch.equal(result[key][12:],before[key][12:]),key)
        self.assertTrue(bool(result['sampling_weights'].eq(1).all()))

    def test_policy_cycle_rng_resume_repeats_same_new_source_windows(self):
        first=PolicyCycleData(self.receipt,device='cpu',horizon=16,seed=173)
        old=PolicyReplay(self.old,device='cpu',horizon=16,seed=13)
        first.merge(old.batch(6),6);saved=deepcopy(first.state_dict())
        resumed=PolicyCycleData(self.receipt,device='cpu',horizon=16,seed=173,previous=saved)
        base=old.batch(6)
        a=first.merge({k:v.clone()for k,v in base.items()},6);b=resumed.merge({k:v.clone()for k,v in base.items()},6)
        for key in a:self.assertTrue(torch.equal(a[key],b[key]),key)
        self.assertEqual(first.state_dict(),resumed.state_dict())

    def test_policy_cycle_refuses_unlisted_npz_in_declared_train_directory(self):
        receipt=read(self.receipt);directory=Path(receipt['nav_train_directory'])
        (directory/'unlisted.npz').write_bytes((directory/'episode.npz').read_bytes())
        with self.assertRaises(ValueError):PolicyCycleData(self.receipt,device='cpu',horizon=16,seed=173)

    def test_force_refresh_same_policy_boundary_never_deletes_current_world_snapshot(self):
        source=self.root/'world.pt'
        torch.save(dict(stage='world',training_steps=17,global_training_step=17,
            run_state=dict(global_training_step=17),config=dict(architecture='dreamer_rssm'),
            model=dict(weight=torch.ones(1))),source)
        learner=PolicyTraining.__new__(PolicyTraining);learner.root=self.root;learner.step=200
        learner.args=SimpleNamespace(world_source=source,refresh_every=250,refresh_warmup=20)
        current=self.root/'world_snapshots/refresh_000200_111.pt'
        learner.world_identity=pinned_world(source,current);original=current.read_bytes();learner.load_world=Mock()
        with patch('turkeyworld.train_dreamer_policy.time.time_ns',side_effect=[222,333]):
            learner.refresh(force=True);learner.refresh(force=True)
        self.assertTrue(current.exists());self.assertEqual(current.read_bytes(),original)
        self.assertEqual(learner.world_identity['path'],str(current.resolve()));learner.load_world.assert_not_called()
        self.assertEqual(sorted(path.name for path in current.parent.glob('*.pt')),[current.name])

    def test_policy_existing_stop_prevents_even_one_sgd_update(self):
        learner=PolicyTraining.__new__(PolicyTraining);learner.root=self.root;learner.step=200;learner.actor_updates=100
        learner.args=SimpleNamespace(refresh_world_now=False,updates=0,until_step=0,until_actor_updates=150)
        learner.state={};learner.prepare=Mock();learner.save=Mock();learner.publish=Mock();learner.refresh=Mock()
        learner.train_update=Mock(side_effect=AssertionError('User stop must precede all new SGD'))
        (self.root/'stop.request').write_text('user stopped')
        learner.run()
        learner.train_update.assert_not_called();self.assertEqual(learner.state['status'],'stopped')


if __name__=='__main__':unittest.main()
