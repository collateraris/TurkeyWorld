"""Paired input/clock tests without a display, checkpoint loading or GPU."""
from collections import defaultdict
import math
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from turkeyworld.compare_runtime import PairedRuntime
from turkeyworld.engine import Game, STATE_DIM


def initial_vector(enemies_enabled=False):
    state=np.zeros(STATE_DIM,np.float32)
    state[:2]=(.2,.25);state[3:6]=(1,1,1)
    state[8:10]=(.5,.5)
    if enemies_enabled: state[12:14]=(.6,1)
    return state


class SourceStub:
    mutate_action=False
    def __init__(self,*,seed,endless,enemies_enabled):
        self.seed,self.endless,self.enemies_enabled=seed,endless,enemies_enabled
        self.map=np.zeros((16,16),np.uint8)
        self.state=initial_vector(enemies_enabled)
        self.steps=0;self.actions=[];self.dts=[];self.action_objects=[];self.render_sizes=[]
    def state_vector(self): return self.state.copy()
    def step(self,action,dt):
        control=np.asarray(action).copy();self.actions.append(control);self.action_objects.append(action)
        self.dts.append(dt);self.steps+=1
        self.state[:2]+=control[:2]*.001
        yaw=math.atan2(float(self.state[2]),float(self.state[3]))+float(control[2])*2.2*dt
        self.state[2:4]=(math.sin(yaw),math.cos(yaw));self.state[-1]+=dt/300
        if self.mutate_action: action[:]=-999
    def render(self,*,size,overlays=False):
        if overlays: raise AssertionError('Comparison pixels must exclude overlays')
        self.render_sizes.append(size)
        return np.full((size,size,3),self.steps%255,np.uint8)


class NeuralStub:
    def __init__(self,*,seed=73,navigation_only=True):
        self.args=SimpleNamespace(seed=seed)
        self.model=SimpleNamespace(config=SimpleNamespace(endless=True,navigation_only=navigation_only,
            image_size=32,simulation_hz=30.),weight_identity=object())
        self.world_metadata={'training_steps':3151}
        self.initial_map=np.zeros((16,16),np.uint8)
        self.initial_rgb=np.zeros((32,32,3),np.uint8)
        self.initial_state=torch.tensor(initial_vector(not navigation_only))[None]
        self.initial_memory=torch.arange(7,dtype=torch.float32)[None]
        self.simulation_hz=30.;self.simulation_dt=1/30
        self.steps=0;self.samples=defaultdict(list);self.event_counts=dict.fromkeys(('shot','hit','kill','player_hit'),0)
        self.last_frame=None;self.actions=[];self.dts=[];self.action_objects=[];self.reset_calls=0
        self.fail_next=False;self.mutate_action=False
        self.state=self.initial_state.clone();self.memory=self.initial_memory.clone()
    def reset_from_cached_initial_state(self):
        self.reset_calls+=1;self.state=self.initial_state.clone();self.memory=self.initial_memory.clone()
    def step(self,action):
        if self.fail_next:
            self.fail_next=False
            raise FloatingPointError('Fake nonfinite neural inference')
        control=np.asarray(action).copy();self.actions.append(control);self.action_objects.append(action)
        self.dts.append(self.simulation_dt)
        self.state=self.state.clone();self.state[:,:2]+=torch.tensor(control[:2]*.002)[None]
        yaw=math.atan2(float(self.state[0,2]),float(self.state[0,3]))+float(control[2])*2.2*self.simulation_dt
        self.state[0,2:4]=torch.tensor((math.sin(yaw),math.cos(yaw)))
        self.state[:,-1]+=self.simulation_dt/300
        self.memory=self.memory+.25;self.steps+=1
        self.samples['pipeline_ms'].append(1.)
        self.event_counts['shot']+=int(control[3]>0)
        self.last_frame=np.full((32,32,3),40+self.steps%100,np.uint8)
        if self.mutate_action: action[:]=999
        return self.last_frame
    def report(self,wall_seconds):
        return {'steps':self.steps,'wall_seconds':wall_seconds,'source_engine_calls':0}


class PoseRenderer:
    def __init__(self): self.calls=[];self.poison_argument=False
    def __call__(self,neural,state):
        self.calls.append(state.detach().clone())
        if self.poison_argument: state.add_(.125)
        return np.full((32,32,3),int(float(state[0,0])*100)%255,np.uint8)


class CompareRuntimeTests(unittest.TestCase):
    def paired(self,neural=None,**kwargs):
        neural=neural or NeuralStub();renderer=PoseRenderer()
        result=PairedRuntime(neural,source_factory=SourceStub,render_pose=renderer,source_size=64,**kwargs)
        return result,neural,renderer

    def test_one_shared_tape_fixed_clock_and_two_rgb_results(self):
        pair,neural,renderer=self.paired()
        self.assertEqual(pair.source.seed,73)
        for index,action in enumerate(([1,0,.25,0],[0,-1,-.5,1],[.5,.25,0,0]),1):
            source_frame,neural_frame=pair.step(action)
            np.testing.assert_array_equal(pair.source.actions[-1],np.asarray(action,np.float32))
            np.testing.assert_array_equal(neural.actions[-1],np.asarray(action,np.float32))
            self.assertEqual((pair.steps,pair.source.steps,neural.steps),(index,index,index))
            self.assertEqual(source_frame.shape,(64,64,3))
            self.assertEqual(neural_frame.shape,(32,32,3))
            self.assertEqual((source_frame.dtype,neural_frame.dtype),(np.uint8,np.uint8))
            self.assertIs(neural_frame,neural.last_frame)
        self.assertEqual(pair.source.dts,[1/30]*3)
        self.assertEqual(neural.dts,pair.source.dts)
        status=pair.status()
        self.assertEqual(status['shared_input_steps'],3)
        self.assertEqual((status['source_steps'],status['neural_steps']),(3,3))
        self.assertAlmostEqual(status['simulated_seconds'],.1)
        self.assertTrue(status['navigation_only'])
        self.assertFalse(status['rssm_future_source_feedback'])
        self.assertEqual(status['engine_corrections'],0)
        self.assertEqual(pair.report()['actions'],pair.actions)

    def test_mutating_either_consumer_cannot_change_other_action_or_caller_tape(self):
        pair,neural,_=self.paired()
        neural.mutate_action=True;pair.source.mutate_action=True
        control=np.asarray([.8,-.3,.4,1.],np.float32);original=control.copy()
        pair.step(control)
        np.testing.assert_array_equal(control,original)
        np.testing.assert_array_equal(neural.actions[0],original)
        np.testing.assert_array_equal(pair.source.actions[0],original)
        np.testing.assert_array_equal(pair.actions[0],original)
        self.assertIsNot(neural.action_objects[0],pair.source.action_objects[0])
        self.assertIsNot(neural.action_objects[0],control)
        self.assertIsNot(pair.source.action_objects[0],control)

    def test_failed_neural_inference_does_not_advance_reference_or_pair_clock(self):
        pair,neural,_=self.paired()
        source_before=pair.source.state_vector();own_before=neural.state.clone();memory_before=neural.memory.clone()
        neural.fail_next=True
        with self.assertRaisesRegex(FloatingPointError,'nonfinite'):
            pair.step([1,0,0,0])
        self.assertEqual((pair.steps,pair.source.steps,neural.steps),(0,0,0))
        self.assertEqual(pair.actions,[])
        np.testing.assert_array_equal(pair.source.state_vector(),source_before)
        self.assertTrue(torch.equal(neural.state,own_before) and torch.equal(neural.memory,memory_before))
        pair.step([1,0,0,0])
        self.assertEqual((pair.steps,pair.source.steps,neural.steps),(1,1,1))

    def test_divergent_neural_state_and_memory_are_never_corrected_from_source(self):
        pair,neural,_=self.paired();initial_memory=neural.initial_memory.clone()
        for _ in range(4): pair.step([1,0,0,0])
        expected=neural.initial_state.clone();expected[:,0]+=.008
        self.assertTrue(torch.allclose(neural.state[:,:2],expected[:,:2],atol=1e-7,rtol=0))
        self.assertFalse(np.array_equal(pair.source.state_vector()[:2],neural.state[0,:2].numpy()))
        self.assertTrue(torch.equal(neural.memory,initial_memory+1.))
        self.assertEqual(neural.reset_calls,1)
        self.assertAlmostEqual(pair.xy_drift_tiles,.004*16,places=6)
        # Circular comparison across signedpi must report .02rad, not~2pi.
        pair.source.state[2:4]=(math.sin(math.pi-.01),math.cos(math.pi-.01))
        neural.state[0,2:4]=torch.tensor((math.sin(-math.pi+.01),math.cos(-math.pi+.01)))
        self.assertAlmostEqual(pair.status()['yaw_error_degrees'],math.degrees(.02),places=5)

    def test_reset_is_explicit_cached_trial_boundary_and_clears_all_diagnostics(self):
        pair,neural,_=self.paired(seed=19)
        model_identity=neural.model.weight_identity;source=pair.source
        pair.step([1,0,0,1]);pair.step([0,-1,0,1]);pair.set_mode('shared_camera')
        self.assertEqual(neural.event_counts['shot'],2)
        source_frame,frame=pair.reset()
        self.assertIsNot(pair.source,source)
        self.assertEqual(pair.source.seed,19)
        self.assertEqual((pair.steps,pair.source.steps,neural.steps),(0,0,0))
        self.assertFalse(pair.finished)
        self.assertEqual(pair.actions,[])
        self.assertEqual(dict(neural.samples),{})
        self.assertTrue(all(value==0 for value in neural.event_counts.values()))
        self.assertIsNone(neural.last_frame)
        self.assertTrue(torch.equal(neural.state,neural.initial_state))
        self.assertTrue(torch.equal(neural.memory,neural.initial_memory))
        self.assertNotEqual(neural.memory.data_ptr(),neural.initial_memory.data_ptr())
        self.assertIs(neural.model.weight_identity,model_identity)
        self.assertEqual((pair.mode,pair.status()['trial_resets']),('shared_camera',1))
        self.assertEqual((source_frame.shape,frame.shape),((64,64,3),(32,32,3)))

    def test_exact450_horizon_remains_visible_and_requires_explicit_reset(self):
        pair=PairedRuntime(NeuralStub(),source_size=32,source_factory=SourceStub,render_pose=PoseRenderer())
        for _ in range(450): pair.step([0,0,0,0])
        self.assertTrue(pair.finished)
        self.assertEqual(pair.status()['steps'],450)
        self.assertEqual(pair.simulated_seconds,15.)
        held_frames=(pair.source_frame.copy(),pair.neural_frame.copy())
        with self.assertRaisesRegex(RuntimeError,'complete'):
            pair.step([1,0,0,0])
        self.assertEqual((pair.steps,pair.source.steps,pair.neural.steps),(450,450,450))
        np.testing.assert_array_equal(pair.source_frame,held_frames[0])
        np.testing.assert_array_equal(pair.neural_frame,held_frames[1])
        pair.reset();pair.step([0,0,0,0])
        self.assertEqual((pair.steps,pair.source.steps,pair.neural.steps),(1,1,1))

    def test_initial_state_map_and_cached_observation_mismatches_rejected(self):
        for field in ('initial_state','initial_map','initial_rgb'):
            neural=NeuralStub()
            if field=='initial_state': neural.initial_state[0,0]+=.01
            elif field=='initial_map': neural.initial_map[3,4]=1
            else: neural.initial_rgb[2,3,0]=1
            with self.subTest(field=field),self.assertRaises(ValueError):
                PairedRuntime(neural,source_factory=SourceStub,render_pose=PoseRenderer())
            self.assertEqual(neural.reset_calls,0)

    def test_shared_camera_uses_exact_source_pose_only_for_display_without_state_alias(self):
        pair,neural,renderer=self.paired()
        pair.step([1,0,0,0]);own=neural.state.clone();hidden=neural.memory.clone()
        autonomous=pair.neural_frame
        pair.set_mode('shared_camera')
        np.testing.assert_array_equal(renderer.calls[-1][0].numpy(),pair.source.state_vector())
        self.assertTrue(torch.equal(neural.state,own) and torch.equal(neural.memory,hidden))
        self.assertEqual((pair.steps,neural.steps,pair.source.steps),(1,1,1))
        status=pair.status()
        self.assertTrue(status['reference_pose_for_display_only'])
        self.assertFalse(status['rssm_future_source_feedback'])
        pair.step([0,-1,0,0])
        np.testing.assert_array_equal(renderer.calls[-1][0].numpy(),pair.source.state_vector())
        own=neural.state.clone();hidden=neural.memory.clone();source=pair.source.state_vector()
        # Even a callback mutating its pose argument cannot corrupt either state.
        renderer.poison_argument=True;pair.set_mode('shared_camera')
        self.assertTrue(torch.equal(neural.state,own) and torch.equal(neural.memory,hidden))
        np.testing.assert_array_equal(pair.source.state_vector(),source)
        pair.set_mode('autonomous')
        self.assertIs(pair.neural_frame,neural.last_frame)
        self.assertFalse(pair.status()['reference_pose_for_display_only'])
        with self.assertRaises(ValueError): pair.set_mode('silently_teacher_corrected')
        self.assertEqual(pair.mode,'autonomous')

    def test_actual_source_game_matches_neural_cached_profile_map_and_initial_rgb(self):
        neural=NeuralStub(seed=137,navigation_only=True)
        reference=Game(seed=137,endless=True,enemies_enabled=False)
        neural.initial_state=torch.tensor(reference.state_vector())[None]
        neural.initial_map=reference.map.copy()
        neural.initial_rgb=reference.render(size=32,overlays=False)
        pair=PairedRuntime(neural,source_factory=Game,render_pose=PoseRenderer(),source_size=32)
        self.assertFalse(pair.source.enemies_enabled)
        self.assertTrue(pair.source.endless)
        np.testing.assert_array_equal(pair.source.map,neural.initial_map)
        np.testing.assert_array_equal(pair.source.state_vector(),neural.initial_state[0].numpy())
        before=pair.source.state_vector()
        pair.step([1,0,0,0])
        reference.step([1,0,0,0],dt=1/30)
        np.testing.assert_array_equal(pair.source.state_vector(),reference.state_vector())
        self.assertFalse(np.array_equal(pair.source.state_vector()[:2],neural.state[0,:2].numpy()))
        self.assertFalse(pair.source.state_vector()[12:14].any())
        self.assertEqual(pair.status()['engine_corrections'],0)

    def test_bad_controls_or_dimensions_fail_before_either_game_advances(self):
        pair,neural,_=self.paired()
        for value in ([1,0,0],[1,0,0,0,0],[float('nan'),0,0,0],[0,0,float('inf'),0]):
            with self.subTest(value=value),self.assertRaises(ValueError): pair.step(value)
        self.assertEqual((pair.steps,neural.steps,pair.source.steps),(0,0,0))
        for settings in ({'horizon':0},{'horizon':True},{'source_size':16},{'source_size':True}):
            with self.subTest(settings=settings),self.assertRaises(ValueError):
                PairedRuntime(NeuralStub(),source_factory=SourceStub,render_pose=PoseRenderer(),**settings)


if __name__=='__main__': unittest.main()
