"""Runtime Euler work is separate from the strictly bound trained sampler."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from turkeyworld.models import STATE_DIM
from turkeyworld.neural_play import configure_runtime_rf_steps
from turkeyworld.renderer_student import (build_renderer_student,export_renderer_student,
    load_renderer_student)
from test_renderer_student import TinyTeacher


class RuntimeRFStepsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def loaded_one_step(self,folder):
        teacher=TinyTeacher().eval()
        student=build_renderer_student(teacher,steps=1,width=32,depth=1,heads=4,mlp_hidden=64)
        path=Path(folder)/'renderer.pt'
        export_renderer_student(teacher,student,path,training_steps=7)
        metadata=load_renderer_student(teacher,path)
        teacher.eval()
        self.assertEqual(teacher.flow_renderer.steps,1)
        self.assertEqual(teacher.config.flow_steps,1)
        self.assertEqual(metadata['solver_steps'],1)
        return teacher,metadata,path

    def test_actual_five_Euler_calls_times_and_left_integral_preserve_every_binding(self):
        with TemporaryDirectory()as folder:
            model,metadata,path=self.loaded_one_step(folder)
            config=deepcopy(asdict(model.config));saved_metadata=deepcopy(metadata)
            weights={key:value.clone()for key,value in model.state_dict().items()}
            checksum=hashlib.sha256(path.read_bytes()).hexdigest()
            torch_rng=torch.get_rng_state().clone();python_rng=random.getstate();np_rng=np.random.get_state()
            override=configure_runtime_rf_steps(model,5)
            self.assertEqual(override,dict(kind='runtime_rf_euler_steps_v1',checkpoint_solver_steps=1,
                solver='euler',solver_steps=5,runtime_only=True))
            base=torch.full((1,3,32,32),.5);state=torch.zeros(1,STATE_DIM)
            state[:,0:5]=torch.tensor([.25,.25,0.,1.,1.])
            flow=model.flow_renderer
            noise=torch.zeros(1,flow.latent_channels,flow.latent_size,flow.latent_size)
            original_noise=noise.clone();calls=[]
            def quadratic_time(current,time,condition,labels,self_condition=None,*extra):
                calls.append(time.clone())
                return time.square()[:,None,None,None].expand_as(current)
            with torch.inference_mode(),patch.object(flow,'velocity',side_effect=quadratic_time)as velocity:
                endpoint=flow.latent_sample(base,state,noise=noise,map_condition=torch.ones(1,16,16,dtype=torch.long))
            self.assertEqual(velocity.call_count,5)  # Student Euler, not teacher Heun.
            torch.testing.assert_close(torch.cat(calls),torch.tensor([0.,.2,.4,.6,.8]),rtol=0,atol=1e-7)
            # Left Euler for v(t)=t^2 is .2*(0+.04+.16+.36+.64)=.24.
            # A single-step call gives0; five repeated full steps also fail this.
            torch.testing.assert_close(endpoint,torch.full_like(noise,.24),rtol=0,atol=1e-7)
            self.assertTrue(torch.equal(noise,original_noise))
            self.assertTrue(torch.equal(torch.get_rng_state(),torch_rng))
            self.assertEqual(random.getstate(),python_rng)
            after_np=np.random.get_state();self.assertEqual(np_rng[0],after_np[0])
            np.testing.assert_array_equal(np_rng[1],after_np[1]);self.assertEqual(np_rng[2:],after_np[2:])
            self.assertEqual(asdict(model.config),config)
            self.assertEqual(metadata,saved_metadata)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),checksum)
            self.assertTrue(all(torch.equal(value,model.state_dict()[key])for key,value in weights.items()))
            self.assertNotIn('steps',flow.state_dict())

    def test_none_is_an_exact_noop_even_without_a_flow_renderer(self):
        model=SimpleNamespace(config={'flow_steps':1})
        before=deepcopy(vars(model));rng=torch.get_rng_state().clone()
        self.assertIsNone(configure_runtime_rf_steps(model,None))
        self.assertEqual(vars(model),before)
        self.assertTrue(torch.equal(torch.get_rng_state(),rng))

    def test_invalid_steps_or_non_Euler_student_reject_without_mutation(self):
        flow=SimpleNamespace(kind='student',mode='rectified_flow',steps=1)
        model=SimpleNamespace(flow_renderer=flow)
        for value in(True,False,0,-1,1.5,'5',float('inf')):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):configure_runtime_rf_steps(model,value)
                self.assertEqual(flow.steps,1)
        for kind,mode in(('teacher','flow_matching'),('teacher','rectified_flow'),('student','flow_matching')):
            invalid=SimpleNamespace(flow_renderer=SimpleNamespace(kind=kind,mode=mode,steps=4))
            with self.subTest(kind=kind,mode=mode):
                with self.assertRaises(ValueError):configure_runtime_rf_steps(invalid,5)
                self.assertEqual(invalid.flow_renderer.steps,4)
        with self.assertRaises(ValueError):configure_runtime_rf_steps(SimpleNamespace(),5)

    def test_runtime_override_does_not_relax_strict_bundle_solver_validation(self):
        with TemporaryDirectory()as folder:
            model,metadata,path=self.loaded_one_step(folder)
            configure_runtime_rf_steps(model,5)
            original_flow=model.flow_renderer;original_config=model.config
            original={key:value.clone()for key,value in model.state_dict().items()}
            payload=torch.load(path,map_location='cpu',weights_only=False)
            payload['solver_steps']=5;payload['config']['flow_steps']=5
            bad=Path(folder)/'invalid_five_step_bundle.pt';torch.save(payload,bad)
            with self.assertRaisesRegex(ValueError,'one or four Euler steps'):
                load_renderer_student(model,bad)
            self.assertIs(model.flow_renderer,original_flow)
            self.assertIs(model.config,original_config)
            self.assertEqual(model.flow_renderer.steps,5)
            self.assertEqual(model.config.flow_steps,1)
            self.assertEqual(metadata['solver_steps'],1)
            self.assertTrue(all(torch.equal(value,model.state_dict()[key])for key,value in original.items()))


if __name__=='__main__':unittest.main()
