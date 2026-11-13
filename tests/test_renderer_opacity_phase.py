from copy import deepcopy
import random
import unittest

import numpy as np
import torch

from test_renderer_geometry_opacity import working_model
from turkeyworld.renderer_geometry_opacity import enable_geometry_opacity
from turkeyworld.renderer_refinement import configure_refinement
from turkeyworld.renderer_native_phase import _same
from turkeyworld.renderer_opacity_phase import restore_renderer_opacity_parent,opacity_geometry_protocol


class Sampler:
    def __init__(self,draws=0):self.value={'phase_draws':draws,'parent':{'mix_counts':{'navigation':draws}}}
    def sampler_state(self):return deepcopy(self.value)
    def restore_sampler(self,state):self.value=deepcopy(state)


class OpacityPhaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def fixture(self):
        old=working_model();configure_refinement(old)
        opt=torch.optim.AdamW([{'params':old.base_student.parameters(),'lr':1e-4},
            {'params':[p for p in old.flow_renderer.parameters()if p.requires_grad],'lr':1e-4}],weight_decay=.01)
        for _ in range(3):
            opt.zero_grad();sum(p.square().sum()*1e-8 for group in opt.param_groups for p in group['params']).backward();opt.step()
        config=dict(run_dir='old',steps=3,geometry_wall_supervision=True,batch=16,lr=.0001)
        identity={'source':'unchanged'}
        samplers=[Sampler(n)for n in (48,12,24)]
        saved=deepcopy(dict(base=old.base_student.state_dict(),flow=old.flow_renderer.state_dict(),
            optimizer=opt.state_dict(),step=3,configuration=config,identity=identity,frozen_fingerprint='frozen',
            sampler=samplers[0].sampler_state(),far_wall_aux_sampler=samplers[1].sampler_state(),
            geometry_wall_aux_sampler=samplers[2].sampler_state(),rng_torch=torch.get_rng_state(),
            rng_numpy=np.random.get_state(),rng_python=random.getstate(),rng_cuda=[]))
        model=deepcopy(old);enable_geometry_opacity(model);configure_refinement(model)
        optimizer=torch.optim.AdamW([
            {'params':model.base_student.parameters(),'lr':.0001},
            {'params':[p for n,p in model.flow_renderer.named_parameters()if p.requires_grad and not n.startswith('geometry_opacity.')],'lr':.0001},
            {'params':model.flow_renderer.geometry_opacity.parameters(),'lr':.00005}],weight_decay=.01)
        return model,optimizer,saved,{**identity,'private_geometry_protocol':opacity_geometry_protocol()}, \
            {**config,'run_dir':'new','steps':6,'private_geometry_supervision':True,'private_geometry_parent':'old/checkpoint.pt'}

    def adopt(self,model,opt,saved,identity,config):
        samplers=[Sampler()for _ in range(3)]
        proof=restore_renderer_opacity_parent(saved,model,opt,*samplers,identity,config,'frozen',
            parent_checkpoint_sha256='a'*64,protocol=opacity_geometry_protocol())
        return proof,samplers

    def test_mixed_age_Adam_old_weights_three_samplers_and_rng_preserved(self):
        model,opt,saved,identity,config=self.fixture()
        private=deepcopy(model.flow_renderer.geometry_opacity.state_dict())
        proof,samplers=self.adopt(model,opt,saved,identity,config)
        self.assertTrue(_same(saved['base'],model.base_student.state_dict()))
        self.assertTrue(_same(saved['flow'],{k:v for k,v in model.flow_renderer.state_dict().items()if not k.startswith('geometry_opacity.')}))
        self.assertTrue(_same(private,model.flow_renderer.geometry_opacity.state_dict()))
        self.assertTrue(_same(saved['optimizer']['state'],opt.state_dict()['state']))
        self.assertTrue(_same(saved['optimizer']['param_groups'],opt.state_dict()['param_groups'][:2]))
        self.assertTrue(torch.equal(saved['rng_torch'],torch.get_rng_state()))
        self.assertEqual(proof['fresh_parameter_tensors'],4)
        for sampler,key in zip(samplers,('sampler','far_wall_aux_sampler','geometry_wall_aux_sampler')):
            self.assertTrue(_same(sampler.sampler_state(),saved[key]))
        opt.zero_grad();sum(p.square().sum()*1e-8 for group in opt.param_groups for p in group['params']).backward();opt.step()
        states=opt.state_dict()['state'];groups=opt.state_dict()['param_groups']
        for group in groups[:2]:self.assertTrue(all(float(states[i]['step'])==4 for i in group['params']))
        self.assertTrue(all(float(states[i]['step'])==1 for i in groups[2]['params']))

    def test_changed_source_options_private_initialization_or_old_identity_rejected(self):
        for change in ('identity','lr','private','setting'):
            model,opt,saved,identity,config=self.fixture()
            if change=='identity':identity['source']='tampered'
            elif change=='lr':opt.param_groups[0]['lr']=.002
            elif change=='setting':config['batch']=32
            else:
                with torch.no_grad():model.flow_renderer.geometry_opacity[-1].bias.add_(.1)
            with self.assertRaises(ValueError):self.adopt(model,opt,saved,identity,config)
            self.assertFalse(opt.state)


if __name__=='__main__':unittest.main()
