"""Full named Adam migration and frozen-asset training over real RF experts."""
from copy import deepcopy
import json
import random
import unittest

import numpy as np
import torch
from torch import nn

from turkeyworld.renderer_moe import RendererTokenMoE,FIVE_SCHEMA
from turkeyworld.renderer_moe_five_phase import renderer_five_parameter_groups
from turkeyworld.renderer_native_phase import _same
from turkeyworld.renderer_rf5_experts import upgrade_rf5_experts
from turkeyworld.renderer_rf5_phase import (renderer_rf5_protocol,configure_rf5_training,
    build_rf5_optimizer,rf5_frozen_fingerprint,rf5_parameter_counts)


class Head(nn.Module):
    def __init__(self,base=False):
        super().__init__();self.body=nn.Linear(3,3)
        self.output_moe=RendererTokenMoE(3 if base else 256,3 if base else 64,
            8 if base else 32,2 if base else 4,role_schema=FIVE_SCHEMA,experts=5)
        if not base:
            self.codec=nn.Linear(3,3)
            self.geometry_opacity=nn.Sequential(nn.Linear(2,4),nn.Tanh(),nn.Linear(4,1))
            self.kind,self.mode,self.steps='student','rectified_flow',1
            self.register_buffer('fixed_noise',torch.ones(1,4,8,8))


class Model(nn.Module):
    def __init__(self):
        super().__init__();self.base_student=Head(True);self.flow_renderer=Head()
        self.world=nn.Linear(3,3);self.register_buffer('map_buffer',torch.arange(16))


def fixture():
    model=Model();model.flow_renderer.geometry_opacity.requires_grad_(False)
    groups,names=renderer_five_parameter_groups(model)
    optimizer=torch.optim.AdamW([dict(params=params,lr=lr)for params,lr in
        zip(groups,(1e-4,1e-4,5e-5,1e-4,1e-4))],weight_decay=.01)
    # Create real old moments, including the now-frozen historical opacity group.
    for p in model.flow_renderer.geometry_opacity.parameters():p.requires_grad_(True)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        sum(p.square().mean()+.01*p.mean()for group in groups for p in group).backward()
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model.flow_renderer.geometry_opacity.requires_grad_(False)
    moments=optimizer.state_dict()
    for index,moment in moments['state'].items():moment['step'].fill_(2+index%7)
    saved=dict(base=deepcopy(model.base_student.state_dict()),flow=deepcopy(model.flow_renderer.state_dict()),
        optimizer=deepcopy(moments),optimizer_parameter_names=deepcopy(names),step=1409,
        renderer_training_steps=29609,base_training_steps=29209,
        identity={'renderer_moe_protocol':{'name':'one_enemy_renderer_token_moe_five_v1'}},
        sampler={'draws':1},far_wall_aux_sampler={'draws':2},geometry_wall_aux_sampler={'draws':3},
        rng_torch=torch.get_rng_state().clone(),rng_numpy=deepcopy(np.random.get_state()),rng_python=random.getstate())
    upgrade_rf5_experts(model.flow_renderer.output_moe)
    model.flow_renderer.steps=5;configure_rf5_training(model)
    return model,saved


class RendererRF5PhaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_all_five_old_groups_and_mixed_ages_are_exact_and_new80_are_fresh(self):
        model,saved=fixture();original=deepcopy(saved);weights=deepcopy(model.state_dict())
        rng=torch.get_rng_state().clone();numpy_rng=deepcopy(np.random.get_state());python_rng=random.getstate()
        optimizer,proof=build_rf5_optimizer(model,saved);state=optimizer.state_dict()
        self.assertTrue(_same(state['state'],saved['optimizer']['state']))
        self.assertEqual(state['param_groups'][:5],saved['optimizer']['param_groups'])
        self.assertEqual(len(state['param_groups']),6)
        self.assertEqual(state['param_groups'][5]['lr'],1e-4)
        self.assertFalse(set(state['param_groups'][5]['params'])&set(state['state']))
        self.assertEqual(len(state['param_groups'][5]['params']),80)
        self.assertEqual(proof['added_parameters'],2472640)
        self.assertEqual(proof['new_Adam_states_at_start'],0)
        self.assertEqual(len(proof['renamed_legacy_parameter_names']),60)
        for source,destination in proof['renamed_legacy_parameter_names'].items():
            self.assertIn('.experts.',source);self.assertIn('.legacy.',destination)
        self.assertEqual(len(proof['optimizer_parameter_names']),6)
        self.assertTrue(_same(saved,original));self.assertTrue(_same(weights,model.state_dict()))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(_same(numpy_rng,np.random.get_state()));self.assertEqual(python_rng,random.getstate())
        self.assertEqual(proof['frozen_base_training_steps'],29209)
        self.assertEqual(proof['parent_renderer_training_steps'],29609)
        json.dumps(proof,allow_nan=False)

    def test_actual_three_steps_train_RF_and_all_five_expansions_keep_frozen_weights_and_Adam_exact(self):
        model,saved=fixture();optimizer,proof=build_rf5_optimizer(model,saved)
        fingerprint=rf5_frozen_fingerprint(model);before=deepcopy(model.state_dict())
        old_moments=deepcopy(optimizer.state_dict());names=proof['optimizer_parameter_names']
        parameters=dict(model.named_parameters());old_ids=[i for group in old_moments['param_groups'][:5]for i in group['params']]
        frozen_ids=[index for index,name in zip(old_ids,[n for group in names for n in group])if not parameters[name].requires_grad]
        tokens=torch.linspace(-1,1,2*16*256).reshape(2,16,256)
        state=torch.zeros(2,20);state[:,0:5]=torch.tensor([.25,.25,0.,1.,1.])
        state[:,8:10]=torch.tensor([.4,.25]);state[:,12:14]=1
        second_gradients=[]
        for update in range(3):
            optimizer.zero_grad(set_to_none=True)
            result=model.flow_renderer.output_moe(tokens,state)
            loss=(result-.2).square().mean()+model.flow_renderer.body(result[...,:3]).square().mean()
            loss.backward()
            if update==1:
                second_gradients=[sum(float(p.grad.abs().sum())for n,p in expert.named_parameters()
                    if n.startswith('private_')and p.grad is not None)for expert in model.flow_renderer.output_moe.experts]
                self.assertTrue(all(value>0 and np.isfinite(value)for value in second_gradients))
            optimizer.step()
            self.assertEqual(rf5_frozen_fingerprint(model),fingerprint)
        after=optimizer.state_dict()
        for index in frozen_ids:self.assertTrue(_same(after['state'][index],old_moments['state'][index]))
        self.assertTrue(all(p.grad is None for module in(model.base_student,model.world,
            model.flow_renderer.codec,model.flow_renderer.geometry_opacity)for p in module.parameters()))
        self.assertTrue(any(not torch.equal(before[n],v)for n,v in model.state_dict().items()if '.private_output.'in n))
        for index in after['param_groups'][5]['params']:
            self.assertEqual(float(after['state'][index]['step']),3.)
            self.assertTrue(all(bool(torch.isfinite(v).all())for v in after['state'][index].values()if torch.is_tensor(v)))
        for index,name in zip(old_ids,[n for group in names for n in group]):
            if parameters[name].requires_grad:
                self.assertEqual(float(after['state'][index]['step']),float(old_moments['state'][index]['step'])+3)

    def test_missing_moment_wrong_names_old_weights_age_shape_and_nonzero_new_head_reject(self):
        template,parent=fixture()
        for mode in('names','duplicate_id','missing_moment','age','shape','nan','old_weight','new_output','trainable_Base'):
            model,saved=deepcopy(template),deepcopy(parent)
            if mode=='names':saved['optimizer_parameter_names'][3][0:2]=list(reversed(saved['optimizer_parameter_names'][3][0:2]))
            elif mode=='duplicate_id':saved['optimizer']['param_groups'][1]['params'][0]=saved['optimizer']['param_groups'][0]['params'][0]
            elif mode=='missing_moment':saved['optimizer']['state'].pop(0)
            elif mode=='age':saved['optimizer']['state'][0]['step'].zero_()
            elif mode=='shape':saved['optimizer']['state'][0]['exp_avg']=torch.zeros(1)
            elif mode=='nan':saved['optimizer']['state'][0]['exp_avg'].fill_(float('nan'))
            elif mode=='old_weight':
                with torch.no_grad():model.flow_renderer.output_moe.experts[0].legacy.linear1.weight.add_(.01)
            elif mode=='new_output':
                with torch.no_grad():model.flow_renderer.output_moe.experts[0].private_output.bias.add_(.01)
            else:model.base_student.requires_grad_(True)
            before=deepcopy(model.state_dict());old=deepcopy(saved)
            with self.subTest(mode=mode),self.assertRaises(ValueError):build_rf5_optimizer(model,saved)
            self.assertTrue(_same(before,model.state_dict()))
            # NaN is deliberately invalid, so compare its exact byte content separately.
            if mode!='nan':self.assertTrue(_same(old,saved))

    def test_configuration_and_whole_frozen_fingerprint_include_Base_world_codec_opacity_and_buffers(self):
        model,saved=fixture();model.train();report=configure_rf5_training(model)
        self.assertFalse(model.base_student.training);self.assertTrue(model.flow_renderer.training)
        self.assertFalse(model.flow_renderer.codec.training);self.assertFalse(model.flow_renderer.geometry_opacity.training)
        self.assertTrue(all(not p.requires_grad for module in(model.base_student,model.world,
            model.flow_renderer.codec,model.flow_renderer.geometry_opacity)for p in module.parameters()))
        self.assertTrue(all(p.requires_grad for n,p in model.flow_renderer.named_parameters()
            if not n.startswith(('codec.','geometry_opacity.'))))
        original=rf5_frozen_fingerprint(model)
        with torch.no_grad():model.flow_renderer.body.bias.add_(.1)
        self.assertEqual(original,rf5_frozen_fingerprint(model))
        with torch.no_grad():model.flow_renderer.fixed_noise.add_(.1)
        self.assertNotEqual(original,rf5_frozen_fingerprint(model))
        self.assertEqual(report['parameters']['parameters_per_expert'],503072)
        self.assertEqual(rf5_parameter_counts(model)['added_parameter_tensors'],80)
        protocol=renderer_rf5_protocol(5);self.assertEqual(protocol['solver_steps'],5)
        self.assertEqual(renderer_rf5_protocol()['solver_steps'],3)
        self.assertEqual(protocol['expert_architecture']['private_initialization_seeds'],[173,174,175,176,177])
        self.assertFalse(protocol['automatic_active_promotion'])


if __name__=='__main__':unittest.main()
