from copy import deepcopy
import random
import unittest
import numpy as np
import torch

from test_base_geometry_phase import parent_fixture
from turkeyworld.base_geometry_phase import adopt_base_geometry_optimizer
from turkeyworld.renderer_native_phase import _same
from turkeyworld.renderer_rf_chain_phase import (adopt_rf_chain_optimizer,
    configure_rf_chain_training,rf_chain_frozen_fingerprint)


def selected_parent(positive=True):
    model,old=parent_fixture();optimizer,proof=adopt_base_geometry_optimizer(model,old)
    if positive:
        optimizer.zero_grad(set_to_none=True)
        sum(p.square().mean()+.01*p.mean()for p in model.base_student.parameters()).backward()
        optimizer.step();optimizer.zero_grad(set_to_none=True)
    step=int(positive)
    saved=dict(base=deepcopy(model.base_student.state_dict()),flow=deepcopy(model.flow_renderer.state_dict()),
        optimizer=deepcopy(optimizer.state_dict()),optimizer_parameter_names=proof['optimizer_parameter_names'],
        step=step,base_training_steps=29209+step,renderer_training_steps=30717,
        phase=dict(protocol=dict(name='base_expanded_geometry_v1')),
        sampler=dict(PCG='main'),far_wall_aux_sampler=dict(PCG='far'),geometry_wall_aux_sampler=dict(PCG='geometry'))
    configure_rf_chain_training(model)
    return model,saved


class RFChainPhaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_seven_groups_and_all_mixed_moments_exact_while_Base_and_codec_stay_frozen(self):
        model,saved=selected_parent();parent=deepcopy(saved);clock=torch.get_rng_state().clone()
        numpy,python=deepcopy(np.random.get_state()),random.getstate()
        optimizer,proof=adopt_rf_chain_optimizer(model,saved)
        self.assertTrue(_same(saved['optimizer'],optimizer.state_dict()))
        self.assertEqual(len(optimizer.param_groups),7)
        self.assertEqual(len(optimizer.state),sum(map(len,proof['optimizer_parameter_names'])))
        self.assertEqual(proof['new_optimizer_groups'],0);self.assertTrue(_same(saved,parent))
        self.assertTrue(torch.equal(clock,torch.get_rng_state()));self.assertTrue(_same(numpy,np.random.get_state()))
        self.assertEqual(python,random.getstate())
        frozen=rf_chain_frozen_fingerprint(model);before=deepcopy(optimizer.state_dict())
        tokens=torch.linspace(-1,1,2*16*256).reshape(2,16,256);state=torch.zeros(2,20)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            (model.flow_renderer.output_moe(tokens,state)-.3).square().mean().backward();optimizer.step()
            self.assertEqual(frozen,rf_chain_frozen_fingerprint(model))
        after=optimizer.state_dict();params=dict(model.named_parameters())
        flat=[n for g in proof['optimizer_parameter_names']for n in g]
        for i,n in enumerate(flat):
            if not params[n].requires_grad:self.assertTrue(_same(before['state'][i],after['state'][i]),n)
        self.assertTrue(_same(saved,parent),'No aliased parent moments')

    def test_best0_keeps_all80_missing_frozen_states_absent_instead_of_inventing_them(self):
        model,saved=selected_parent(False);optimizer,proof=adopt_rf_chain_optimizer(model,saved)
        self.assertEqual(len(optimizer.state),sum(map(len,proof['optimizer_parameter_names']))-80)
        self.assertEqual(len(proof['missing_frozen_Base_state_ids']),80)
        self.assertTrue(_same(optimizer.state_dict(),saved['optimizer']))
        for positive in (False,True):
            net,parent=selected_parent(positive);broken=deepcopy(parent)
            broken['optimizer']['state'].pop(0)
            with self.assertRaises(ValueError):adopt_rf_chain_optimizer(net,broken)

    def test_wrong_whole_checkpoint_names_clocks_or_moments_are_rejected(self):
        model,parent=selected_parent()
        for mutate in (lambda p:p['optimizer_parameter_names'][3].reverse(),
                       lambda p:p['optimizer']['state'][0]['step'].zero_(),
                       lambda p:p['optimizer']['state'][0]['exp_avg'].fill_(float('nan')),
                       lambda p:p.update(renderer_training_steps=30718),
                       lambda p:p['base']['rgb.bias'].add_(.01)):
            saved=deepcopy(parent);mutate(saved)
            with self.assertRaises(ValueError):adopt_rf_chain_optimizer(model,saved)


if __name__=='__main__':unittest.main()
