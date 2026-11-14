from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from turkeyworld.decoder_rf_chain_phase import adopt_decoder_rf_chain_optimizer,validate_decoder_rf_chain_optimizer
from turkeyworld.renderer_native_phase import _same
from turkeyworld.train_decoder_rf_adaptation import decoder_state


def fixture(*,missing_base=False):
    model=nn.Module();model.base_student=nn.Module();model.base_student.config=SimpleNamespace(output_moe_expanded_enabled=True)
    model.base_student.old=nn.ParameterList([nn.Parameter(torch.tensor([.1]))for _ in range(146)])
    model.base_student.output_moe=nn.Module();model.base_student.output_moe.experts=nn.ModuleList()
    for _ in range(5):
        expert=nn.Module();expert.fresh=nn.ParameterList([nn.Parameter(torch.tensor([.2]))for _ in range(16)])
        model.base_student.output_moe.experts.append(expert)
    model.flow_renderer=nn.Module();model.flow_renderer.steps=3
    model.flow_renderer.old=nn.ParameterList([nn.Parameter(torch.tensor([.3]))for _ in range(391)])
    model.flow_renderer.codec=nn.Module();model.flow_renderer.codec.decoder=nn.ParameterList([nn.Parameter(torch.tensor([.4]))for _ in range(60)])
    model.flow_renderer.codec.post_quant_conv=nn.ParameterList([nn.Parameter(torch.tensor([.5]))for _ in range(2)])
    parameters=dict(model.named_parameters());base=[n for n in parameters if n.startswith('base_student.old.')]
    flow=[n for n in parameters if n.startswith('flow_renderer.old.')]
    groups=[base,flow[:100],flow[100:200],flow[200:300],flow[300:350],flow[350:]]
    groups.append([n for n in parameters if '.fresh.'in n])
    ids=[];offset=0;states={};opts=[]
    for index,names in enumerate(groups):
        current=list(range(offset,offset+len(names)));ids.extend(current)
        opts.append(dict(params=current,lr=.0001+index*.00001,betas=(.8,.95),eps=1e-7,weight_decay=.01,amsgrad=False,
            maximize=False,foreach=None,capturable=False,differentiable=False,fused=None,decoupled_weight_decay=True))
        for key in current:
            if missing_base and index==6:continue
            states[key]=dict(step=torch.tensor(float(20+index)),exp_avg=torch.tensor([key*.0001]),exp_avg_sq=torch.tensor([.2]))
        offset+=len(names)
    rf=dict(base=deepcopy(model.base_student.state_dict()),flow=deepcopy(model.flow_renderer.state_dict()),
        optimizer_parameter_names=groups,optimizer=dict(state=states,param_groups=opts))
    dnames=[n for n in parameters if n.startswith(('flow_renderer.codec.decoder.','flow_renderer.codec.post_quant_conv.'))]
    dopt=torch.optim.AdamW([parameters[n]for n in dnames],lr=1e-4,betas=(.75,.97),eps=1e-6)
    dstate=dopt.state_dict();dstate['state']={i:dict(step=torch.tensor(1725.),exp_avg=torch.tensor([.01]),exp_avg_sq=torch.tensor([.1]))for i in range(62)}
    parent=dict(step=1725,decoder=deepcopy(decoder_state(model)),optimizer_parameter_names=dnames,optimizer=dstate)
    model._test_canonical=deepcopy(groups)
    return model,rf,parent


def configure(model):
    model.eval().requires_grad_(False)
    model.flow_renderer.codec.requires_grad_(True)


class DecoderRFChainPhaseTests(unittest.TestCase):
    def setUp(self):
        self.canonical=patch('turkeyworld.decoder_rf_chain_phase.rf_chain_canonical_names',side_effect=lambda m:m._test_canonical)
        self.canonical.start();self.addCleanup(self.canonical.stop)
    def test_all679_mixed_ages_options_and_names_survive_and_only_D_updates(self):
        model,rf,parent=fixture();before=deepcopy(model.state_dict());rng=torch.get_rng_state().clone()
        with patch('turkeyworld.decoder_rf_chain_phase.configure_decoder_rf_chain',side_effect=configure):
            optimizer,report=adopt_decoder_rf_chain_optimizer(model,rf,parent)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertEqual(report['total_Adam_states'],679)
        adopted=optimizer.state_dict();self.assertTrue(_same(adopted['param_groups'][:7],rf['optimizer']['param_groups']))
        for key,value in rf['optimizer']['state'].items():self.assertTrue(_same(adopted['state'][key],value))
        for key,value in parent['optimizer']['state'].items():self.assertTrue(_same(adopted['state'][617+key],value))
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            sum(p.square().sum()for p in model.parameters()if p.requires_grad).backward();optimizer.step()
        for name,value in model.state_dict().items():
            if not name.startswith(('flow_renderer.codec.decoder.','flow_renderer.codec.post_quant_conv.')):
                self.assertTrue(torch.equal(value,before[name]))
        for key,value in rf['optimizer']['state'].items():self.assertTrue(_same(optimizer.state_dict()['state'][key],value))
        self.assertTrue(all(float(optimizer.state_dict()['state'][617+i]['step'])==1727 for i in range(62)))
        state=optimizer.state_dict();validate_decoder_rf_chain_optimizer(model,state,rf,parent,local_updates=2)
        for mutate in(lambda s:s['state'][0]['exp_avg'].add_(.1),
                       lambda s:s['state'][617].update(step=torch.tensor(1726.)),
                       lambda s:s['param_groups'][7].update(lr=.01)):
            broken=deepcopy(state);mutate(broken)
            with self.assertRaises(ValueError):validate_decoder_rf_chain_optimizer(model,broken,rf,parent,local_updates=2)

    def test_untrained_frozen_Base80_remain_absent_without_fabricated_states(self):
        model,rf,parent=fixture(missing_base=True)
        with patch('turkeyworld.decoder_rf_chain_phase.configure_decoder_rf_chain',side_effect=configure):
            optimizer,report=adopt_decoder_rf_chain_optimizer(model,rf,parent)
        self.assertEqual(report['old_RF_Adam_states'],537);self.assertEqual(len(optimizer.state),599)
        self.assertEqual(report['missing_frozen_Base_state_ids'],list(range(537,617)))

    def test_partial_missing_moments_bad_D_parent_or_named_RF_changes_are_rejected(self):
        for mutate in(lambda r,d:r['optimizer']['state'].pop(0),
                       lambda r,d:r['optimizer']['state'].pop(616),
                       lambda r,d:d['optimizer']['state'][0].update(step=torch.tensor(1724.)),
                       lambda r,d:d['decoder'][next(iter(d['decoder']))].add_(.1),
                       lambda r,d:r['optimizer_parameter_names'][0].reverse()):
            model,rf,parent=fixture();mutate(rf,parent)
            with patch('turkeyworld.decoder_rf_chain_phase.configure_decoder_rf_chain',side_effect=configure):
                with self.assertRaises(ValueError):adopt_decoder_rf_chain_optimizer(model,rf,parent)


if __name__=='__main__':unittest.main()
