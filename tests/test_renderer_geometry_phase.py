"""Actual Adam/main/far-PCG continuity for the additive geometry phase."""
import copy
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_renderer_native_phase import RendererNativePhaseTests
from test_renderer_far_wall_data import SourceParent, labels_for_ids
from turkeyworld.renderer_far_wall_data import FarWallFrames
from turkeyworld.renderer_far_wall_protocol import far_wall_protocol
from turkeyworld.renderer_geometry_objectives import geometry_wall_protocol
from turkeyworld.renderer_geometry_phase import restore_renderer_geometry_parent
from turkeyworld.renderer_native_phase import _same, restore_renderer_native_rng
from turkeyworld.renderer_student import _hash_tensors


class RendererGeometryPhaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.fixture_owner = RendererNativePhaseTests()
        self.fixture_owner.setUp()
        self.addCleanup(self.fixture_owner.doCleanups)
        self.addCleanup(self.fixture_owner.tearDown)

    def far(self):
        with patch('turkeyworld.renderer_far_wall_data.source_wall_depth_labels', side_effect=labels_for_ids):
            return FarWallFrames(SourceParent(), candidate_count=9)

    def fixture(self):
        old, opt, data, saved, target, target_opt, fresh, _, _ = self.fixture_owner.fixture()
        far = self.far()
        for _ in range(2): far.batch(4)
        saved['configuration'].update(run_dir='v6', far_wall_supervision=True,
            native_phase_parent='original_v5.pt')
        saved['identity'].update(far_wall_protocol=far_wall_protocol(),far_wall_aux_pool=far.identity)
        saved['far_wall_aux_sampler'] = copy.deepcopy(far.sampler_state())
        config={**saved['configuration'],'run_dir':'v7','steps':6,
                'geometry_wall_supervision':True,'geometry_phase_parent':'v6/completed.pt'}
        identity={**copy.deepcopy(saved['identity']), 'geometry_wall_protocol':geometry_wall_protocol(),
            'geometry_wall_aux_pool':{'split':'train','fingerprint':'e'*64,
                'parent_source_fingerprint':saved['sampler']['fingerprint']},
            'geometry_wall_panel':[4,5,6,7], 'geometry_wall_receipt':{'split':'val'},
            'geometry_wall_fingerprint':'d'*64}
        return old,opt,data,far,saved,target,target_opt,fresh,self.far(),config,identity

    def adopt(self, saved, model, opt, data, far, config, identity):
        return restore_renderer_geometry_parent(saved,model,opt,data,far,identity,config,
            saved['frozen_fingerprint'],parent_checkpoint_sha256='a'*64,geometry_protocol=geometry_wall_protocol())

    def test_all_weights_moments_rng_and_both_sampler_next_draws_then_adam_match(self):
        old,opt,data,far,saved,target,new_opt,fresh,new_far,config,identity=self.fixture()
        expected_main=data.batch(16)['indices']
        expected_far=far.batch(4)['row_index']
        restore_renderer_native_rng(saved,device='cpu')
        draws=(torch.rand(3),np.random.random(3),random.random())
        torch.rand(11);np.random.random(11);random.random()
        receipt=self.adopt(saved,target,new_opt,fresh,new_far,config,identity)
        self.assertEqual(receipt['start_step'],3)
        self.assertEqual(receipt['far_wall_auxiliary_draws_at_adoption'],8)
        self.assertEqual(receipt['parent_mix_counts'],saved['sampler']['parent']['mix_counts'])
        self.assertGreater(sum(receipt['parent_mix_counts'].values()),0)
        self.assertNotIn('selected_6600_reference_retained',receipt)
        self.assertTrue(receipt['main_sampler_and_far_auxiliary_bitexact'])
        self.assertTrue(_same(saved['optimizer'],new_opt.state_dict()))
        self.assertTrue(_same(saved['sampler'],fresh.sampler_state()))
        self.assertTrue(_same(saved['far_wall_aux_sampler'],new_far.sampler_state()))
        np.testing.assert_array_equal(expected_main,fresh.batch(16)['indices'])
        self.assertTrue(torch.equal(expected_far,new_far.batch(4)['row_index']))
        self.assertTrue(torch.equal(torch.rand(3),draws[0]))
        np.testing.assert_array_equal(np.random.random(3),draws[1]);self.assertEqual(random.random(),draws[2])
        for model,optimizer in ((old,opt),(target,new_opt)):
            optimizer.zero_grad(set_to_none=True)
            sum(p.square().sum()for g in optimizer.param_groups for p in g['params']).backward()
            optimizer.step()
        self.assertTrue(_same(old.state_dict(),target.state_dict()))
        self.assertTrue(_same(opt.state_dict(),new_opt.state_dict()))
        self.assertEqual({int(v['step'])for v in new_opt.state.values()},{4})

    def test_only_new_geometry_flags_run_and_budget_may_change(self):
        _,_,_,_,saved,target,opt,data,far,config,identity=self.fixture()
        before=_hash_tensors(target.state_dict())
        for key,value in [('lr',2e-4),('batch',32),('world','otherworld.pt'),
                ('native_phase_parent','different_v5.pt'),('source_latent_weight',.03)]:
            with self.assertRaisesRegex(ValueError,'inherited settings'):
                self.adopt(saved,target,opt,data,far,{**config,key:value},identity)
            self.assertEqual(_hash_tensors(target.state_dict()),before)
            self.assertFalse(opt.state)
            self.assertEqual(data.draws,0);self.assertEqual(far.draws,0)

    def test_changed_old_panel_frozen_asset_or_new_val_pool_rejected_before_mutation(self):
        _,_,_,_,saved,target,opt,data,far,config,identity=self.fixture()
        changes=[]
        for key in ('panel_fingerprint','codec_sha256','world_sha256'):
            changes.append({**identity,key:'tampered'})
        bad=copy.deepcopy(identity);bad['far_wall_aux_pool']['fingerprint']='tampered';changes.append(bad)
        bad=copy.deepcopy(identity);bad['geometry_wall_aux_pool']['split']='val';changes.append(bad)
        bad=copy.deepcopy(identity);bad['geometry_wall_panel']=[4,4];changes.append(bad)
        bad=copy.deepcopy(identity);bad['geometry_wall_protocol']['weights_per_head']['wall_l1']=2;changes.append(bad)
        before=_hash_tensors(target.state_dict())
        for changed in changes:
            with self.assertRaises(ValueError):self.adopt(saved,target,opt,data,far,config,changed)
            self.assertEqual(_hash_tensors(target.state_dict()),before)
            self.assertFalse(opt.state)

    def test_bad_old_far_counters_adam_or_incomplete_parent_rejected_atomically(self):
        _,_,_,_,saved,target,opt,data,far,config,identity=self.fixture()
        changes=[]
        bad=copy.deepcopy(saved);bad['far_wall_aux_sampler']['draws']+=1;changes.append(bad)
        bad=copy.deepcopy(saved);next(iter(bad['optimizer']['state'].values()))['step']+=1;changes.append(bad)
        bad=copy.deepcopy(saved);bad['step']=2;changes.append(bad)
        bad=copy.deepcopy(saved);bad['flow']['latent_mean']+=.1;changes.append(bad)
        bad=copy.deepcopy(saved);del bad['rng_python'];changes.append(bad)
        before=_hash_tensors(target.state_dict())
        for changed in changes:
            with self.assertRaises(ValueError):self.adopt(changed,target,opt,data,far,config,identity)
            self.assertEqual(_hash_tensors(target.state_dict()),before)
            self.assertFalse(opt.state)
            self.assertEqual(data.draws,0);self.assertEqual(far.draws,0)


class GeometryCandidateTests(unittest.TestCase):
    def report(self, value):
        from turkeyworld.renderer_geometry_objectives import geometry_wall_report
        from test_renderer_enemy_objectives import batch
        data=batch((False,True)); target=data['target']
        arrays={'map':data['map'],'states':data['state'],'mask':data['object_mask'],
            'source':target,'base':target,'rgb':target,'codec_reference':target}
        geometry=geometry_wall_report(arrays)
        for head in ('base','rf1'):
            for band in geometry[head]['bands'].values():
                for key in geometry_wall_protocol()['weights_per_head']:
                    if band[key.removesuffix('_l1')+'_images']:band[key]=value
        stats=dict(overall=dict(frames=8,mae_mean=.03),
            visible_enemy=dict(frames=4,foreground_frames=4,foreground_mae=.09),
            categories=dict(empty_navigation=dict(frames=4,mae_mean=.02,ssim_mean=.8)))
        return dict(geometry_wall_guard_validation=geometry,base=copy.deepcopy(stats),rf1=copy.deepcopy(stats))

    def test_geometry_only_improvement_allowed_without_npc_improvement_but_not_empty_regression(self):
        from turkeyworld.renderer_geometry_phase import geometry_candidate_assessment
        before=self.report(.1);after=self.report(.09)
        gate=geometry_candidate_assessment(after,before)
        self.assertTrue(gate['passed']);self.assertFalse(gate['release_qualified'])
        self.assertFalse(geometry_candidate_assessment(before,before)['passed'])
        bad=copy.deepcopy(after);bad['rf1']['visible_enemy']['foreground_mae']+=.003
        self.assertFalse(geometry_candidate_assessment(bad,before)['passed'])
        bad=copy.deepcopy(after);bad['base']['categories']['empty_navigation']['mae_mean']+=.003
        self.assertFalse(geometry_candidate_assessment(bad,before)['passed'])

    def test_actual_auxiliary_weight_and_gradient_are_added_once_and_legacy_does_not_read(self):
        from turkeyworld.train_renderer_enemy import _add_geometry_auxiliary_loss
        from turkeyworld.renderer_geometry_objectives import geometry_refinement_loss
        from test_renderer_enemy_objectives import tiny_model,batch
        model=tiny_model(); source=batch((False,)*8)
        source['row_index']=torch.arange(8)
        class Private:
            draws=0
            def batch(self,size):
                self.draws+=size
                return source
        private=Private()
        base=next(model.base_student.parameters()).sum()*0+.3
        expected,_=geometry_refinement_loss(model,source)
        result,metrics=_add_geometry_auxiliary_loss(model,base,{},private,record_indices=True)
        self.assertTrue(torch.equal(result,base+expected))
        self.assertEqual(private.draws,8)
        self.assertEqual(metrics['geometry_wall_aux_examples'],8)
        self.assertEqual(metrics['geometry_wall_aux_weight'],1.)
        self.assertEqual(metrics['geometry_wall_aux_source_batch_indices'],list(range(8)))
        result.backward()
        self.assertGreater(sum(float(p.grad.abs().sum())for p in model.base_student.parameters()if p.grad is not None),0)
        self.assertGreater(sum(float(p.grad.abs().sum())for p in model.flow_renderer.parameters()if p.grad is not None),0)
        legacy,fields=_add_geometry_auxiliary_loss(model,base,{},None)
        self.assertIs(legacy,base);self.assertEqual(fields,{})


if __name__=='__main__':unittest.main()
