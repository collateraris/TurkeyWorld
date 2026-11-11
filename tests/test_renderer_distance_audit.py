"""Independent CPU audit of distance sampling and native tiny-object crops."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_renderer_enemy_objectives import tiny_model, batch
from turkeyworld.object_losses import decoded_ssim_loss
from turkeyworld.renderer_distance_data import DistanceBalancedFrames, distance_band
from turkeyworld.renderer_distance_training import restore_distance_parent, distance_candidate_assessment
from turkeyworld.renderer_enemy_objectives import enemy_refinement_loss, source_object_crop_ssim
from turkeyworld.renderer_refinement import frozen_fingerprint
from turkeyworld.renderer_student import _hash_tensors


class FakeParent:
    MIX={'combat':.5,'appearance':.25,'navigation':.25}

    def __init__(self, path):
        self.files=[{'path':path}]
        self.rows=[{'file':0,'frame':i,'mask_fraction':0. if i==5 else .01,'category':category}
            for i,category in enumerate(['near','mid','mid','far','dead','empty'])]
        self.rng=np.random.default_rng(173);self.draws=0
        self.case_counts=dict.fromkeys([row['category']for row in self.rows],0)
        self.mix_counts=dict.fromkeys(self.MIX,0)
        self.cohort_counts={key:{'original':0}for key in self.MIX}
        self.identity={'fingerprint':'fixed_source_train','split':'train'}

    def _mixture_select(self,size,rng,pools,cases,draws):
        ids=[]
        for _ in range(size):
            pool=max(self.MIX,key=lambda key:(draws+1)*self.MIX[key]-pools[key])
            ids.append(int(rng.integers(len(self.rows))));pools[pool]+=1;cases[pool]['original']+=1;draws+=1
        return ids,draws

    def batch(self,size,*,indices=None):
        if indices is None:
            indices,self.draws=self._mixture_select(size,self.rng,self.mix_counts,self.cohort_counts,self.draws)
            for i in indices:self.case_counts[self.rows[i]['category']]+=1
        return {'indices':np.asarray(indices,np.int64)}

    def sampler_state(self):
        return {'fingerprint':self.identity['fingerprint'],'draws':self.draws,'case_counts':dict(self.case_counts),
            'mix_counts':dict(self.mix_counts),'cohort_counts':copy.deepcopy(self.cohort_counts),
            'rng':copy.deepcopy(self.rng.bit_generator.state)}

    def restore_sampler(self,state):
        if (state['fingerprint']!=self.identity['fingerprint'] or sum(state['case_counts'].values())!=state['draws']
                or sum(state['mix_counts'].values())!=state['draws']):
            raise ValueError('Parent state changed')
        self.draws=state['draws'];self.case_counts=dict(state['case_counts']);self.mix_counts=dict(state['mix_counts'])
        self.cohort_counts=copy.deepcopy(state['cohort_counts']);self.rng.bit_generator.state=copy.deepcopy(state['rng'])


class RendererDistanceAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_source_only_bands_balanced_quotas_parent_history_and_exact_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'train.npz';states=np.zeros((6,20),np.float32)
            states[:,0:2]=.125;states[:,12]=.6;states[:,13]=1
            states[:,8]=.125+np.asarray([1.,2.,4.,5.,1.,1.])/16;states[:,9]=.125
            states[4,13]=0;states[4,12]=0
            # No RGB exists: initialization must need only Source state labels.
            np.savez(path,states=states)
            parent=FakeParent(path);parent.batch(8);old=parent.sampler_state()
            fresh=FakeParent(path)
            with patch('turkeyworld.engine.Game.step',side_effect=AssertionError('No engine in sampler')), \
                    patch('turkeyworld.engine.Game.render',side_effect=AssertionError('No RGB in sampler')):
                data=DistanceBalancedFrames(fresh)
            self.assertEqual(data.distance_buckets,{'near':[0],'mid':[1,2],'far':[3]})
            self.assertIsNone(distance_band(states[4],1));self.assertIsNone(distance_band(states[0],0))
            data.adopt_parent_sampler(old);self.assertEqual(data.draws,8);self.assertEqual(data.phase_draws,0)
            original_rng=copy.deepcopy(old['rng']);self.assertEqual(fresh.rng.bit_generator.state,original_rng)
            first=data.batch(16)['indices'];self.assertEqual(len(first),16)
            self.assertEqual(data.distance_counts,{'near':4,'mid':4,'far':4,'replay':4})
            self.assertEqual(data.draws,24);self.assertEqual(fresh.draws,12)
            self.assertEqual(sum(data.actual_distance_counts.values()),16)
            self.assertEqual(sum(fresh.mix_counts.values()),12);self.assertEqual(sum(data.case_counts.values()),24)
            saved=data.sampler_state();restored=DistanceBalancedFrames(FakeParent(path));restored.restore_sampler(copy.deepcopy(saved))
            np.testing.assert_array_equal(data.batch(32)['indices'],restored.batch(32)['indices'])
            self.assertEqual(data.sampler_state(),restored.sampler_state())
            prior=copy.deepcopy(data.sampler_state());data.batch(2,indices=[0,3]);self.assertEqual(prior,data.sampler_state())
            with self.assertRaises(ValueError):data.adopt_parent_sampler(old)
            for key,value in [('fingerprint','wrong'),('phase_draws',saved['phase_draws']+1),('protocol','old')]:
                bad=copy.deepcopy(saved);bad[key]=value
                with self.assertRaises(ValueError):DistanceBalancedFrames(FakeParent(path)).restore_sampler(bad)

    def test_native_one_pixel_crop_has_gradients_no_label_grad_and_no_background_or_area_compensation(self):
        prediction=torch.full((2,3,32,32),.2,requires_grad=True)
        target=torch.full_like(prediction,.65,requires_grad=True)
        mask=torch.zeros((2,1,32,32),requires_grad=True)
        with torch.no_grad():mask[0,0,16,16]=1;mask[1,0,8:14,8:14]=1
        expected=torch.stack([decoded_ssim_loss(prediction[:1,:,14:19,14:19],target[:1,:,14:19,14:19],mask[:1,:,14:19,14:19]),
            decoded_ssim_loss(prediction[1:,:,6:16,6:16],target[1:,:,6:16,6:16],mask[1:,:,6:16,6:16])]).mean()
        value=source_object_crop_ssim(prediction,target,mask)
        torch.testing.assert_close(value,expected,rtol=0,atol=0)
        value.backward();self.assertGreater(float(prediction.grad[0,:,16,16].abs().sum()),0)
        self.assertGreater(float(prediction.grad[1,:,8:14,8:14].abs().sum()),0)
        self.assertEqual(float(prediction.grad[0,:,:10,:].abs().sum()),0)
        self.assertIsNone(target.grad);self.assertIsNone(mask.grad)
        changed=prediction.detach().clone();changed[:,:,:5,:]=1
        torch.testing.assert_close(source_object_crop_ssim(changed,target,mask),value.detach(),rtol=0,atol=0)
        empty=torch.zeros_like(mask);candidate=prediction.detach().clone().requires_grad_()
        zero=source_object_crop_ssim(candidate,target,empty);self.assertEqual(float(zero.detach()),0)
        zero.backward();self.assertTrue(torch.equal(candidate.grad,torch.zeros_like(candidate)))

    def test_real_base_rf_default_is_exact_and_optin_crop_trains_heads_while_whole_codec_stays_frozen(self):
        model=tiny_model();source=batch((False,False))
        source['object_mask'].zero_();source['object_mask'][:,:,16,16]=1
        source['target']=source['target'].detach().requires_grad_();source['object_mask'].requires_grad_()
        frozen=frozen_fingerprint(model);codec={n:v.clone()for n,v in model.flow_renderer.codec.state_dict().items()}
        reference,old=enemy_refinement_loss(model,source,latent_weight=.1,base_regression_weight=1.)
        explicit,unchanged=enemy_refinement_loss(model,source,latent_weight=.1,base_regression_weight=1.,object_crop_weight=0.)
        self.assertTrue(torch.equal(reference,explicit));self.assertEqual(old,unchanged)
        model.zero_grad(set_to_none=True)
        value,metrics=enemy_refinement_loss(model,source,latent_weight=.1,base_regression_weight=1.,object_crop_weight=.2)
        self.assertAlmostEqual(float(value.detach()-reference.detach()),.2*(metrics['base_object_crop_ssim_loss']+metrics['rf1_object_crop_ssim_loss']),places=5)
        value.backward()
        self.assertGreater(sum(float(p.grad.abs().sum())for p in model.base_student.parameters()if p.grad is not None),0)
        self.assertGreater(sum(float(p.grad.abs().sum())for p in model.flow_renderer.parameters()if p.grad is not None),0)
        self.assertIsNone(source['target'].grad);self.assertIsNone(source['object_mask'].grad)
        optimizer=torch.optim.AdamW([p for p in model.parameters()if p.requires_grad],lr=.001);optimizer.step()
        self.assertEqual(frozen,frozen_fingerprint(model));self.assertTrue(all(torch.equal(v,model.flow_renderer.codec.state_dict()[n])for n,v in codec.items()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.flow_renderer.codec.parameters()))

    def test_new_distance_phase_restores_completed_base_flow_adam_and_historical_sampler_exactly(self):
        class Heads(torch.nn.Module):
            def __init__(self):
                super().__init__();self.base_student=torch.nn.Linear(3,2)
                self.flow_renderer=torch.nn.Linear(2,2);self.flow_renderer.register_buffer('codec_frozen',torch.tensor([.25,.75]))
        def optimizer(model):
            return torch.optim.AdamW([{'params':model.base_student.parameters(),'lr':.0001},
                {'params':model.flow_renderer.parameters(),'lr':.0001}],weight_decay=.01)
        def assert_tree(first,second):
            if torch.is_tensor(first):torch.testing.assert_close(first,second,rtol=0,atol=0)
            elif isinstance(first,dict):
                self.assertEqual(first.keys(),second.keys())
                for key in first:assert_tree(first[key],second[key])
            elif isinstance(first,(list,tuple)):
                self.assertEqual(len(first),len(second))
                for a,b in zip(first,second):assert_tree(a,b)
            else:self.assertEqual(first,second)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'states.npz';states=np.zeros((6,20),np.float32);states[:,12:14]=1
            states[:,8]=np.asarray([1.,2.,4.,5.,1.,1.])/16;np.savez(path,states=states)
            parent=FakeParent(path);parent.batch(8)
            original=Heads();old_optimizer=optimizer(original)
            inputs=torch.arange(15,dtype=torch.float32).reshape(5,3)/15
            for _ in range(2):
                old_optimizer.zero_grad(set_to_none=True);original.flow_renderer(original.base_student(inputs)).square().mean().backward();old_optimizer.step()
            identity={'world_sha256':'world','renderer_sha256':'parent_renderer','panel_fingerprint':'fixed256',
                'codec_sha256':'frozen_codec','train':{'fingerprint':'Source_train'},'val':{'fingerprint':'Source_val'}}
            configuration={'run_dir':'parent','steps':6000,'batch':16,'lr':.0001,'base_lr':.0001,
                'source_latent_weight':.1,'base_regression_weight':1.,'couple_base_flow':False,'world':'same_world'}
            saved={'base':copy.deepcopy(original.base_student.state_dict()),'flow':copy.deepcopy(original.flow_renderer.state_dict()),
                'optimizer':copy.deepcopy(old_optimizer.state_dict()),'sampler':copy.deepcopy(parent.sampler_state()),
                'identity':copy.deepcopy(identity),'configuration':copy.deepcopy(configuration),'frozen_fingerprint':'fixed_assets',
                'step':6000,'best':None,'streak':0,'enemy_streak':0}
            current_identity={**copy.deepcopy(identity),'distance_sampling':{'weights':'new'},'distance_panel_fingerprint':'new192'}
            current_configuration={**configuration,'run_dir':'new_phase','steps':10000,'distance_balance':True,
                'distance_panel':'fixed192.json','object_crop_weight':.2,'source_training_checkpoint':'last6000.pt'}
            restored=Heads();new_optimizer=optimizer(restored);data=DistanceBalancedFrames(FakeParent(path))
            before_saved=copy.deepcopy(saved)
            proof=restore_distance_parent(saved,restored,new_optimizer,data,current_identity,current_configuration,'fixed_assets')
            self.assertEqual(proof['start_step'],6000);self.assertEqual(proof['parent_draws'],8)
            self.assertTrue(proof['all_base_flow_codec_tensors_bitexact']);self.assertTrue(proof['adam_preserved'])
            self.assertTrue(proof['rng_restoration_required']);self.assertFalse(proof['automatic_active_promotion'])
            assert_tree(original.state_dict(),restored.state_dict());assert_tree(old_optimizer.state_dict(),new_optimizer.state_dict())
            assert_tree(before_saved,saved);self.assertEqual(data.parent.sampler_state(),saved['sampler'])
            self.assertEqual(data.draws,8);self.assertEqual(data.phase_draws,0)
            for model,opt in [(original,old_optimizer),(restored,new_optimizer)]:
                opt.zero_grad(set_to_none=True);model.flow_renderer(model.base_student(inputs)).square().mean().backward();opt.step()
            assert_tree(original.state_dict(),restored.state_dict());assert_tree(old_optimizer.state_dict(),new_optimizer.state_dict())
            for invalid_identity,invalid_config,invalid_frozen in [
                ({**current_identity,'codec_sha256':'changed'},current_configuration,'fixed_assets'),
                (current_identity,{**current_configuration,'source_latent_weight':.03},'fixed_assets'),
                (current_identity,{**current_configuration,'lr':.0002},'fixed_assets'),
                (current_identity,current_configuration,'changed_assets')]:
                candidate=Heads();digest=_hash_tensors(candidate.state_dict())
                with self.assertRaises(ValueError):restore_distance_parent(saved,candidate,optimizer(candidate),DistanceBalancedFrames(FakeParent(path)),
                    invalid_identity,invalid_config,invalid_frozen)
                self.assertEqual(digest,_hash_tensors(candidate.state_dict()))

    def test_distance_release_requires_same_panel_both_heads_all_three64_frame_bands_and_finite_full_mae(self):
        baseline={'panel_fingerprint':'fixed192',**{head:{'categories':{band:{'frames':64,'foreground_mae':.2,'mae_mean':.03}
            for band in ['near','mid','far']}}for head in ['base','rf1']}}
        improved=copy.deepcopy(baseline)
        for head in ['base','rf1']:
            for band in ['near','mid','far']:improved[head]['categories'][band]['foreground_mae']=.19
        self.assertTrue(distance_candidate_assessment(improved,baseline)['passed'])
        variants=[]
        for head in ['base','rf1']:
            for band in ['near','mid','far']:
                for field,value in [('frames',63),('foreground_mae',.2),('mae_mean',.04),('mae_mean',float('nan')),('foreground_mae',float('inf'))]:
                    report=copy.deepcopy(improved);report[head]['categories'][band][field]=value;variants.append(report)
                report=copy.deepcopy(improved);del report[head]['categories'][band]['mae_mean'];variants.append(report)
        variants.extend([{**copy.deepcopy(improved),'panel_fingerprint':'other'},
                         {**copy.deepcopy(improved),'rf1':{}},{}])
        for report in variants:
            result=distance_candidate_assessment(report,baseline);self.assertFalse(result['passed']);self.assertTrue(result['failures'])
        missing_old=copy.deepcopy(baseline);del missing_old['rf1']['categories']['far']['mae_mean']
        self.assertFalse(distance_candidate_assessment(improved,missing_old)['passed'])


if __name__=='__main__':unittest.main()
