from copy import deepcopy
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from turkeyworld.cache_enemy_decoder_hypotheses import (
    build_hypothesis_cache,initialize_cache_device,predict_hypotheses,rng_fingerprint,
)
from turkeyworld.enemy_decoder_data import EnemyDecoderData
from test_enemy_decoder_data import paired_fixture


class FrozenWorld(nn.Module):
    def __init__(self):
        super().__init__();self.weight=nn.Parameter(torch.tensor(.1))
        self.register_buffer('constant',torch.tensor(1.));self.observations=[];self.priors=[]
    def encode_map(self,grid):return grid.float()
    def initialize_from_image(self,image,state,features):
        self.observations.append((image.clone(),state.clone()))
        random.random();np.random.rand();torch.rand(1)
        return image.mean((1,2,3))[:,None]
    def transition(self,state,action,features,memory):
        self.priors.append((state.clone(),action.clone()))
        result=state.clone();result[:,0]+=action[:,0]*.01+memory[:,0]*.01
        return {'state':result,'features':memory.expand(-1,128),'memory':memory}


class HypothesisCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_default_cuda_device_resolves_index_before_initialization_without_gpu(self):
        calls=[]
        with patch.object(torch.cuda,'set_device',side_effect=lambda i:calls.append(('device',i))), \
                patch.object(torch.cuda,'init',side_effect=lambda:calls.append(('init',None))):
            self.assertEqual(initialize_cache_device('cuda'),torch.device('cuda:0'))
            self.assertEqual(calls,[('device',0),('init',None)])
            calls.clear()
            self.assertEqual(initialize_cache_device('cuda:2'),torch.device('cuda:2'))
            self.assertEqual(calls,[('device',2),('init',None)])
            calls.clear();self.assertEqual(initialize_cache_device('cpu'),torch.device('cpu'))
            self.assertFalse(calls)

    def test_dynamic_prior_is_causal_future_poison_invariant_and_preserves_all_rng(self):
        model=FrozenWorld().eval();device=torch.device('cpu')
        batch={'dynamic':torch.tensor([True,False]),'previous_state':torch.ones(2,20)*.2,
            'previous_rgb':torch.ones(2,3,32,32)*.3,'action':torch.ones(2,4)*.1,
            'map':torch.zeros(2,16,16,dtype=torch.long),'state':torch.ones(2,20)*.9,
            'hr_source':torch.ones(2,3,128,128),'mask_hr':torch.ones(2,1,128,128)}
        before=rng_fingerprint(device);weights=deepcopy(model.state_dict())
        first=predict_hypotheses(model,batch)
        self.assertEqual(before,rng_fingerprint(device))
        self.assertEqual(first['valid'].tolist(),[True,False])
        self.assertFalse(first['pred_state'][1].any())
        self.assertTrue(np.allclose(first['pred_state'][0,0],.204))
        self.assertEqual(len(model.observations[0][0]),1)
        self.assertTrue(torch.equal(model.observations[0][1],batch['previous_state'][:1]))
        poisoned={**batch,'state':torch.full_like(batch['state'],float('nan')),
            'hr_source':torch.full_like(batch['hr_source'],float('nan')),
            'mask_hr':torch.full_like(batch['mask_hr'],float('nan'))}
        other=predict_hypotheses(model,poisoned)
        self.assertTrue(np.array_equal(first['pred_state'],other['pred_state']))
        self.assertEqual(before,rng_fingerprint(device))
        self.assertTrue(all(torch.equal(v,model.state_dict()[k])for k,v in weights.items()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_static_rows_do_not_initialize_or_transition_even_without_future_or_past_fields(self):
        model=FrozenWorld()
        result=predict_hypotheses(model,{'dynamic':torch.tensor([False,False])})
        self.assertFalse(result['valid'].any());self.assertFalse(result['pred_state'].any())
        self.assertEqual(model.observations,[]);self.assertEqual(model.priors,[])

    def test_cache_complete_resume_identity_and_source_bytes_are_preserved(self):
        with TemporaryDirectory()as directory:
            paired=paired_fixture(directory);output=Path(directory)/'sidecar';model=FrozenWorld().train()
            model_before=deepcopy(model.state_dict());rng_before=rng_fingerprint(torch.device('cpu'))
            result=build_hypothesis_cache(paired,output,model,world_sha256='world-one',world_step=5510,batch_size=2)
            self.assertEqual(result['status'],'complete');self.assertEqual(result['current'],16)
            self.assertEqual(result['valid_dynamic_rows'],0);self.assertEqual(result['invalid_static_rows'],16)
            self.assertTrue(result['rng_state_unchanged']);self.assertTrue(result['frozen_model_tensors_bitexact'])
            self.assertEqual(result['optimizer_steps'],0);self.assertFalse(result['autonomous_450_step_claim'])
            self.assertEqual(rng_before,rng_fingerprint(torch.device('cpu')))
            self.assertTrue(model.training);self.assertTrue(model.weight.requires_grad)
            self.assertTrue(all(torch.equal(v,model.state_dict()[k])for k,v in model_before.items()))
            raw=(output/'manifest.json').read_bytes()
            build_hypothesis_cache(paired,output,model,world_sha256='world-one',world_step=5510,batch_size=2)
            self.assertEqual(raw,(output/'manifest.json').read_bytes())
            with self.assertRaisesRegex(ValueError,'identity changed'):
                build_hypothesis_cache(paired,output,model,world_sha256='world-two',batch_size=2)
            data=EnemyDecoderData(paired,'val',hypotheses_root=output)
            self.assertFalse(data.batch(8,indices=data.fixed_panel())['hypothesis'].any())
            json.dumps(result,allow_nan=False)


if __name__=='__main__':unittest.main()
