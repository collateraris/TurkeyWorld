from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch
from torch import nn

from turkeyworld.cache_sr_neural_pairs import build_cache, render_cache_batch
from turkeyworld.sr_neural_pairs_data import SourcePairData, SemanticSRData, collect_source_pairs
from turkeyworld.small_enemy_data import digest
from test_sr_neural_pairs_data import source_snapshots


class Codec(nn.Module):
    def forward(self, image):
        return {'reconstruction':image*.8+.1}


class FrozenRenderer(nn.Module):
    """Tiny CPU facade with spies; no real world/checkpoint/GPU is loaded."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.1))
        self.register_buffer('constant',torch.tensor(.25))
        self.flow_renderer = nn.Module(); self.flow_renderer.codec = Codec()
        self.map_sizes = []; self.observations = []; self.priors = []; self.render_queries = []

    def encode_map(self, grid):
        self.map_sizes.append(len(grid))
        return grid.float()

    def initialize_from_image(self, image, state, features):
        self.observations.append((image.clone(),state.clone()))
        return image.mean((1,2,3),keepdim=False)[:,None]

    def transition(self, state, action, features, memory):
        self.priors.append((state.clone(),action.clone(),memory.clone()))
        value = memory*.1+state[:,:1]*.1+action[:,:1]*.1
        return {'features':value.expand(-1,128),'memory':memory}

    def render(self, state, features, map_features):
        self.render_queries.append((state.clone(),features.clone()))
        value = (self.constant+self.weight+state[:,:1]*.1+features.mean(1,keepdim=True)*.1).clamp(0,1)
        return value[:,:,None,None].expand(-1,3,32,32)


def identity():
    return {'world_sha256':'fake-fixed-world','renderer_sha256':'fake-fixed-npr',
        'codec_tensor_sha256':'fake-fixed-codec','future_rgb_features':False,'paired_same_pose_map_tick':True}


class CacheNeuralSRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_prior_reads_only_genuine_past_and_static_uses_zero_features(self):
        model = FrozenRenderer()
        batch = {'state':torch.ones(2,20)*.5,'map':torch.zeros(2,16,16,dtype=torch.long),
            'dynamic':torch.tensor([True,False]),'previous_state':torch.ones(2,20)*.2,
            'previous_rgb':torch.ones(2,3,32,32)*.3,'action':torch.ones(2,4)*.1,
            'lr_source':torch.ones(2,3,32,32)*.9,
            'hr_source':torch.ones(2,3,128,128),'mask_hr':torch.ones(2,1,128,128)}
        before = deepcopy(model.state_dict())
        first = render_cache_batch(model,batch,include_codec=False)['lr_npr']
        self.assertEqual(model.map_sizes,[1,2])
        self.assertEqual(len(model.observations[0][0]),1)
        self.assertTrue(torch.equal(model.observations[0][0],batch['previous_rgb'][:1]))
        self.assertTrue(torch.equal(model.priors[0][0],batch['previous_state'][:1]))
        self.assertTrue(torch.equal(model.priors[0][1],batch['action'][:1]))
        self.assertEqual(float(model.render_queries[0][1][1].abs().sum()),0.)
        # Poison all current/future RGB/labels. NPR must remain byte-identical.
        changed = {**batch,'lr_source':torch.zeros_like(batch['lr_source']),
            'hr_source':torch.full_like(batch['hr_source'],float('nan')),
            'mask_hr':torch.full_like(batch['mask_hr'],float('nan'))}
        self.assertTrue(np.array_equal(first,render_cache_batch(model,changed,include_codec=False)['lr_npr']))
        changed_past = {**batch,'previous_rgb':torch.zeros_like(batch['previous_rgb'])}
        other = render_cache_batch(model,changed_past,include_codec=False)['lr_npr']
        self.assertFalse(np.array_equal(first[0],other[0]))
        self.assertTrue(np.array_equal(first[1],other[1]))
        self.assertTrue(all(torch.equal(before[k],v) for k,v in model.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_cache_preserves_all_source_arrays_and_fixed_split_while_adding_actual_inputs(self):
        with TemporaryDirectory() as temporary:
            source = Path(temporary)/'source'
            for split in ('train','val'):
                source_snapshots(source,split)
            paired = Path(temporary)/'paired_source'
            original = collect_source_pairs(paired,[source],train_count=8,val_count=8,workers=1,shard_size=3)
            original_manifest = (paired/'manifest.json').read_bytes()
            model = FrozenRenderer().train()
            model.flow_renderer.codec.eval()
            modes = {name:m.training for name,m in model.named_modules()}
            flags = [p.requires_grad for p in model.parameters()]
            cache = Path(temporary)/'cache'
            result = build_cache(paired,cache,model,cache_identity=identity(),batch_size=2)
            self.assertEqual(result['status'],'complete')
            self.assertTrue(result['frozen_model_tensors_bitexact'])
            self.assertEqual(result['optimizer_steps'],0)
            self.assertEqual(result['records'],original['records'])
            self.assertEqual(original_manifest,(paired/'manifest.json').read_bytes())
            self.assertEqual(modes,{name:m.training for name,m in model.named_modules()})
            self.assertEqual(flags,[p.requires_grad for p in model.parameters()])
            for shard in original['shards']:
                self.assertEqual(digest(paired/shard['path']),shard['sha256'])
                with np.load(paired/shard['path']) as old,np.load(cache/shard['path']) as new:
                    self.assertTrue(all(np.array_equal(old[k],new[k]) for k in old.files))
                    self.assertEqual(new['lr_npr'].shape,new['lr_source'].shape)
                    self.assertFalse(np.array_equal(new['lr_npr'],new['lr_source']))
                    self.assertEqual(new['lr_npr'].dtype,np.uint8)
                    self.assertIn('codec_lr',new.files)
            train = SemanticSRData(cache,'train')
            val = SemanticSRData(cache,'val',exclude_scene_keys=train.scene_keys)
            self.assertEqual(len(val.fixed_panel()),8)
            batch = val.batch(4,indices=[0,1,2,3])
            self.assertEqual(batch['lr_npr'].shape,(4,3,32,32))
            self.assertEqual(batch['hr_source'].shape,(4,3,128,128))
            self.assertFalse(train.scene_keys&val.scene_keys)
            self.assertFalse(np.array_equal(batch['lr_npr'].numpy(),batch['lr_source'].numpy()))
            cached_bytes = (cache/'manifest.json').read_bytes(); calls = len(model.render_queries)
            build_cache(paired,cache,model,cache_identity=identity(),batch_size=2)
            self.assertEqual(cached_bytes,(cache/'manifest.json').read_bytes())
            self.assertEqual(calls,len(model.render_queries))
            with self.assertRaisesRegex(ValueError,'identity changed'):
                build_cache(paired,cache,model,cache_identity={**identity(),'renderer_sha256':'other'})
            json.dumps(result,allow_nan=False)

    def test_future_rgb_identity_and_source_overwrite_are_rejected(self):
        with TemporaryDirectory() as temporary:
            p=Path(temporary)
            with self.assertRaisesRegex(ValueError,'prohibit future RGB'):
                build_cache(p,p/'out',FrozenRenderer(),cache_identity={**identity(),'future_rgb_features':True})
            with self.assertRaisesRegex(ValueError,'overwrite'):
                build_cache(p,p,FrozenRenderer(),cache_identity=identity())


if __name__ == '__main__':
    unittest.main()
