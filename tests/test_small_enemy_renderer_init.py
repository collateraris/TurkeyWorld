from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

import test_codec_enemy_swap as fixtures
from turkeyworld.renderer_refinement import configure_refinement,export_refined_bundle
from turkeyworld.renderer_student import load_renderer_student,_hash_tensors
from turkeyworld.small_enemy_renderer_init import prepare_small_renderer_initial,restore_small_renderer_parent


class SmallRendererInitializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def setUp(self):
        self.fixture=fixtures.CodecEnemySwapTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.root=self.fixture.root;self.source=self.fixture.source;self.working=self.fixture.working
        configure_refinement(self.working)
        self.opt=self.optimizer(self.working)
        sum(p.square().mean()for g in self.opt.param_groups for p in g['params']).backward();self.opt.step();self.opt.zero_grad(set_to_none=True)
        self.saved={'base':deepcopy(self.working.base_student.state_dict()),'flow':deepcopy(self.working.flow_renderer.state_dict()),
            'optimizer':deepcopy(self.opt.state_dict()),'step':8000,
            'sampler':{'parent':{'rng':np.random.default_rng(173).bit_generator.state,'draws':1234},'phase_draws':8000}}
        self.parent=self.root/'parent8000.pt';torch.save(self.saved,self.parent)
        self.assets=self.root/'assets';self.assets.mkdir();(self.assets/'world_navigation.pt').write_bytes(b'immutable current4250 world')
        export_refined_bundle(deepcopy(self.source),self.working,self.assets,updates=8000,
            initial_renderer_steps=9600,initial_base_steps=9200,report={'passed':False},provenance={'old_world_step':3780})
        (self.assets/'renderer_student.pt').rename(self.assets/'renderer_parent_navigation.pt')
        self.new_codec=self.root/'selected_codec.pt';torch.save(self.fixture.payload,self.new_codec)
        # Deliberately different current dynamics; renderer dependency assets
        # are the same and are validated by the real strict bundle loaders.
        with torch.no_grad():self.source.vjepa_backend.core.weight.add_(.01)

    @staticmethod
    def optimizer(model):
        return torch.optim.AdamW([{'params':model.base_student.parameters(),'lr':1e-4},
            {'params':[p for p in model.flow_renderer.parameters()if p.requires_grad],'lr':2e-4}],weight_decay=.01)

    def load(self,*args,**kwargs):return deepcopy(self.source),{'training_steps':4250}

    def test_private_init_strict_reload_only_changes_codec_archives_adam_and_is_idempotent(self):
        output=self.root/'new_initial';parent_bytes=self.parent.read_bytes();rng=torch.get_rng_state().clone()
        with patch('turkeyworld.small_enemy_renderer_init.load_world_checkpoint',side_effect=self.load):
            path=prepare_small_renderer_initial(self.assets,self.new_codec,output,self.parent)
        self.assertTrue(torch.equal(torch.get_rng_state(),rng))
        self.assertEqual(self.parent.read_bytes(),parent_bytes)
        self.assertEqual((output/'renderer_adam_parent.pt').read_bytes(),parent_bytes)
        restored=deepcopy(self.source);meta=load_renderer_student(restored,path,allow_unqualified_codec=True)
        self.assertEqual(meta['training_steps'],17600);self.assertEqual(restored.base_student_bundle['training_steps'],17200)
        self.assertTrue(all(torch.equal(v,restored.base_student.state_dict()[k])for k,v in self.saved['base'].items()))
        for key,value in self.saved['flow'].items():
            if not key.startswith('codec.')or key.startswith('codec.quantize.'):
                self.assertTrue(torch.equal(value,restored.flow_renderer.state_dict()[key]),key)
        self.assertNotEqual(_hash_tensors(self.working.flow_renderer.codec.state_dict()),_hash_tensors(restored.flow_renderer.codec.state_dict()))
        bytes_before=path.read_bytes()
        self.assertEqual(path,prepare_small_renderer_initial(self.assets,self.new_codec,output,self.parent))
        self.assertEqual(path.read_bytes(),bytes_before)
        with self.new_codec.open('ab')as stream:stream.write(b'changed')
        with self.assertRaises(ValueError):prepare_small_renderer_initial(self.assets,self.new_codec,output,self.parent)

    def test_adam_adoption_preserves_new_ed_weights_and_every_moment_option_and_sampler_rng(self):
        from turkeyworld.codec_enemy_swap import prepare_codec_enemy_swap
        flow,_=prepare_codec_enemy_swap(self.source,self.working,self.new_codec,allow_unqualified_codec=True)
        self.working.flow_renderer=flow;configure_refinement(self.working)
        opt=self.optimizer(self.working);before={k:v.clone()for k,v in self.working.state_dict().items()}
        proof=restore_small_renderer_parent(self.saved,self.working,opt)
        self.assertTrue(proof['adam_all_moments_options_bitexact']);self.assertEqual(proof['start_step'],8000)
        self.assertGreater(proof['changed_codec_ed_tensors'],0)
        self.assertEqual(proof['parent_sampler_rng'],self.saved['sampler']['parent']['rng'])
        self.assertEqual(opt.state_dict()['param_groups'],self.saved['optimizer']['param_groups'])
        for index,values in self.saved['optimizer']['state'].items():
            for name,value in values.items():self.assertTrue(torch.equal(value,opt.state_dict()['state'][index][name]))
        self.assertTrue(all(torch.equal(v,self.working.state_dict()[k])for k,v in before.items()))
        self.assertTrue(all(p.grad is None for p in self.working.flow_renderer.codec.parameters()))

    def test_tampered_base_quantizer_or_parameter_order_fails_before_optimizer_mutation(self):
        opt=self.optimizer(self.working)
        with torch.no_grad():self.working.base_student.rgb.bias.add_(.01)
        with self.assertRaisesRegex(ValueError,'Base'):restore_small_renderer_parent(self.saved,self.working,opt)
        self.assertFalse(opt.state)
        self.working.base_student.load_state_dict(self.saved['base'])
        with torch.no_grad():self.working.flow_renderer.codec.quantize.usage_ema.add_(1.)
        with self.assertRaisesRegex(ValueError,'quantizer'):restore_small_renderer_parent(self.saved,self.working,opt)
        self.assertFalse(opt.state)
        self.working.flow_renderer.load_state_dict(self.saved['flow'])
        opt.param_groups.reverse()
        with self.assertRaisesRegex(ValueError,'ordered'):restore_small_renderer_parent(self.saved,self.working,opt)
        self.assertFalse(opt.state)


if __name__=='__main__':unittest.main()
