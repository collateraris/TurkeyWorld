"""Approved codec E/D changes preserve RF/Base, coordinates and source lineage."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from torch import nn

from turkeyworld.base_student import build_base_student,export_base_student,load_base_student
from turkeyworld.codec_enemy_swap import prepare_codec_enemy_swap
from turkeyworld.compact_codec import build_compact_codec
from turkeyworld.distill_codec import export_codec_student
from turkeyworld.latent_flow_renderer import LatentFlowRenderer
from turkeyworld.models import WorldModelConfig,STATE_DIM
from turkeyworld.renderer_student import (build_renderer_student,export_renderer_student,
    load_renderer_student,codec_tensor_fingerprint,renderer_condition_identity)


class SourceTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.config=WorldModelConfig(kind='teacher',architecture='vjepa2_ac',image_size=32,
            endless=True,navigation_only=True,renderer_mode='flow_matching',renderer_codec='vqgan',
            renderer_codec_config=dict(image_size=32,base_channels=64,latent_channels=128,
                embedding_dim=16,n_embed=1488,data_init=False),flow_steps=2,
            flow_width=16,flow_depth=1,flow_heads=2,flow_patch_size=2,flow_block_conditioning=True,
            spatial_entities=True,spatial_combat=True,game_training_v5=True,
            vjepa_width=16,vjepa_encoder_depth=1,vjepa_predictor_depth=1,vjepa_heads=2)
        self.vjepa_backend=nn.Module()
        self.vjepa_backend.renderer=nn.Linear(STATE_DIM,12)
        self.vjepa_backend.feature_head=nn.Linear(16,128)
        self.vjepa_backend.core=nn.Linear(STATE_DIM,16)
        self.flow_renderer=LatentFlowRenderer(kind='teacher',image_size=32,mode='flow_matching',
            steps=2,width=16,depth=1,heads=2,codec_config=self.config.renderer_codec_config,
            spatial_conditioning=True,patch_size=2,block_conditioning=True,improved_loss=True)
        self.flow_renderer.codec_loaded.fill_(True)
        self.flow_renderer.latent_mean.fill_(.1);self.flow_renderer.latent_std.fill_(.7)


class CodecEnemySwapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def setUp(self):
        temporary=TemporaryDirectory();self.addCleanup(temporary.cleanup);self.root=Path(temporary.name)
        self.source=SourceTeacher().eval().requires_grad_(False)
        compact=build_compact_codec(self.source.flow_renderer.codec)
        self.codec_path=self.root/'parent_codec.pt'
        export_codec_student(compact,self.source.flow_renderer,self.codec_path,training_steps=6000,
            report={'passed':False,'scope':'original teacher imitation','summary':{'ssim_mean':.92}})
        initial=deepcopy(self.source)
        export_base_student(initial,build_base_student(initial),self.root/'base.pt',training_steps=1100)
        load_base_student(initial,self.root/'base.pt')
        flow=build_renderer_student(initial,steps=1,width=16,depth=1,heads=2,mlp_hidden=32,
            codec_checkpoint=self.codec_path,allow_unqualified_codec=True)
        self.renderer_path=self.root/'parent_rf.pt'
        export_renderer_student(initial,flow,self.renderer_path,training_steps=1500)
        self.working=deepcopy(self.source)
        load_renderer_student(self.working,self.renderer_path,allow_unqualified_codec=True)
        self.payload=torch.load(self.codec_path,weights_only=False)
        self.payload['model']['decoder.conv_out.bias'].add_(.1)
        self.payload['model']['quant_conv.bias'].add_(.02)
        self.payload['training_steps']=6100
        self.payload['quality_gate']={'passed':False,'scope':'Source NPC reconstruction',
            'summary':{'npc_mae':.1},'thresholds':{'npc_mae_max':.05}}
        self.payload['refinement_provenance']={'parent_compact_step':6000,
            'old_imitation_quality_gate':{'passed':False}}

    def prepare(self,payload=None,**kwargs):
        return prepare_codec_enemy_swap(self.source,self.working,payload or self.payload,
                                       allow_unqualified_codec=kwargs.get('allow',True))

    def test_private_swap_changes_only_codec_and_preserves_inputs_rng_noise_and_base(self):
        source={k:v.clone() for k,v in self.source.state_dict().items()}
        working={k:v.clone() for k,v in self.working.state_dict().items()}
        rng=torch.random.get_rng_state().clone()
        with patch.object(torch.cuda,'manual_seed_all',side_effect=AssertionError('No CUDA seeding')):
            flow,meta=self.prepare()
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        self.assertIsNot(flow,self.working.flow_renderer)
        for key,value in self.working.flow_renderer.state_dict().items():
            if not key.startswith('codec.'):self.assertTrue(torch.equal(value,flow.state_dict()[key]),key)
        self.assertNotEqual(codec_tensor_fingerprint(flow.codec),codec_tensor_fingerprint(self.working.flow_renderer.codec))
        self.assertTrue(all(torch.equal(v,self.source.state_dict()[k]) for k,v in source.items()))
        self.assertTrue(all(torch.equal(v,self.working.state_dict()[k]) for k,v in working.items()))
        self.assertTrue(meta['rf_noncodec_tensors_bitexact'] and meta['base_bitexact'])
        self.assertFalse(meta['automatic_deployment'] or meta['gameplay_qualified'])
        self.assertFalse(meta['codec_quality_qualified'])
        self.assertTrue(meta['experimental_preview'])
        self.assertEqual(flow._codec_student_bundle['training_steps'],6100)
        self.assertEqual(flow._codec_student_identity,meta['student_codec_identity'])
        self.assertFalse(flow.codec.training or flow.codec.quantize.training)
        self.assertTrue(all(not p.requires_grad for p in flow.codec.parameters()))

    def test_unqualified_requires_explicit_optin_but_qualified_source_gate_retains_its_scope(self):
        with self.assertRaisesRegex(ValueError,'Unqualified'):
            self.prepare(allow=False)
        payload=deepcopy(self.payload);payload['quality_gate']['passed']=True
        flow,meta=self.prepare(payload,allow=False)
        self.assertTrue(meta['codec_quality_qualified'])
        self.assertFalse(meta['experimental_preview'])
        self.assertEqual(flow._renderer_codec_distillation['quality_gate']['scope'],'Source NPC reconstruction')
        self.assertFalse(meta['gameplay_qualified'])
        with self.assertRaises(ValueError):
            prepare_codec_enemy_swap(self.source,self.working,self.payload,allow_unqualified_codec=1)

    def test_wrong_original_fingerprint_quantizer_norm_or_training_proof_fails_transactionally(self):
        original={k:v.clone() for k,v in self.working.state_dict().items()}
        mutations=(lambda p:p.update(teacher_codec_fingerprint='wrong'),
            lambda p:p['model']['quantize.embedding.weight'].add_(.01),
            lambda p:p['model']['quantize.usage_ema'].add_(1),
            lambda p:p['latent_normalization']['mean'].__setitem__(0,.2),
            lambda p:p.update(training_steps=0),lambda p:p.update(frozen=False),
            lambda p:p['quality_gate'].update(passed='yes'))
        for mutate in mutations:
            payload=deepcopy(self.payload);mutate(payload)
            with self.assertRaises(ValueError):self.prepare(payload)
            self.assertTrue(all(torch.equal(v,self.working.state_dict()[k]) for k,v in original.items()))

    def test_wrong_live_rf_noise_or_unapproved_base_rejected_without_source_changes(self):
        before=renderer_condition_identity(self.source)
        self.working.flow_renderer.fixed_noise.add_(.01)
        with self.assertRaisesRegex(ValueError,'noise'):self.prepare()
        self.working.flow_renderer.fixed_noise.copy_(self.source.flow_renderer.fixed_noise)
        with torch.no_grad():self.working.base_student.rgb.bias.add_(.01)
        with self.assertRaisesRegex(ValueError,'Base'):self.prepare()
        self.assertEqual(renderer_condition_identity(self.source),before)

    def test_roundtrip_exports_original_source_identity_with_new_embedded_codec_and_same_base(self):
        flow,meta=self.prepare()
        reference=deepcopy(self.source)
        load_base_student(reference,self.working.base_student_bundle)
        output=self.root/'new_rf.pt'
        export_renderer_student(reference,flow,output,training_steps=1600,
            metrics={'codec_replacement':meta,'source_quality_passed':False})
        bundle=torch.load(output,weights_only=False)
        self.assertEqual(bundle['conditioning_identity'],renderer_condition_identity(self.source))
        self.assertEqual(bundle['student_codec_identity']['codec_sha256'],codec_tensor_fingerprint(flow.codec))
        self.assertNotEqual(bundle['student_codec_identity']['codec_sha256'],bundle['conditioning_identity']['codec_sha256'])
        loaded=deepcopy(self.source);before={k:v.clone() for k,v in loaded.state_dict().items() if not k.startswith('flow_renderer.')}
        load_renderer_student(loaded,output,allow_unqualified_codec=True)
        self.assertTrue(all(torch.equal(v,loaded.state_dict()[k]) for k,v in before.items()))
        self.assertTrue(all(torch.equal(v,loaded.base_student.state_dict()[k]) for k,v in self.working.base_student.state_dict().items()))
        self.assertEqual(codec_tensor_fingerprint(loaded.flow_renderer.codec),codec_tensor_fingerprint(flow.codec))
        self.assertEqual(loaded.flow_renderer.steps,1)

    def test_full_rf_forward_uses_new_encoder_and_decoder_with_original_noise_and_frozen_quantizer(self):
        flow,meta=self.prepare();flow.eval()
        base=torch.rand(1,3,32,32);state=torch.zeros(1,20);state[:,:6]=torch.tensor([.2,.3,0,1,1,1])
        state[:,8:10]=torch.tensor([.5,.3]);state[:,12:14]=torch.tensor([.6,1.])
        grid=torch.zeros(1,16,16,dtype=torch.long)
        quantizer={k:v.clone() for k,v in flow.codec.quantize.state_dict().items()}
        with (torch.no_grad(),patch('turkeyworld.engine.Game.step',side_effect=AssertionError('No Source feedback')),
              patch.object(flow.codec,'encode',wraps=flow.codec.encode) as encode,
              patch.object(flow.codec,'decode',wraps=flow.codec.decode) as decode,
              patch.object(flow,'velocity',wraps=flow.velocity) as velocity):
            rgb=flow.sample(base,state,map_condition=grid)
        self.assertEqual((encode.call_count,decode.call_count,velocity.call_count),(1,1,1))
        self.assertEqual(rgb.shape,(1,3,32,32));self.assertTrue(torch.isfinite(rgb).all())
        self.assertTrue(all(torch.equal(v,flow.codec.quantize.state_dict()[k]) for k,v in quantizer.items()))
        self.assertTrue(torch.equal(flow.fixed_noise,self.working.flow_renderer.fixed_noise))
        self.assertTrue(torch.equal(flow.latent_mean,self.source.flow_renderer.latent_mean))
        self.assertTrue(torch.equal(flow.latent_std,self.source.flow_renderer.latent_std))


if __name__=='__main__':unittest.main()
