from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import json
import hashlib

import torch

from turkeyworld.compact_codec import CompactCodec
from turkeyworld.models import build_upscaler, load_sr_checkpoint
from turkeyworld.renderer_student import _hash_tensors
from turkeyworld.train_sr_combat import load_augmentation_codec, source_sr_loss, validate_continuation
from turkeyworld.vqgan import VQGANConfig


class SRCombatTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_existing_sr_weights_roundtrip_and_source_update_leaves_full_codec_frozen(self):
        with TemporaryDirectory() as folder:
            root=Path(folder)
            model=build_upscaler(dict(architecture='edsr',channels=64,blocks=16,scale=4,res_scale=.1))
            self.assertEqual(sum(p.numel()for p in model.parameters()),1517571)
            with torch.no_grad():model.output.weight.fill_(.002)
            parent=root/'parent.pt';torch.save(dict(stage='sr',config=model.config,model=model.state_dict(),training_steps=20000),parent)
            parent_bytes=parent.read_bytes()
            loaded,metadata=load_sr_checkpoint(parent)
            self.assertEqual(metadata['training_steps'],20000)
            self.assertTrue(all(torch.equal(v,loaded.state_dict()[name])for name,v in model.state_dict().items()))
            config=VQGANConfig(image_size=32,base_channels=4,latent_channels=8,embedding_dim=4,n_embed=8,data_init=False)
            original=CompactCodec(config).eval().requires_grad_(False)
            codec_path=root/'codec.pt';torch.save(dict(stage='distilled_vqgan_codec',codec_config=original.get_config(),
                model=original.state_dict(),training_steps=10,frozen=True,fixed_codebook=True,quality_gate={'passed':False}),codec_path)
            codec,info=load_augmentation_codec(codec_path,'cpu')
            self.assertFalse(info['quality_gate_passed'])
            before=_hash_tensors(codec.state_dict());sr_before=loaded.output.weight.detach().clone()
            lr=torch.rand(2,3,32,32);target=torch.rand(2,3,128,128,requires_grad=True)
            mask=torch.zeros(2,1,128,128);mask[0,:,20:70,30:90]=1
            with torch.no_grad():augmented=codec(lr)['reconstruction']
            optimizer=torch.optim.AdamW(loaded.parameters(),lr=5e-5)
            loss,parts=source_sr_loss(loaded(augmented),target,mask)
            loss.backward();optimizer.step()
            self.assertTrue(torch.isfinite(loss));self.assertGreater(parts['pixel'],0)
            self.assertFalse(torch.equal(sr_before,loaded.output.weight))
            self.assertIsNone(target.grad)
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in codec.parameters()))
            self.assertFalse(codec.training);self.assertFalse(codec.quantize.training)
            self.assertEqual(before,_hash_tensors(codec.state_dict()))
            self.assertEqual(parent.read_bytes(),parent_bytes)

    def test_budget_extension_requires_immutable_source_checkpoint_and_original_plan(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source.pt';source.write_bytes(b'archived full Adam checkpoint')
            sha=hashlib.sha256(source.read_bytes()).hexdigest()
            plan=dict(reason='explicit_user_request',from_completed_updates=2000,from_budget_updates=2000,
                to_budget_updates=10000,training_plan_sha256='original',source_checkpoint=str(source),source_checkpoint_sha256=sha)
            path=root/'continuation.json';path.write_text(json.dumps(plan))
            saved=dict(refinement_updates=2000,run_state={'total':2000})
            accepted=validate_continuation(path,saved,10000,'original',sha)
            self.assertEqual(accepted['from_completed_updates'],2000)
            with self.assertRaises(ValueError):validate_continuation(path,saved,10000,'changed',sha)
            with self.assertRaises(ValueError):validate_continuation(path,saved,10000,'original','wrong_checkpoint')
            with self.assertRaises(ValueError):validate_continuation(path,saved,20000,'original',sha)
            resumed=dict(refinement_updates=2100,run_state={'total':10000})
            self.assertEqual(validate_continuation(path,resumed,10000,'original','later_cp')['to_budget_updates'],10000)


if __name__=='__main__':unittest.main()
