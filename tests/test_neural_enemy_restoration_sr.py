from copy import deepcopy
import io
import unittest
from unittest.mock import patch

import torch

from turkeyworld.models import build_upscaler
from turkeyworld.neural_enemy_restoration_sr import(NeuralEnemyRestorationSR,FullFrameEnemyDecoder,
    initialize_neural_enemy_restoration,frozen_background_state,neural_restoration_metadata)
from turkeyworld.renderer_student import _hash_tensors
from turkeyworld.sr_transformer_moe import CONFIG


def parent_payload():
    model=NeuralEnemyRestorationSR(channels=8,blocks=1,semantic_channels=8,semantic_blocks=1,
                                    restoration_channels=8)
    values={name:value.detach().clone()for name,value in frozen_background_state(model).items()}
    values.update(atlas_frames_rgba=torch.arange(28,dtype=torch.uint8).reshape(7,4,1,1),
                  atlas_native_sizes=torch.ones(7,2,dtype=torch.int64))
    config={key:value for key,value in model.config.items()if key not in
        ('restoration_channels','restoration_bottleneck_blocks','frozen_sr_moe')}
    config.update(architecture='atlas_enemy_edsr',**deepcopy(CONFIG))
    return dict(config=config,model=values,training_steps=1234,sr_moe_binding=dict(
        state_sha256=_hash_tensors({name[len('sr_moe.'):]:value for name,value in values.items()if name.startswith('sr_moe.')}),
        parent_model_sha256=_hash_tensors({name:value for name,value in values.items()if not name.startswith('sr_moe.')})))


class NeuralEnemyRestorationModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_parent_copy_never_constructs_or_reads_atlas_and_copies_frozen_tensors_bitexact(self):
        parent=parent_payload();before={key:value.clone()for key,value in parent['model'].items()}
        rng=torch.get_rng_state().clone()
        with patch('builtins.open',side_effect=AssertionError('No asset file reads')):
            model,report=initialize_neural_enemy_restoration(parent,restoration_channels=8)
            output=model.forward_with_aux(torch.rand(1,3,32,32),hypothesis=torch.full((1,21),float('nan')))
        # The only RNG use outside the preservation block is the explicit input above.
        torch.set_rng_state(rng)
        self.assertTrue(report['frozen_background_bitexact']);self.assertFalse(report['atlas_class_used'])
        self.assertFalse(report['atlas_buffers_copied']);self.assertFalse(report['old_Adam_reused'])
        self.assertEqual(report['parent_training_steps'],1234)
        self.assertTrue(all(torch.equal(value,before[name])for name,value in parent['model'].items()))
        self.assertTrue(all(torch.equal(value,parent['model'][name])for name,value in frozen_background_state(model).items()))
        self.assertTrue(all('atlas'not in name.lower()for name in model.state_dict()))
        self.assertEqual(output['image'].shape,(1,3,128,128));self.assertFalse(output['background'].requires_grad)
        self.assertFalse(output['hypothesis_used']);self.assertTrue(output['inference_rgb_only'])

    def test_initializer_preserves_rng_and_is_reproducible_with_explicit_new_decoder_seed(self):
        parent=parent_payload();state=torch.get_rng_state().clone()
        first,_=initialize_neural_enemy_restoration(parent,seed=175,restoration_channels=8)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        second,_=initialize_neural_enemy_restoration(parent,seed=175,restoration_channels=8)
        self.assertTrue(all(torch.equal(a,b)for a,b in zip(first.state_dict().values(),second.state_dict().values())))
        third,_=initialize_neural_enemy_restoration(parent,seed=176,restoration_channels=8)
        self.assertFalse(torch.equal(first.enemy_restoration.rgb.weight,third.enemy_restoration.rgb.weight))

    def test_only_new_fullframe_decoder_trains_and_background_is_identical_after_adam(self):
        model,_=initialize_neural_enemy_restoration(parent_payload(),restoration_channels=8)
        before={name:value.clone()for name,value in frozen_background_state(model).items()}
        self.assertTrue(all(not module.training for name,module in model.named_children()if name!='enemy_restoration'))
        optimizer=torch.optim.Adam([p for p in model.parameters()if p.requires_grad],lr=1e-3)
        output=model.forward_with_aux(torch.rand(2,3,32,32));loss=(output['image']-.25).abs().mean()
        loss=loss+(output['enemy_rgb']-.1).square().mean()+output['alpha_logits'].square().mean()
        loss.backward();optimizer.step()
        self.assertTrue(all(parameter.grad is None for name,parameter in model.named_parameters()if not name.startswith('enemy_restoration.')))
        self.assertGreater(float(model.enemy_restoration.rgb.weight.grad.abs().sum()),0.)
        self.assertGreater(float(model.enemy_restoration.opacity.weight.grad.abs().sum()),0.)
        self.assertTrue(all(torch.equal(model.state_dict()[name],value)for name,value in before.items()))
        metadata=neural_restoration_metadata(model);self.assertFalse(metadata['atlas_asset_required'])
        self.assertTrue(metadata['direct_RGB_and_alpha_supervision'])

    def test_standalone_config_state_roundtrip_and_hypothesis_independence(self):
        model,_=initialize_neural_enemy_restoration(parent_payload(),restoration_channels=8);model.eval()
        value=torch.rand(1,3,32,32);expected=model(value)
        torch.testing.assert_close(expected,model(value,hypothesis=torch.randn(1,21)),atol=0,rtol=0)
        stream=io.BytesIO();torch.save(dict(config=model.config,model=model.state_dict()),stream);stream.seek(0)
        payload=torch.load(stream,weights_only=False)
        with patch('builtins.open',side_effect=AssertionError('Standalone inference needs no file assets')):
            restored=build_upscaler(payload['config']);restored.load_state_dict(payload['model'],strict=True);restored.eval()
            torch.testing.assert_close(expected,restored(value),atol=0,rtol=0)

    def test_missing_parent_tensor_changed_binding_wrong_profile_and_invalid_input_are_rejected(self):
        for mutate in(lambda p:p['model'].pop('semantic_input.weight'),
                       lambda p:p['sr_moe_binding'].update(state_sha256='bad'),
                       lambda p:p['config'].update(architecture='semantic_edsr')):
            parent=parent_payload();mutate(parent)
            with self.assertRaises(ValueError):initialize_neural_enemy_restoration(parent,restoration_channels=8)
        model=NeuralEnemyRestorationSR(channels=8,blocks=1,semantic_channels=8,semantic_blocks=1,restoration_channels=8)
        for value in(torch.zeros(1,3,64,64),torch.full((1,3,32,32),float('nan'))):
            with self.assertRaises(ValueError):model(value)
        with self.assertRaises(ValueError):FullFrameEnemyDecoder(channels=7)

    def test_default_decoder_is_a_compact_learned_fullframe_network(self):
        decoder=FullFrameEnemyDecoder();count=sum(parameter.numel()for parameter in decoder.parameters())
        self.assertGreater(count,1_000_000);self.assertLess(count,3_000_000)
        rgb,opacity=decoder(torch.rand(1,3,32,32));self.assertEqual(rgb.shape,(1,3,128,128))
        self.assertEqual(opacity.shape,(1,1,128,128));self.assertTrue(bool(((rgb>=0)&(rgb<=1)).all()))


if __name__=='__main__':unittest.main()
