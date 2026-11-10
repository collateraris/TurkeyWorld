import copy
import unittest

import torch

from turkeyworld.compact_codec import (COMPACT_CODEC_PARAMETERS, build_compact_codec,
    compact_codec_config, compact_codec_parameter_counts)
from turkeyworld.vqgan import VQGAN, VQGANConfig


class CompactCodecTests(unittest.TestCase):
    @staticmethod
    def source(**changes):
        values = dict(image_size=32, base_channels=8, latent_channels=16,
                      num_res_blocks=1, n_embed=1488, data_init=False)
        values.update(changes)
        return VQGAN(VQGANConfig(**values)).eval()

    def test_exact_real_parameter_budget_and_unchanged_latent_contract(self):
        config = compact_codec_config()
        self.assertEqual((config.image_size, config.downsample_factor,
                          config.latent_size, config.embedding_dim, config.n_embed), (32, 4, 8, 16, 1488))
        with torch.device('meta'):
            codec = VQGAN(config)
        counts = compact_codec_parameter_counts(codec)
        self.assertEqual(counts['total'], COMPACT_CODEC_PARAMETERS)
        self.assertEqual(counts['total'], 997539)
        self.assertEqual((counts['encoder'], counts['decoder'], counts['codebook']), (445328, 528403, 23808))
        self.assertEqual(counts['encoder']+counts['decoder']+counts['codebook'], counts['total'])

    def test_preserves_every_quantizer_tensor_without_alias_or_global_rng_changes(self):
        teacher = self.source()
        before = {key: value.clone() for key, value in teacher.state_dict().items()}
        rng = torch.random.get_rng_state().clone()
        codec = build_compact_codec(teacher, seed=173)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        for key, value in teacher.quantize.state_dict().items():
            self.assertTrue(torch.equal(value, codec.quantize.state_dict()[key]), key)
        self.assertNotEqual(codec.quantize.embedding.weight.data_ptr(), teacher.quantize.embedding.weight.data_ptr())
        for key, value in teacher.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)
        self.assertEqual(compact_codec_parameter_counts(codec)['trainable'], 973731)

    def test_training_updates_encoder_decoder_and_keeps_complete_quantizer_frozen(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            teacher = self.source()
            codec = build_compact_codec(teacher).requires_grad_(True).train()
            quantizer = {key: value.clone() for key, value in codec.quantize.state_dict().items()}
            inputs = torch.linspace(0, 1, 2*3*32*32).reshape(2, 3, 32, 32)
            result = codec(inputs)
            self.assertEqual(result['prequant'].shape, (2, 16, 8, 8))
            self.assertEqual(result['quantized'].shape, (2, 16, 8, 8))
            self.assertEqual(result['reconstruction'].shape, inputs.shape)
            self.assertTrue(torch.isfinite(result['reconstruction']).all())
            self.assertGreaterEqual(float(result['reconstruction'].detach().min()), 0)
            self.assertLessEqual(float(result['reconstruction'].detach().max()), 1)
            loss = (result['reconstruction']-inputs).abs().mean()+result['vq_loss']
            loss.backward()
            self.assertGreater(float(codec.encoder.conv_in.weight.grad.abs().sum()), 0)
            self.assertGreater(float(codec.decoder.conv_out.weight.grad.abs().sum()), 0)
            self.assertIsNone(codec.quantize.embedding.weight.grad)
            self.assertFalse(codec.quantize.training)
            self.assertTrue(codec.encoder.training)
            self.assertTrue(codec.decoder.training)
            torch.optim.AdamW([value for value in codec.parameters() if value.requires_grad], lr=1e-4).step()
            for key, value in codec.quantize.state_dict().items():
                self.assertTrue(torch.equal(value, quantizer[key]), key)
        finally:
            torch.set_num_threads(previous_threads)

    def test_decoder_passes_flow_latent_gradients_through_frozen_codec(self):
        codec = build_compact_codec(self.source()).eval().requires_grad_(False)
        latents = torch.randn(1, 16, 8, 8, requires_grad=True)
        codec.decode(codec.quantize(latents)['quantized']).square().mean().backward()
        self.assertGreater(float(latents.grad.abs().sum()), 0)
        self.assertTrue(all(value.grad is None for value in codec.parameters()))

    def test_rejects_incompatible_or_uninitialized_teacher_interfaces(self):
        for changes in ({'image_size': 64}, {'embedding_dim': 8}, {'n_embed': 1024},
                        {'channel_multipliers': (1, 2)}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                compact_codec_config(self.source(**changes).config)
        with self.assertRaises(ValueError):
            build_compact_codec(self.source(data_init=True))
        teacher = self.source()
        settings = copy.deepcopy(teacher.get_config())
        settings.update(commitment_beta=.75, usage_decay=.95, distance_chunk_size=512)
        config = compact_codec_config(settings)
        self.assertEqual((config.commitment_beta, config.usage_decay, config.distance_chunk_size), (.75, .95, 512))


if __name__ == '__main__':
    unittest.main()
