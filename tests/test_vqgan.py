"""Real VQ code assignments, corrected beta, codec gradients and honest state."""
from dataclasses import asdict
import io
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Optional PyTorch dependency is not installed')
class VQGANTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def config(self, **overrides):
        from turkeyworld.vqgan import VQGANConfig
        values = dict(image_size=32, base_channels=8, latent_channels=16,
                      embedding_dim=4, n_embed=32, kmeans_iterations=2,
                      max_init_samples=128, distance_chunk_size=17)
        values.update(overrides)
        return VQGANConfig(**values)

    def test_default_codec_has_real_compact_parameters_and_fourfold_downsampling(self):
        from turkeyworld.vqgan import VQGAN, VQGANConfig
        config = VQGANConfig()
        with torch.device('meta'):
            model = VQGAN(config)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 4_756_307)
        self.assertEqual((config.downsample_factor, config.latent_size, config.embedding_dim), (4, 32, 16))
        self.assertEqual(sum(parameter.numel() for parameter in model.encoder.parameters()), 2_257_216)
        self.assertEqual(sum(parameter.numel() for parameter in model.decoder.parameters()), 2_478_467)
        self.assertIs(model.get_last_layer(), model.decoder.conv_out.weight)
        self.assertEqual(model.get_config(), asdict(config))

    def test_vq_commitment_beta_scales_encoder_gradient_not_codebook_gradient(self):
        from turkeyworld.vqgan import VectorQuantizer
        quantizer = VectorQuantizer(self.config(embedding_dim=2, n_embed=2, data_init=False,
                                               commitment_beta=.25))
        with torch.no_grad():
            quantizer.embedding.weight.copy_(torch.tensor([[0., 0.], [3., 3.]]))
        features = torch.tensor([[[[1.]], [[0.]]]], requires_grad=True)
        result = quantizer(features)
        self.assertEqual(int(result['indices'][0, 0, 0]), 0)
        self.assertAlmostEqual(float(result['vq_loss'].detach()), .625)
        result['vq_loss'].backward()
        torch.testing.assert_close(features.grad.flatten(), torch.tensor([.25, 0.]), rtol=0, atol=0)
        torch.testing.assert_close(quantizer.embedding.weight.grad,
            torch.tensor([[-1., 0.], [0., 0.]]), rtol=0, atol=0)

    def test_straight_through_quantization_preserves_values_and_reconstruction_gradient(self):
        from turkeyworld.vqgan import VectorQuantizer
        quantizer = VectorQuantizer(self.config(embedding_dim=2, n_embed=2, data_init=False))
        with torch.no_grad():
            quantizer.embedding.weight.copy_(torch.tensor([[0., 0.], [3., 3.]]))
        features = torch.tensor([[[[1., 2.9]], [[0., 3.1]]]], requires_grad=True)
        result = quantizer(features)
        torch.testing.assert_close(result['indices'], torch.tensor([[[0, 1]]]), rtol=0, atol=0)
        expected = torch.tensor([[[[0., 3.]], [[0., 3.]]]])
        torch.testing.assert_close(result['quantized'], expected, rtol=0, atol=1e-6)
        result['quantized'].sum().backward()
        torch.testing.assert_close(features.grad, torch.ones_like(features), rtol=0, atol=0)
        self.assertIsNone(quantizer.embedding.weight.grad)

    def test_near_tie_code_assignment_remains_true_fp32_inside_bfloat16_autocast(self):
        from turkeyworld.vqgan import VectorQuantizer
        quantizer = VectorQuantizer(self.config(embedding_dim=2, n_embed=2, data_init=False)).eval()
        with torch.no_grad():
            quantizer.embedding.weight.copy_(torch.tensor([[1., 1.], [1.004, 1.]]))
        features = torch.tensor([[[[1.003]], [[1.001]]]])
        expected = quantizer(features)
        self.assertEqual(int(expected['indices'][0, 0, 0]), 1)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            actual = quantizer(features)
        torch.testing.assert_close(actual['indices'], expected['indices'], rtol=0, atol=0)
        torch.testing.assert_close(actual['vq_loss'], expected['vq_loss'], rtol=0, atol=0)
        self.assertEqual(actual['vq_loss'].dtype, torch.float32)

    def test_rgb_encode_decode_shapes_range_and_encoder_decoder_codebook_gradients(self):
        from turkeyworld.vqgan import VQGAN
        model = VQGAN(self.config()).train()
        image = torch.rand(2, 3, 32, 32)
        result = model(image)
        self.assertEqual(result['prequant'].shape, (2, 4, 8, 8))
        self.assertEqual(result['quantized'].shape, (2, 4, 8, 8))
        self.assertEqual(result['indices'].shape, (2, 8, 8))
        self.assertEqual(result['reconstruction'].shape, image.shape)
        self.assertTrue(torch.isfinite(result['reconstruction']).all())
        self.assertTrue(((result['reconstruction'] >= 0) & (result['reconstruction'] <= 1)).all())
        objective = (result['reconstruction']-image).abs().mean()+result['vq_loss']
        objective.backward()
        for parameter in (model.encoder.conv_in.weight, model.quant_conv.weight,
                          model.quantize.embedding.weight, model.post_quant_conv.weight, model.get_last_layer()):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)
        with torch.no_grad():
            from_codes = model.decode_codes(result['indices'])
        torch.testing.assert_close(from_codes, result['reconstruction'].detach(), rtol=2e-5, atol=2e-6)

    def test_training_data_initialization_is_once_only_and_eval_does_not_mutate_any_buffer(self):
        from turkeyworld.vqgan import VectorQuantizer
        quantizer = VectorQuantizer(self.config(embedding_dim=2, n_embed=4, max_init_samples=128))
        values = torch.tensor([[[[-2., -1.], [1., 2.]], [[-1., 2.], [-2., 1.]]]])
        quantizer.eval()
        original = {name: value.clone() for name, value in quantizer.state_dict().items()}
        quantizer(values)
        for name, value in quantizer.state_dict().items():
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)
        self.assertFalse(bool(quantizer.initialized))
        quantizer.train()
        result = quantizer(values)
        self.assertTrue(bool(quantizer.initialized))
        self.assertEqual(int(result['metrics']['active_codes']), 4)
        self.assertAlmostEqual(float(result['vq_loss'].detach()), 0., places=6)
        trained_table = quantizer.embedding.weight.detach().clone()
        quantizer(values*20)
        torch.testing.assert_close(trained_table, quantizer.embedding.weight, rtol=0, atol=0)
        self.assertEqual(int(quantizer.usage_updates), 2)

    def test_perplexity_usage_and_dead_code_statistics_reflect_actual_assignments(self):
        from turkeyworld.vqgan import VectorQuantizer
        quantizer = VectorQuantizer(self.config(embedding_dim=1, n_embed=4, data_init=False))
        with torch.no_grad():
            quantizer.embedding.weight.copy_(torch.arange(4)[:, None])
        result = quantizer(torch.arange(4.).reshape(1, 1, 2, 2))
        self.assertAlmostEqual(float(result['metrics']['perplexity']), 4., places=5)
        self.assertEqual(int(result['metrics']['active_codes']), 4)
        self.assertAlmostEqual(float(result['metrics']['codebook_usage']), 1.)
        self.assertEqual(int(result['metrics']['dead_codes']), 0)
        quantizer.eval()
        before = quantizer.usage_ema.clone()
        collapsed = quantizer(torch.zeros(1, 1, 2, 2))
        self.assertAlmostEqual(float(collapsed['metrics']['perplexity']), 1.)
        self.assertEqual(int(collapsed['metrics']['active_codes']), 1)
        torch.testing.assert_close(before, quantizer.usage_ema, rtol=0, atol=0)

    def test_frozen_codec_quantization_and_decoder_still_train_latent_inputs(self):
        from turkeyworld.vqgan import VQGAN
        model = VQGAN(self.config(data_init=False)).eval().requires_grad_(False)
        latents = torch.randn(1, 4, 8, 8, requires_grad=True)
        state_before = {name: value.clone() for name, value in model.state_dict().items()}
        quantized = model.quantize(latents)['quantized']
        reconstruction = model.decode(quantized)
        reconstruction.square().mean().backward()
        self.assertGreater(float(latents.grad.abs().sum()), 0.)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, state_before[name], rtol=0, atol=0)

    def test_checkpoint_restores_codebook_initialization_usage_and_outputs(self):
        from turkeyworld.vqgan import VQGAN
        model = VQGAN(self.config()).train()
        image = torch.rand(1, 3, 32, 32)
        model(image)
        model.eval()
        with torch.no_grad():
            expected = model(image)
        stream = io.BytesIO()
        torch.save({'config': model.get_config(), 'model': model.state_dict()}, stream)
        stream.seek(0)
        saved = torch.load(stream, map_location='cpu', weights_only=True)
        loaded = VQGAN(saved['config']).eval()
        loaded.load_state_dict(saved['model'])
        with torch.no_grad():
            actual = loaded(image)
        self.assertTrue(bool(loaded.quantize.initialized))
        self.assertEqual(int(loaded.quantize.usage_updates), 1)
        torch.testing.assert_close(expected['indices'], actual['indices'], rtol=0, atol=0)
        torch.testing.assert_close(expected['reconstruction'], actual['reconstruction'], rtol=0, atol=0)

    def test_config_and_public_shapes_fail_explicitly(self):
        from turkeyworld.vqgan import VQGAN, VQGANConfig
        for options in ({'image_size': 31}, {'commitment_beta': -1.}, {'embedding_dim': 0},
                        {'channel_multipliers': ()}, {'usage_decay': 1.}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                VQGANConfig(**options)
        model = VQGAN(self.config())
        with self.assertRaises(ValueError):
            model.encode(torch.rand(1, 3, 31, 32))
        with self.assertRaises(ValueError):
            model.decode(torch.rand(1, 4, 7, 8))


if __name__ == '__main__':
    unittest.main()
