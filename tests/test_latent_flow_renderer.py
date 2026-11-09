"""Actual latent ODEs, shared Gaussian reflow noise and frozen-codec gradients."""
from dataclasses import asdict
import io
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Optional PyTorch dependency is not installed')
class LatentFlowRendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def codec_config(self):
        return dict(image_size=32, base_channels=8, latent_channels=16, embedding_dim=4,
                    n_embed=32, data_init=False, kmeans_iterations=0)

    def renderer(self, kind='student', load=True, **options):
        from turkeyworld.latent_flow_renderer import LatentFlowRenderer
        from turkeyworld.vqgan import VQGAN
        values = dict(kind=kind, image_size=32, mode='rectified_flow' if kind == 'student' else 'flow_matching',
                      width=32, depth=1, heads=4, codec_config=self.codec_config())
        values.update(options)
        model = LatentFlowRenderer(**values)
        if load:
            source = VQGAN(self.codec_config())
            model.load_codec({'config': source.get_config(), 'model': source.state_dict()})
        return model

    def inputs(self):
        state = torch.zeros(1, 90)
        state[:, :8] = torch.tensor([.2, .3, 0., 1., 1., 1., 0., 0.])
        return torch.rand(1, 3, 32, 32), state

    def test_default_latent_geometry_uses_64_tokens_and_includes_frozen_codec_parameters(self):
        from turkeyworld.latent_flow_renderer import LatentFlowRenderer
        with torch.device('meta'):
            model = LatentFlowRenderer(kind='student', mode='rectified_flow')
        self.assertEqual((model.latent_channels, model.latent_size, model.patch, model.grid), (16, 32, 4, 8))
        self.assertEqual(model.position.shape, (1, 64, 128))
        self.assertEqual(model.fixed_noise.shape, (1, 3, 128, 128))
        self.assertEqual(sum(parameter.numel() for parameter in model.codec.parameters()), 4_756_307)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.codec.parameters()))
        self.assertFalse(model.codec.training)

    def test_explicit_hidden_width_preserves_old_defaults_and_uses_selected_new_capacity(self):
        from turkeyworld.flow_renderer import FlowBlock
        torch.manual_seed(722)
        original = FlowBlock(32, 4)
        torch.manual_seed(722)
        explicit_default = FlowBlock(32, 4, mlp_hidden=0)
        for name, value in original.state_dict().items():
            torch.testing.assert_close(value, explicit_default.state_dict()[name], rtol=0, atol=0)
        compact = FlowBlock(32, 4, mlp_hidden=96)
        self.assertEqual(compact.mlp[0].out_features, 96)
        self.assertEqual(compact.mlp[-1].in_features, 96)
        with self.assertRaises(ValueError):
            FlowBlock(32, 4, mlp_hidden=-1)

    def test_full_teacher_and_student_budgets_include_the_entire_1488_code_codec(self):
        from turkeyworld.models import NeuralWorldModel, WorldModelConfig
        from turkeyworld.latent_flow_renderer import LatentFlowRenderer
        from turkeyworld.vjepa2_ac import VJEPA500_PROFILE
        with torch.device('meta'):
            student = NeuralWorldModel(WorldModelConfig(kind='student', architecture='vjepa2_ac',
                image_size=128, endless=True, renderer_mode='rectified_flow', flow_steps=4,
                vjepa_width=320, vjepa_encoder_depth=4, vjepa_predictor_depth=5, vjepa_heads=5,
                vjepa_readout_mlp_ratio=5, flow_width=192, flow_depth=4, flow_heads=4,
                flow_mlp_hidden=736, renderer_codec='vqgan', renderer_codec_config={'n_embed': 1488}))
            teacher = NeuralWorldModel(WorldModelConfig(kind='teacher', architecture='vjepa2_ac',
                image_size=128, endless=True, renderer_mode='flow_matching', flow_steps=12,
                flow_adapter_width=512, flow_adapter_depth=12, flow_adapter_heads=8,
                renderer_codec='vqgan', renderer_codec_config={'n_embed': 1488}, **VJEPA500_PROFILE))
        self.assertEqual(sum(parameter.numel() for parameter in student.parameters()), 22_475_096)
        self.assertEqual(sum(parameter.numel() for parameter in teacher.parameters()), 504_857_752)
        for model in (student, teacher):
            self.assertIsInstance(model.flow_renderer, LatentFlowRenderer)
            codec = model.flow_renderer.codec
            self.assertEqual(sum(parameter.numel() for parameter in codec.parameters()), 4_763_731)
            self.assertTrue(all(not parameter.requires_grad for parameter in codec.parameters()))
        self.assertEqual(student.flow_renderer.blocks[0].mlp[0].out_features, 736)
        self.assertEqual((len(student.core.encoder.blocks), len(student.core.predictor.blocks)), (4, 5))

    def test_unloaded_codec_is_explicitly_rejected_and_training_keeps_it_frozen_eval(self):
        model = self.renderer(load=False)
        base, state = self.inputs()
        with self.assertRaisesRegex(RuntimeError, 'Load the trained VQGAN'):
            model.sample(base, state)
        with self.assertRaisesRegex(RuntimeError, 'Load the trained VQGAN'):
            model.loss(base, base, state)
        model = self.renderer()
        model.train().requires_grad_(True)
        self.assertFalse(model.codec.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.codec.parameters()))
        self.assertTrue(any(parameter.requires_grad for name, parameter in model.named_parameters()
                            if not name.startswith('codec.')))

    def test_rgb_noise_polyphase_projection_is_exact_and_accepts_identical_latent_noise(self):
        model = self.renderer()
        rgb_noise = torch.arange(3*32*32).float().reshape(1, 3, 32, 32)
        expected = torch.nn.functional.pixel_unshuffle(rgb_noise, 4)[:, :4]
        projected = model._noise(rgb_noise, 1, torch.device('cpu'))
        torch.testing.assert_close(projected, expected, rtol=0, atol=0)
        torch.testing.assert_close(model._noise(expected, 1, torch.device('cpu')), expected, rtol=0, atol=0)
        self.assertEqual(model._noise(rgb_noise, 3, torch.device('cpu')).shape, (3, 4, 8, 8))
        base, state = self.inputs()
        with torch.no_grad():
            first = model.latent_sample(base, state, noise=rgb_noise)
            second = model.latent_sample(base, state, noise=expected)
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_teacher_heun_and_student_euler_cover_full_interval_without_mutating_noise(self):
        base, state = self.inputs()
        initial = torch.full((1, 4, 8, 8), -.3)
        constant = torch.full_like(initial, .4)
        for kind, calls in (('teacher', 8), ('student', 4)):
            with self.subTest(kind=kind):
                model = self.renderer(kind)
                with patch.object(model, 'velocity', return_value=constant) as velocity:
                    latent = model.latent_sample(base, state, noise=initial, steps=4)
                self.assertEqual(velocity.call_count, calls)
                torch.testing.assert_close(latent, torch.full_like(initial, .1), rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(initial, torch.full_like(initial, -.3), rtol=0, atol=0)
        teacher = self.renderer('teacher')

        def linear_time(noisy, time, condition, labels, self_condition=None):
            return time[:, None, None, None].expand_as(noisy)

        with patch.object(teacher, 'velocity', side_effect=linear_time) as velocity:
            latent = teacher.latent_sample(base, state, noise=torch.zeros_like(initial))
        self.assertEqual(velocity.call_count, 24)
        torch.testing.assert_close(latent, torch.full_like(initial, .5), rtol=1e-5, atol=1e-6)

    def test_terminal_quantization_decode_and_fixed_noise_are_deterministic_and_stateless(self):
        model = self.renderer().eval()
        base, state = self.inputs()
        saved = {name: value.clone() for name, value in model.codec.state_dict().items()}
        with torch.no_grad():
            model.output.weight.normal_(std=.01)
            first = model.sample(base, state)
            torch.manual_seed(912)
            second = model.sample(base, state)
            latent = model.latent_sample(base, state)
            expected = model.codec.decode(model.codec.quantize(latent)['quantized'])
        self.assertEqual(first.shape, (1, 3, 32, 32))
        self.assertTrue(torch.isfinite(first).all())
        self.assertTrue(((first >= 0) & (first <= 1)).all())
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(first, expected, rtol=0, atol=0)
        for name, value in model.codec.state_dict().items():
            torch.testing.assert_close(value, saved[name], rtol=0, atol=0)

    def test_velocity_loss_and_sampled_rgb_backpropagate_without_training_codec_or_source(self):
        model = self.renderer().train()
        base, state = self.inputs()
        base.requires_grad_(True)
        target = torch.rand_like(base, requires_grad=True)
        with torch.no_grad():
            model.output.weight.normal_(std=.01)
        loss, metrics = model.loss(target, base, state, time=torch.tensor([.4]))
        self.assertIn('flow_velocity', metrics)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNone(target.grad)
        self.assertGreater(float(base.grad.abs().sum()), 0.)
        self.assertGreater(float(model.output.weight.grad.abs().sum()), 0.)
        self.assertTrue(all(parameter.grad is None for parameter in model.codec.parameters()))
        model.zero_grad(set_to_none=True)
        generated = model.sample(base.detach(), state)
        (generated-target.detach()).abs().mean().backward()
        self.assertGreater(float(model.output.weight.grad.abs().sum()), 0.)
        self.assertTrue(all(parameter.grad is None for parameter in model.codec.parameters()))

    def test_reflow_encodes_teacher_endpoint_and_uses_the_same_supplied_rgb_noise(self):
        model = self.renderer().eval()
        base, state = self.inputs()
        unrelated_source = torch.rand_like(base)
        endpoint = torch.rand_like(base)
        noise = torch.randn_like(base)
        time = torch.tensor([.25])
        with torch.no_grad():
            latent_target = model.codec.encode(endpoint)['quantized']
            projected_noise = model._noise(noise, 1, base.device)
        expected_velocity = latent_target-projected_noise
        observed = []

        def velocity(noisy, supplied_time, condition, labels, self_condition=None):
            observed.append(noisy.clone())
            return expected_velocity

        with patch.object(model, 'velocity', side_effect=velocity), \
             patch.object(model.codec, 'encode', wraps=model.codec.encode) as encode:
            loss, _ = model.loss(unrelated_source, base, state, noise=noise,
                                 time=time, target_endpoint=endpoint)
        self.assertLess(float(loss), 1e-7)
        self.assertIs(encode.call_args_list[0].args[0], endpoint)
        self.assertEqual(encode.call_count, 2)
        torch.testing.assert_close(observed[0], .75*projected_noise+.25*latent_target, rtol=0, atol=0)

    def test_visible_state_filter_and_zero_adapter_preserve_conditioning_contract(self):
        model = self.renderer(adapter_width=32, adapter_depth=1, adapter_heads=4).eval()
        base, state = self.inputs()
        with torch.no_grad():
            model.output.weight.normal_(std=.01)
            latent = model.codec.encode(base)['quantized']
            noise = torch.randn_like(latent)
            time = torch.tensor([.3])
            first = model.velocity(noise, time, latent, state)
            altered = state.clone()
            altered[:, 4:7] = torch.tensor([[.5, .2, .6]])
            altered[:, -2:] = 7.
            second = model.velocity(noise, time, latent, altered)
            adapter = model.residual_adapter(noise, time, latent, state)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(adapter, torch.zeros_like(adapter), rtol=0, atol=0)

    def test_saved_renderer_restores_codec_loaded_flag_and_all_inference_weights(self):
        model = self.renderer().eval()
        base, state = self.inputs()
        with torch.no_grad():
            expected = model.sample(base, state)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        loaded = self.renderer(load=False).eval()
        loaded.load_state_dict(torch.load(buffer, weights_only=True))
        self.assertTrue(bool(loaded.codec_loaded))
        with torch.no_grad():
            actual = loaded.sample(base, state)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_codec_normalization_scales_training_conditions_targets_and_sample_endpoints(self):
        model = self.renderer().eval()
        payload = {'config': model.codec.get_config(), 'model': model.codec.state_dict(),
                   'latent_normalization': {'mean': [.1, -.2, .3, .4], 'std': [.5, 1., 2., 3.]}}
        parameters = sum(parameter.numel() for parameter in model.parameters())
        model.load_codec(payload)
        base, state = self.inputs()
        target = torch.rand_like(base)
        noise = torch.randn(1, 4, 8, 8)
        with torch.no_grad():
            expected_target = model.normalize_latents(model.codec.encode(target)['quantized'])
            expected_base = model.normalize_latents(model.codec.encode(base)['quantized'])
        expected_velocity = expected_target-noise
        conditions = []

        def velocity(noisy, time, condition, labels, self_condition=None):
            conditions.append(condition)
            return expected_velocity

        with patch.object(model, 'velocity', side_effect=velocity):
            loss, _ = model.loss(target, base, state, noise=noise, time=torch.tensor([.25]))
        self.assertLess(float(loss), 1e-7)
        torch.testing.assert_close(conditions[0], expected_base, rtol=0, atol=0)
        with patch.object(model, 'velocity', return_value=torch.zeros_like(noise)), torch.no_grad():
            continuous = model.latent_sample(base, state, noise=torch.zeros_like(noise))
            sample = model.sample(base, state, noise=torch.zeros_like(noise))
            raw = model.latent_mean.expand_as(noise)
            expected = model.codec.decode(model.codec.quantize(raw)['quantized'])
        torch.testing.assert_close(continuous, torch.zeros_like(noise), rtol=0, atol=0)
        torch.testing.assert_close(sample, expected, rtol=0, atol=0)
        torch.testing.assert_close(model.denormalize_latents(model.normalize_latents(noise)),
                                   noise, rtol=2e-6, atol=2e-7)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), parameters)

    def test_normalization_persists_in_world_state_and_missing_metadata_resets_identity(self):
        model = self.renderer().eval()
        payload = {'config': model.codec.get_config(), 'model': model.codec.state_dict(),
                   'latent_normalization': {'mean': [1., 2., 3., 4.], 'std': [2., 3., 4., 5.]}}
        model.load_codec(payload)
        state = {name: value.clone() for name, value in model.state_dict().items()}
        loaded = self.renderer(load=False).eval()
        loaded.load_state_dict(state)
        torch.testing.assert_close(loaded.latent_mean, model.latent_mean, rtol=0, atol=0)
        torch.testing.assert_close(loaded.latent_std, model.latent_std, rtol=0, atol=0)
        self.assertTrue(bool(loaded.codec_loaded))
        old_unscaled = {name: value for name, value in state.items() if name not in {'latent_mean', 'latent_std'}}
        loaded.load_state_dict(old_unscaled)
        torch.testing.assert_close(loaded.latent_mean, torch.zeros_like(loaded.latent_mean), rtol=0, atol=0)
        torch.testing.assert_close(loaded.latent_std, torch.ones_like(loaded.latent_std), rtol=0, atol=0)
        model.load_codec({'config': model.codec.get_config(), 'model': model.codec.state_dict()})
        torch.testing.assert_close(model.latent_mean, torch.zeros_like(model.latent_mean), rtol=0, atol=0)
        torch.testing.assert_close(model.latent_std, torch.ones_like(model.latent_std), rtol=0, atol=0)

    def test_invalid_normalization_metadata_is_rejected_before_changing_codec(self):
        model = self.renderer().eval()
        before = {name: value.clone() for name, value in model.state_dict().items()}
        for normalization in ({'mean': [0.]*4, 'std': [0.]*4}, {'mean': [0.], 'std': [1.]*4},
                              {'mean': [float('nan')]*4, 'std': [1.]*4}, {'mean': [0.]*4}):
            with self.subTest(normalization=normalization), self.assertRaises(ValueError):
                model.load_codec({'config': model.codec.get_config(), 'model': model.codec.state_dict(),
                                  'latent_normalization': normalization})
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
