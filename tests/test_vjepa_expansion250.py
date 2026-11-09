"""Function-preserving250M capacity growth and active new transformer branches."""
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Optional PyTorch dependency is not installed')
class VJEPA250ExpansionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def source(self):
        from turkeyworld.models import NeuralWorldModel, WorldModelConfig
        return NeuralWorldModel(WorldModelConfig(kind='teacher', image_size=128, architecture='vjepa2_ac',
            endless=True, renderer_mode='flow_matching', flow_steps=1, flow_width=32, flow_depth=1,
            flow_heads=4, vjepa_width=32, vjepa_encoder_depth=1, vjepa_predictor_depth=1,
            vjepa_heads=4, vjepa_context_frames=2)).eval()

    def options(self):
        return dict(require_profile=False, vjepa_width=32, vjepa_encoder_depth=3,
                    vjepa_predictor_depth=3, vjepa_heads=4)

    def save_source(self, model, path):
        torch.save({'stage': 'world', 'config': asdict(model.config), 'model': model.state_dict(),
            'training_steps': 1, 'global_training_step': 1967,
            'optimizer': {'old': True}, 'ema_target_encoder': {'old': True},
            'teacher_quality_gate': {'passed': False}}, path)

    def test_full_generator_is_249956805_used_parameters_and_memory_stays_compatible(self):
        from turkeyworld.models import NeuralWorldModel, WorldModelConfig, parameter_counts
        from turkeyworld.vjepa2_ac import VJEPA250_PROFILE
        with torch.device('meta'):
            model = NeuralWorldModel(WorldModelConfig(kind='teacher', image_size=128, architecture='vjepa2_ac',
                endless=True, renderer_mode='flow_matching', flow_steps=12,
                flow_adapter_width=512, flow_adapter_depth=12, flow_adapter_heads=8, **VJEPA250_PROFILE))
        self.assertEqual(parameter_counts(model)['world_total'], 249_956_805)
        self.assertEqual(len(model.core.encoder.blocks), 32)
        self.assertEqual(len(model.core.predictor.blocks), 23)
        self.assertEqual((model.core.memory_tokens, model.core.width), (264, 512))
        self.assertTrue(model.core.encoder.gradient_checkpointing)
        self.assertTrue(model.core.predictor.gradient_checkpointing)
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        self.assertFalse(any('ema' in name or 'discriminator' in name for name, _ in model.named_parameters()))

    def test_expansion_retains_all_old_weights_and_initial_visual_action_functions(self):
        from turkeyworld.vjepa2_ac import load_vjepa250_checkpoint
        original = self.source()
        with torch.no_grad():
            original.vjepa_backend.feature_head.weight.normal_(std=.01)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'source.pt'
            self.save_source(original, source)
            expanded, payload = load_vjepa250_checkpoint(source, **self.options())
            report = payload['vjepa250_expansion']
            self.assertTrue(report['all_original_weight_values_exact'])
            self.assertEqual(report['widened_tensor_count'], 3)
            self.assertEqual(report['encoder_blocks_added'], 2)
            self.assertEqual(report['predictor_blocks_added'], 2)
            self.assertGreater(report['new_parameters'], 0)
            for name, value in original.state_dict().items():
                copied = expanded.state_dict()[name]
                slices = tuple(slice(0, length) for length in value.shape)
                torch.testing.assert_close(copied[slices], value, rtol=0, atol=0)
            video = torch.rand(1, 3, 3, 128, 128)
            actions, states = torch.randn(1, 3, 4), torch.randn(1, 3, 90)
            with torch.inference_mode():
                old_latents, new_latents = original.core.encode_video(video), expanded.core.encode_video(video)
                torch.testing.assert_close(old_latents, new_latents, rtol=0, atol=0)
                old_prediction = original.core.predict(old_latents, actions, states)
                new_prediction = expanded.core.predict(new_latents, actions, states)
                torch.testing.assert_close(old_prediction, new_prediction, rtol=0, atol=0)
                pooled = old_prediction[:, -1].mean(1)
                old_readout = original.vjepa_backend.readout_neck(pooled)
                new_readout = expanded.vjepa_backend.readout_neck(pooled)
                torch.testing.assert_close(old_readout, new_readout, rtol=2e-6, atol=2e-7)
                changed_actions = actions.clone()
                changed_actions[:, 1, 0] += 1.
                changed = expanded.core.predict(new_latents, changed_actions, states)
                self.assertGreater(float((changed[:, 1]-new_prediction[:, 1]).abs().sum()), 0.)
                torch.testing.assert_close(changed[:, 0], new_prediction[:, 0], rtol=0, atol=0)
                future = video.clone()
                future[:, 2] += 3.
                future_latents = expanded.core.encode_video(future)
                torch.testing.assert_close(new_latents[:, :2], future_latents[:, :2], rtol=0, atol=0)
            self.assertEqual(payload['training_steps'], 1)
            self.assertEqual(payload['global_training_step'], 1967)
            self.assertEqual(payload['expansion_training_steps'], 0)
            for key in ('optimizer', 'ema_target_encoder', 'teacher_quality_gate'):
                self.assertNotIn(key, payload)

    def test_identity_branches_receive_gradients_then_learn_internal_attention(self):
        from turkeyworld.vjepa2_ac import load_vjepa250_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'source.pt'
            self.save_source(self.source(), source)
            model, _ = load_vjepa250_checkpoint(source, **self.options())
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=.05)
        video = torch.rand(1, 2, 3, 128, 128)
        actions, states = torch.randn(1, 2, 4), torch.randn(1, 2, 90)
        for iteration in range(2):
            optimizer.zero_grad(set_to_none=True)
            latents = model.core.encode_video(video)
            prediction = model.core.predict(latents, actions, states)
            target = torch.linspace(-1., 1., 32)[None, None, None]
            loss = (prediction-target).square().mean()+.1*(latents-target).square().mean()
            neck = model.vjepa_backend.readout_neck
            readout = neck(prediction[:, -1].mean(1))
            loss = loss+.1*(readout-target.flatten()).square().mean()
            loss.backward()
            self.assertGreater(float(neck[3].weight.grad[:, 128:].abs().sum()), 0.)
            if iteration:
                self.assertGreater(float(neck[1].weight.grad[128:].abs().sum()), 0.)
            for component in (model.core.encoder, model.core.predictor):
                for block in component.blocks[1:]:
                    self.assertGreater(float(block.attention.proj.weight.grad.abs().sum()), 0.)
                    self.assertGreater(float(block.mlp[-1].weight.grad.abs().sum()), 0.)
                    if iteration:
                        self.assertGreater(float(block.attention.qkv.weight.grad.abs().sum()), 0.)
                        self.assertGreater(float(block.mlp[0].weight.grad.abs().sum()), 0.)
            optimizer.step()

    def test_checkpointing_matches_ordinary_training_outputs_and_gradients(self):
        from turkeyworld.vjepa2_ac import CompactVJEPA2AC
        first = CompactVJEPA2AC(width=32, encoder_depth=2, predictor_depth=2, heads=4, context_frames=2)
        second = CompactVJEPA2AC(width=32, encoder_depth=2, predictor_depth=2, heads=4, context_frames=2,
                                gradient_checkpointing=True)
        second.load_state_dict(first.state_dict())
        video, actions, states = torch.rand(1, 2, 3, 128, 128), torch.randn(1, 2, 4), torch.randn(1, 2, 90)
        outputs = []
        for model in (first, second):
            result = model.predict(model.encode_video(video), actions, states)
            result.square().mean().backward()
            outputs.append(result.detach())
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        for (_, old), (_, new) in zip(first.named_parameters(), second.named_parameters()):
            torch.testing.assert_close(old.grad, new.grad, rtol=0, atol=0)

    def test_migration_creates_separate_loadable_checkpoint_and_refuses_overwrite(self):
        from turkeyworld.models import load_world_checkpoint
        from turkeyworld.vjepa2_ac import migrate_vjepa250_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory)/'source.pt', Path(directory)/'expanded.pt'
            self.save_source(self.source(), source)
            report = migrate_vjepa250_checkpoint(source, target, **self.options())
            model, payload = load_world_checkpoint(target)
            self.assertEqual(payload['pipeline_phase'], 'vjepa2_ac_250m_initial')
            self.assertEqual(report['generator_parameters'], sum(parameter.numel() for parameter in model.parameters()))
            self.assertTrue(source.exists())
            with self.assertRaises(FileExistsError):
                migrate_vjepa250_checkpoint(source, target, **self.options())
            with self.assertRaises(FileExistsError):
                migrate_vjepa250_checkpoint(source, source, **self.options())


if __name__ == '__main__':
    unittest.main()
