"""Retain trained 250M values while adding active, causal 500M capacity."""

from turkeyworld.engine import STATE_DIM, GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Optional PyTorch dependency is not installed')
class VJEPA500ExpansionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def source(self):
        from turkeyworld.models import NeuralWorldModel, WorldModelConfig
        return NeuralWorldModel(WorldModelConfig(kind='teacher', image_size=128, architecture='vjepa2_ac',
            endless=True, renderer_mode='flow_matching', flow_steps=1, flow_width=32, flow_depth=1,
            flow_heads=4, vjepa_width=32, vjepa_encoder_depth=2, vjepa_predictor_depth=2,
            vjepa_heads=4, vjepa_context_frames=2, vjepa_readout_mlp_ratio=7,
            vjepa_gradient_checkpointing=True)).eval()

    def options(self):
        return dict(require_profile=False, vjepa_width=32, vjepa_encoder_depth=4,
                    vjepa_predictor_depth=4, vjepa_heads=4)

    def save_source(self, model, path):
        torch.save({'stage': 'world', 'config': asdict(model.config), 'model': model.state_dict(),
            'training_steps': 145, 'global_training_step': 145,
            'optimizer': {'old': True}, 'ema_target_encoder': {'old': True},
            'teacher_quality_gate': {'passed': False},
            'vjepa250_expansion': {'all_original_weight_values_exact': True}}, path)

    def test_full_profile_has_499012073_generator_parameters_and_separate_ema(self):
        from turkeyworld.models import NeuralWorldModel, WorldModelConfig, parameter_counts
        from turkeyworld.vjepa2_ac import VJEPA500_PROFILE
        with torch.device('meta'):
            model = NeuralWorldModel(WorldModelConfig(kind='teacher', image_size=128, architecture='vjepa2_ac',
                endless=True, renderer_mode='flow_matching', flow_steps=12,
                flow_adapter_width=512, flow_adapter_depth=12, flow_adapter_heads=8, **VJEPA500_PROFILE))
        self.assertEqual(parameter_counts(model)['world_total'], 499_012_073)
        self.assertEqual(sum(parameter.numel() for parameter in model.core.encoder.parameters()), 239_975_936)
        self.assertEqual((len(model.core.encoder.blocks), len(model.core.predictor.blocks)), (76, 58))
        self.assertEqual((model.core.memory_tokens, model.core.width), (264, 512))
        self.assertTrue(model.core.encoder.gradient_checkpointing)
        self.assertTrue(model.core.predictor.gradient_checkpointing)
        self.assertFalse(any('ema' in name or 'discriminator' in name for name, _ in model.named_parameters()))

    def test_trained_values_and_initial_causal_action_predictions_survive_growth(self):
        from turkeyworld.vjepa2_ac import load_vjepa500_checkpoint
        original = self.source()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'source250.pt'
            self.save_source(original, source)
            model, payload = load_vjepa500_checkpoint(source, **self.options())
        report = payload['vjepa500_expansion']
        self.assertTrue(report['all_original_weight_values_exact'])
        self.assertTrue(report['function_preserving_initialization'])
        self.assertEqual(report['encoder_blocks_added'], 2)
        self.assertEqual(report['predictor_blocks_added'], 2)
        self.assertEqual(report['readout_mlp_ratio'], 9)
        for name, value in original.state_dict().items():
            prefix = tuple(slice(0, size) for size in value.shape)
            torch.testing.assert_close(value, model.state_dict()[name][prefix], rtol=0, atol=0)
        video, actions, states = torch.rand(1, 3, 3, 128, 128), torch.randn(1, 3, 4), torch.randn(1, 3, STATE_DIM)
        with torch.inference_mode():
            latent = original.core.encode_video(video)
            grown_latent = model.core.encode_video(video)
            torch.testing.assert_close(latent, grown_latent, rtol=0, atol=0)
            previous = original.core.predict(latent, actions, states)
            prediction = model.core.predict(grown_latent, actions, states)
            self.assertEqual(prediction.shape, (1, 3, 64, 32))
            torch.testing.assert_close(previous, prediction, rtol=0, atol=0)
            pooled = prediction[:, -1].mean(1)
            torch.testing.assert_close(original.vjepa_backend.readout_neck(pooled),
                model.vjepa_backend.readout_neck(pooled), rtol=2e-6, atol=2e-7)
            changed_actions = actions.clone()
            changed_actions[:, 1, 0] += 2.
            changed = model.core.predict(grown_latent, changed_actions, states)
            torch.testing.assert_close(prediction[:, 0], changed[:, 0], rtol=0, atol=0)
            self.assertGreater(float((prediction[:, 1]-changed[:, 1]).abs().sum()), 0.)
            future_video = video.clone()
            future_video[:, 2] += 3.
            torch.testing.assert_close(grown_latent[:, :2], model.core.encode_video(future_video)[:, :2], rtol=0, atol=0)
        self.assertEqual(payload['training_steps'], 145)
        self.assertEqual(payload['global_training_step'], 145)
        self.assertEqual(payload['expansion_training_steps'], 0)
        self.assertEqual(payload['pipeline_phase'], 'vjepa2_ac_500m_initial')
        self.assertTrue(payload['ema_reset_required'])
        self.assertIn('vjepa250_expansion', payload)
        for key in ('optimizer', 'ema_target_encoder', 'teacher_quality_gate'):
            self.assertNotIn(key, payload)

    def test_added_blocks_and_readout_channels_receive_learning_gradients(self):
        from turkeyworld.vjepa2_ac import load_vjepa500_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'source.pt'
            self.save_source(self.source(), source)
            model, _ = load_vjepa500_checkpoint(source, **self.options())
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=.05)
        video, actions, states = torch.rand(1, 2, 3, 128, 128), torch.randn(1, 2, 4), torch.randn(1, 2, STATE_DIM)
        for iteration in range(2):
            optimizer.zero_grad(set_to_none=True)
            latents = model.core.encode_video(video)
            prediction = model.core.predict(latents, actions, states)
            target = torch.linspace(-1., 1., 32)[None, None, None]
            neck = model.vjepa_backend.readout_neck
            readout = neck(prediction[:, -1].mean(1))
            loss = (prediction-target).square().mean()+.1*(latents-target).square().mean()
            loss = loss+.1*(readout-target.flatten()).square().mean()
            loss.backward()
            self.assertGreater(float(neck[3].weight.grad[:, 224:].abs().sum()), 0.)
            for component in (model.core.encoder, model.core.predictor):
                for block in component.blocks[2:]:
                    self.assertGreater(float(block.attention.proj.weight.grad.abs().sum()), 0.)
                    self.assertGreater(float(block.mlp[-1].weight.grad.abs().sum()), 0.)
                    if iteration:
                        self.assertGreater(float(block.attention.qkv.weight.grad.abs().sum()), 0.)
                        self.assertGreater(float(block.mlp[0].weight.grad.abs().sum()), 0.)
            if iteration:
                self.assertGreater(float(neck[1].weight.grad[224:].abs().sum()), 0.)
            optimizer.step()

    def test_migration_saves_loadable_separate_artifact_and_checks_source_profile(self):
        from turkeyworld.models import load_world_checkpoint
        from turkeyworld.vjepa2_ac import load_vjepa500_checkpoint, migrate_vjepa500_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory)/'source250.pt', Path(directory)/'expanded500.pt'
            self.save_source(self.source(), source)
            with self.assertRaisesRegex(ValueError, '250M source'):
                load_vjepa500_checkpoint(source)
            report = migrate_vjepa500_checkpoint(source, target, **self.options())
            model, payload = load_world_checkpoint(target)
            self.assertEqual(payload['vjepa500_expansion'], report)
            self.assertEqual(report['generator_parameters'], sum(parameter.numel() for parameter in model.parameters()))
            self.assertFalse(report['official_pretrained_weights_loaded'])
            self.assertTrue(source.exists())
            with self.assertRaises(FileExistsError):
                migrate_vjepa500_checkpoint(source, target, **self.options())
            with self.assertRaises(FileExistsError):
                migrate_vjepa500_checkpoint(source, source, **self.options())


if __name__ == '__main__':
    unittest.main()
