"""Shared-codec world runner guards, profile hooks and frozen optimizer behavior."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Optional PyTorch dependency is not installed')
class VQWorldRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def codec_info(self):
        from dataclasses import asdict
        from turkeyworld.vqgan import VQGANConfig
        return {'codec_config': asdict(VQGANConfig(n_embed=1488)), 'generator_parameters': 4_763_731,
            'codec_fingerprint': 'test-frozen-codec', 'training_steps': 2000,
            'quality_gate': {'passed': True, 'frozen': True, 'consecutive_passes': 3, 'required_passes': 3}}

    def test_unqualified_codec_is_rejected_before_base_runner_or_world_initialization(self):
        from turkeyworld.train_vq_world import LatentWorldExperiment
        from turkeyworld.train_500m import MassiveExperiment
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)/'unqualified.pt'
            torch.save({'stage': 'vqgan_codec', 'codec_config': {}, 'model': {}}, checkpoint)
            args = SimpleNamespace(codec_checkpoint=checkpoint)
            with patch.object(MassiveExperiment, '__init__', side_effect=AssertionError('No world startup')):
                with self.assertRaisesRegex(ValueError, 'three consecutive'):
                    LatentWorldExperiment(args)

    def test_constructor_records_shared_codec_and_selected_student_before_any_world_work(self):
        from turkeyworld.train_vq_world import LatentWorldExperiment
        from turkeyworld.train_500m import build_parser
        with tempfile.TemporaryDirectory() as directory:
            args = build_parser().parse_args(['--device', 'cpu', '--run-dir', directory])
            args.codec_checkpoint = Path(directory)/'codec.pt'
            args.control_run_dir = None
            with patch('turkeyworld.train_vq_world.validate_frozen_codec', return_value=self.codec_info()):
                experiment = LatentWorldExperiment(args)
            self.assertEqual(experiment.PROFILE, '500vq')
            self.assertEqual(experiment.state['model_parameters']['teacher_total'], 504_857_752)
            self.assertEqual(experiment.state['model_parameters']['target_student_total'], 22_475_096)
            self.assertTrue(experiment.state['codec_training']['frozen'])
            self.assertEqual(experiment.state['codec_training']['consecutive_passes'], 3)
            self.assertEqual(experiment.state['student_plan']['config']['renderer_codec'], 'vqgan')
            self.assertEqual(experiment.state['student_plan']['config']['vjepa_encoder_depth'], 4)
            self.assertTrue(experiment.state['pipeline']['shared_codec_for_teacher_and_student'])

    def test_student_model_hook_uses_real_complete_meta_profile_and_loads_the_shared_codec(self):
        from turkeyworld.train_vq_world import LatentWorldExperiment
        from turkeyworld.latent_flow_renderer import LatentFlowRenderer
        experiment = object.__new__(LatentWorldExperiment)
        experiment.codec_info = self.codec_info()
        experiment.device = torch.device('meta')
        experiment.args = SimpleNamespace(codec_checkpoint=Path('qualified-codec.pt'))
        calls = []

        def load_codec(model, checkpoint):
            calls.append((model, checkpoint))
            return model

        with torch.device('meta'), patch.object(LatentFlowRenderer, 'load_codec', autospec=True, side_effect=load_codec):
            model = experiment.make_student_model()
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 22_475_096)
        self.assertEqual(model.config.renderer_codec, 'vqgan')
        self.assertEqual((model.core.width, len(model.core.encoder.blocks), len(model.core.predictor.blocks)), (320, 4, 5))
        self.assertEqual(model.flow_renderer.steps, 4)
        self.assertEqual(model.flow_renderer.blocks[0].mlp[0].out_features, 736)
        self.assertEqual(calls, [(model.flow_renderer, experiment.args.codec_checkpoint)])

    def test_parent_unfreeze_and_all_parameter_optimizer_never_update_the_codec(self):
        from turkeyworld.models import NeuralWorldModel
        from turkeyworld.vqgan import VQGAN, VQGANConfig
        from turkeyworld.vq_world import student_vq_config
        codec_config = VQGANConfig(base_channels=8, latent_channels=16, embedding_dim=4,
                                  n_embed=32, data_init=False, kmeans_iterations=0)
        model = NeuralWorldModel(student_vq_config(codec_config, vjepa_width=32,
            vjepa_encoder_depth=1, vjepa_predictor_depth=1, vjepa_heads=4,
            flow_width=32, flow_depth=1, flow_heads=4, flow_mlp_hidden=96))
        codec = VQGAN(codec_config)
        model.flow_renderer.load_codec({'config': codec.get_config(), 'model': codec.state_dict()})
        model.requires_grad_(True)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.flow_renderer.codec.parameters()))
        # Even a containing module that toggles Parameters directly is repaired
        # by ordinary .train(), through the latent renderer's frozen-codec hook.
        parent = torch.nn.Sequential(model).requires_grad_(True).train()
        self.assertFalse(model.flow_renderer.codec.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.flow_renderer.codec.parameters()))
        optimizer = torch.optim.AdamW(parent.parameters(), lr=1e-3, weight_decay=.1)
        before = {name: value.clone() for name, value in model.flow_renderer.codec.state_dict().items()}
        state, base = torch.zeros(1, 90), torch.rand(1, 3, 128, 128)
        state[:, 4] = 1.
        image = model.flow_renderer.sample(base, state)
        image.square().mean().backward()
        self.assertGreater(float(model.flow_renderer.output.weight.grad.abs().sum()), 0.)
        optimizer.step()
        for parameter in model.flow_renderer.codec.parameters():
            self.assertIsNone(parameter.grad)
            self.assertNotIn(parameter, optimizer.state)
        for name, value in model.flow_renderer.codec.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_initial_migration_hook_uses_durable_new_phase_resume_path(self):
        from turkeyworld.train_vq_world import LatentWorldExperiment
        experiment = object.__new__(LatentWorldExperiment)
        experiment.root = Path('new-vq-run')
        experiment.args = SimpleNamespace(source=Path('pixel-world.pt'), codec_checkpoint=Path('frozen-codec.pt'))
        initial = experiment.root/'checkpoints'/'teacher_vjepa500vq_initial.pt'
        expected = {'resume_checkpoint': str(experiment.root/'checkpoints'/'teacher500vq_world.pt')}
        with patch('turkeyworld.train_vq_world.migrate_teacher_to_vq', return_value=expected) as migrate:
            report = experiment.create_initial_teacher(initial)
        self.assertEqual(report, expected)
        migrate.assert_called_once_with(experiment.args.source, experiment.args.codec_checkpoint, initial,
            resume_output=experiment.root/'checkpoints'/'teacher500vq_world.pt')

    def test_parent_control_stop_request_is_forwarded_to_the_world_runner(self):
        from turkeyworld.train_vq_world import LatentWorldExperiment
        from turkeyworld.train_500m import MassiveExperiment
        with tempfile.TemporaryDirectory() as directory:
            control, world = Path(directory)/'control', Path(directory)/'world'
            control.mkdir();world.mkdir()
            (control/'stop_training').touch()
            experiment = object.__new__(LatentWorldExperiment)
            experiment.root = world
            experiment.args = SimpleNamespace(control_run_dir=control)
            with patch.object(MassiveExperiment, 'check_stop', return_value='saved') as stop:
                self.assertEqual(experiment.check_stop('model', 'optimizer', 'phase', 10, {}), 'saved')
            self.assertTrue((world/'stop_training').exists())
            stop.assert_called_once_with('model', 'optimizer', 'phase', 10, {})



if __name__ == '__main__':
    unittest.main()
