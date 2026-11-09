"""175M fresh lineage cannot inherit500M training clocks or optimizer state."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import test_fresh_vq_world as fresh_fixture
from test_joint_pretrain_adversarial import GANData, GANHarness
from test_joint_renderer_training import JointHarness, TinyVQWorld
from test_object_render_training import tiny_flow
from turkeyworld.adversarial_training import AdversarialTrainer
from turkeyworld.models import NeuralWorldModel
from turkeyworld.train_vq_world import LatentWorldExperiment, teacher_vq_config
from turkeyworld.vqgan import VQGAN


class Teacher175RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def construct(self, root, *, size='175m_vqgan', from_scratch=True):
        fixture = fresh_fixture.FreshVQWorldTests()
        args = fixture.args(root)
        args.teacher_size, args.from_scratch, args.gradient_accumulation_steps = size, from_scratch, 16
        with patch('turkeyworld.train_vq_world.validate_frozen_codec', return_value=fixture.codec_info()):
            return LatentWorldExperiment(args)

    def test_selected_profile_reports175m_and16_microsteps_and_untrained_student(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment = self.construct(directory)
        self.assertEqual(experiment.PROFILE, '175vq')
        self.assertEqual(experiment.PARAMETER_TARGET, 175_000_000)
        self.assertEqual(experiment.state['teacher_size'], '175m_vqgan')
        self.assertEqual(experiment.state['model_parameters']['teacher_total'], 174_910_616)
        self.assertNotIn('ema_target_total', experiment.state['model_parameters'])
        self.assertEqual(experiment.state['training_protocol']['effective_batch_size'], 16)
        self.assertEqual(experiment.global_step, 0)
        self.assertEqual(experiment.state['student_plan']['status'], 'pending')
        self.assertEqual(experiment.state['student_plan']['parameters'], 22_475_096)
        self.assertEqual(experiment.state['initialization']['teacher_parameters'], 174_910_616)

    def test175_requires_fresh_mode_instead_of_silently_migrating500m(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, '--from-scratch'):
                self.construct(directory, from_scratch=False)

    def test_profile_marker_rejects_cross_size_resume_and_config_changes(self):
        for wrong_size in ('175m_vqgan', '500m_vqgan'):
            with self.subTest(size=wrong_size), tempfile.TemporaryDirectory() as directory:
                original = self.construct(directory, size=wrong_size)
                marker = Path(directory)/'initialization.json'
                marker.write_text(json.dumps(original.state['initialization']), encoding='utf-8')
                other = '500m_vqgan' if wrong_size == '175m_vqgan' else '175m_vqgan'
                with self.assertRaisesRegex(ValueError, 'different scratch experiment'):
                    self.construct(directory, size=other)
        with tempfile.TemporaryDirectory() as directory:
            original = self.construct(directory)
            marker = {**original.state['initialization'], 'teacher_config_fingerprint': 'wrong-config'}
            (Path(directory)/'initialization.json').write_text(json.dumps(marker), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'different scratch experiment'):
                self.construct(directory)

    def test_initial175_checkpoint_reads_only_codec_and_contains_no_previous_training_state(self):
        codec_config = dict(image_size=128, base_channels=8, latent_channels=16,
                            embedding_dim=4, n_embed=32, data_init=False, kmeans_iterations=0)
        tiny = replace(teacher_vq_config(codec_config, '175m_vqgan'), vjepa_width=32,
            vjepa_encoder_depth=1, vjepa_predictor_depth=1, vjepa_heads=4,
            vjepa_readout_mlp_ratio=1, vjepa_gradient_checkpointing=False,
            flow_width=32, flow_depth=1, flow_heads=4,
            flow_adapter_width=0, flow_adapter_depth=0, flow_adapter_heads=0)
        with torch.device('meta'):
            expected = sum(p.numel() for p in NeuralWorldModel(tiny).parameters())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            codec = VQGAN(codec_config)
            codec_path = root/'frozen-codec.pt'
            torch.save({'codec_config': codec_config, 'model': codec.state_dict(),
                        'latent_normalization': {'mean': [.1]*4, 'std': [.7]*4}}, codec_path)
            source = root/'old500.pt'
            source.write_bytes(b'Never load old world, renderer, GAN or optimizer')
            experiment = object.__new__(LatentWorldExperiment)
            experiment.root, experiment.PROFILE, experiment.teacher_size = root, '175vq', '175m_vqgan'
            experiment.args = SimpleNamespace(from_scratch=True, codec_checkpoint=codec_path, source=source)
            experiment.codec_info = {'codec_config': codec_config}
            experiment.state = {'initialization': {'teacher_size': '175m_vqgan'},
                                'enemy_policy': {'version': 'test'}}
            initial = root/'checkpoints/teacher_vjepa175vq_initial.pt'
            loaded, real_load = [], torch.load

            def only_codec(path, *args, **kwargs):
                self.assertEqual(Path(path), codec_path)
                loaded.append(Path(path))
                return real_load(path, *args, **kwargs)

            with patch('turkeyworld.train_vq_world.teacher_vq_config', return_value=tiny) as factory, \
                 patch.object(experiment, 'teacher_parameter_count', return_value=expected), \
                 patch('torch.load', side_effect=only_codec), \
                 patch('turkeyworld.train_vq_world.migrate_teacher_to_vq', side_effect=AssertionError('No migration')):
                experiment.create_initial_teacher(initial)
            factory.assert_called_once_with(codec_config, teacher_size='175m_vqgan')
            self.assertEqual(loaded, [codec_path])
            saved = real_load(initial, weights_only=False, map_location='cpu')
            self.assertEqual(saved['pipeline_phase'], 'teacher175vq_initial')
            self.assertEqual(saved['training_steps'], 0)
            self.assertEqual(saved['global_training_step'], 0)
            for key in ('optimizer', 'ema_target_encoder', 'adversarial_training', 'rng_numpy', 'rng_torch', 'rng_cuda'):
                self.assertNotIn(key, saved)
            for name, value in codec.state_dict().items():
                self.assertTrue(torch.equal(value, saved['model']['flow_renderer.codec.'+name]), name)

    def test_fresh_joint_gan_group_has16_microsteps_and_only_one_g_d_ema_update(self):
        class FreshHarness(GANHarness):
            def __init__(self, root):
                super().__init__(root)
                self.global_step, self.fragments_seen = 0, 0
                self.args.gradient_accumulation_steps = 16

            def restore(self, model, optimizer, phase):
                self.optimizer_initially_empty = not optimizer.state
                return JointHarness.restore(self, model, optimizer, phase)

            def _setup_adversarial(self, phase):
                self._adversarial_trainer = AdversarialTrainer('cpu', base_channels=4,
                    weight=.005, feature_weight=.01, warmup_steps=32, ramp_steps=128, r1_gamma=0.)
                self._adversarial_phase = phase
                self.discriminator_initially_empty = not self._adversarial_trainer.optimizer.state

        model, data = TinyVQWorld(), GANData()
        model.flow_renderer = tiny_flow(64)
        model.flow_renderer.kind, model.flow_renderer.mode, model.flow_renderer.steps = 'teacher', 'flow_matching', 12
        with tempfile.TemporaryDirectory() as directory:
            experiment = FreshHarness(directory)
            experiment.fit(model, data, 'teacher175vq_world', 1, pretrain=True)
        self.assertEqual(data.calls, 16)
        self.assertEqual(experiment.global_step, 1)
        self.assertEqual(experiment.fragments_seen, 16)
        self.assertEqual(experiment._adversarial_trainer.updates, 1)
        self.assertEqual(int(experiment.ema.state_dict()['_ema_steps']), 1)
        self.assertTrue(experiment.optimizer_initially_empty)
        self.assertTrue(experiment.discriminator_initially_empty)
        self.assertTrue(all(float(entry['step']) == 1 for entry in experiment.optimizer.state.values()))
        self.assertTrue(all(float(entry['step']) == 1 for entry in experiment._adversarial_trainer.optimizer.state.values()))
        self.assertEqual(experiment.state['training_batch']['effective_batch_size'], 16)


if __name__ == '__main__':
    unittest.main()
