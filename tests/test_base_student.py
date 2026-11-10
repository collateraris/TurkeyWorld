import copy
from argparse import Namespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from turkeyworld.base_student import (BaseStudentConfig, CompactBaseHead, build_base_student,
    export_base_student, load_base_student, prepare_base_student)
from turkeyworld.distill_base import (GEOMETRY_KEYS, base_distillation_loss,
    evaluate_base_student, prepare_base_targets, BaseDistillation, validate_resume_configuration)
from turkeyworld.models import NeuralWorldModel, WorldModelConfig, parameter_counts


def tiny_teacher():
    config = WorldModelConfig(kind='teacher', architecture='vjepa2_ac', image_size=32,
        endless=True, renderer_mode='flow_matching', renderer_codec='vqgan',
        renderer_codec_config=dict(image_size=32, base_channels=4, latent_channels=8,
            embedding_dim=4, n_embed=8, data_init=False),
        flow_steps=1, flow_width=16, flow_depth=1, flow_heads=2,
        vjepa_width=16, vjepa_encoder_depth=1, vjepa_predictor_depth=1,
        vjepa_heads=2, vjepa_patch_size=8, vjepa_context_frames=2,
        spatial_entities=True, spatial_combat=True, navigation_only=True)
    teacher = NeuralWorldModel(config).eval().requires_grad_(False)
    with torch.no_grad():
        teacher.vjepa_backend.feature_head.weight.fill_(.001)
        teacher.vjepa_backend.feature_head.bias.fill_(.2)
    return teacher


def states(count=2):
    value = torch.zeros(count, 20)
    value[:, :2] = torch.tensor([.2, .3])
    value[:, 3:6] = torch.tensor([1., 1., .99])
    value[:, 8:10] = torch.tensor([.7, .6])
    value[:, 16] = 1.
    return value


class BaseStudentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)/'base.pt'
        self.teacher = tiny_teacher()

    def test_budget_includes_all_head_weights_and_no_shared_geometry_or_map(self):
        with torch.device('meta'):
            head = CompactBaseHead()
        self.assertEqual(sum(value.numel() for value in head.parameters()), 996844)
        self.assertFalse(any('map_encoder' in key or 'geometry_reader' in key for key in head.state_dict()))
        with self.assertRaises(ValueError):
            CompactBaseHead(BaseStudentConfig(width=256, heads=8))

    def test_counts_distinguish_active_renderer_from_resident_teacher_base(self):
        before = parameter_counts(self.teacher)
        self.teacher.base_student = build_base_student(self.teacher)
        measured = parameter_counts(self.teacher)
        compact = sum(value.numel() for value in self.teacher.base_student.parameters())
        flow = sum(value.numel() for value in self.teacher.flow_renderer.parameters())
        self.assertEqual(measured['active_renderer'], compact+flow)
        self.assertEqual(measured['world_total'], before['world_total']+compact)
        self.assertEqual(measured['dynamics_including_map'], before['dynamics_including_map'])
        self.assertEqual(measured['renderer'], measured['active_renderer']+measured['inactive_original_base_renderer'])

    def test_real_forward_uses_actual_features_and_geometry_and_backpropagates(self):
        head = build_base_student(self.teacher).train()
        state = states()
        with torch.no_grad():
            features = self.teacher.encode_map(torch.zeros(2, 16, 16, dtype=torch.long))
            geometry = self.teacher.vjepa_backend.geometry(state, features)
        auxiliary = torch.randn(2, 128, requires_grad=True)
        before = {key: value.clone() for key, value in self.teacher.state_dict().items()}
        with (patch('turkeyworld.engine.Game.render', side_effect=AssertionError('No engine render')),
              patch('turkeyworld.engine.Game.step', side_effect=AssertionError('No engine step'))):
            result = head(state, auxiliary, geometry)
            changed = head(state, auxiliary+1, geometry)
            self.assertEqual(result.shape, (2, 3, 32, 32))
            self.assertTrue(torch.isfinite(result).all())
            self.assertFalse(torch.equal(result, changed))
            result.square().mean().backward()
        self.assertGreater(float(auxiliary.grad.abs().sum()), 0)
        self.assertGreater(float(head.scene_projection.weight.grad.abs().sum()), 0)
        self.assertGreater(float(head.participants[0].weight.grad.abs().sum()), 0)
        self.assertGreater(float(head.blocks[0].cross_attention.in_proj_weight.grad.abs().sum()), 0)
        torch.optim.AdamW(head.parameters(), lr=1e-4).step()
        self.assertTrue(all(torch.equal(value, before[key]) for key, value in self.teacher.state_dict().items()))

    def test_seed_preserves_cpu_and_never_seeds_cuda(self):
        rng = torch.random.get_rng_state().clone()
        with patch.object(torch.cuda, 'manual_seed_all', side_effect=AssertionError('No CUDA RNG mutation')):
            first = build_base_student(self.teacher, seed=173)
            second = build_base_student(self.teacher, seed=173)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertTrue(all(torch.equal(value, second.state_dict()[key]) for key, value in first.state_dict().items()))

    def test_bundle_load_changes_only_new_head_and_accepts_payload_transactionally(self):
        head = build_base_student(self.teacher)
        export_base_student(self.teacher, head, self.path, training_steps=7)
        original = {key: value.clone() for key, value in self.teacher.state_dict().items()}
        prepared, metadata, payload = prepare_base_student(self.teacher, self.path)
        self.assertFalse(hasattr(self.teacher, 'base_student'))
        self.assertEqual(metadata['head_parameters'], 996844)
        load_base_student(self.teacher, payload)
        self.assertIs(self.teacher.base_student_bundle, payload)
        self.assertTrue(all(torch.equal(value, self.teacher.state_dict()[key]) for key, value in original.items()))
        self.assertTrue(all(torch.equal(value, self.teacher.base_student.state_dict()[key]) for key, value in prepared.state_dict().items()))
        current = self.teacher.base_student
        damaged = copy.deepcopy(payload)
        damaged['model']['rgb.weight'][0, 0] = float('nan')
        with self.assertRaises(ValueError):
            load_base_student(self.teacher, damaged)
        self.assertIs(self.teacher.base_student, current)

    def test_changed_shared_map_or_geometry_or_appearance_rejects_before_attach(self):
        export_base_student(self.teacher, build_base_student(self.teacher), self.path, training_steps=1)
        for name in ('vjepa_backend.renderer.map_encoder.tile_embedding.weight',
                     'vjepa_backend.renderer.geometry_reader.background_token',
                     'vjepa_backend.feature_head.weight'):
            original = self.teacher.state_dict()[name].clone()
            with torch.no_grad():
                self.teacher.state_dict()[name].flatten()[0].add_(1)
            with self.subTest(name=name), self.assertRaises(ValueError):
                load_base_student(self.teacher, self.path)
            self.assertFalse(hasattr(self.teacher, 'base_student'))
            with torch.no_grad():
                self.teacher.state_dict()[name].copy_(original)

    def make_data(self, split='val'):
        directory = Path(self.temporary.name)/split
        directory.mkdir(exist_ok=True)
        source = states(3).numpy()
        source[:, -1] = np.arange(3)/(30*300)
        frames = np.stack([np.full((32, 32, 3), value, np.uint8) for value in (10, 80, 200)])
        np.savez_compressed(directory/'episode.npz', split=split, seed=173 if split == 'train' else 2173, states=source,
            frames=frames, map=np.zeros((16, 16), np.int64), actions=np.zeros((2, 4), np.float32))
        return directory

    def test_buffer_own_domain_uses_actual_predicted_features_and_no_future_images(self):
        directory = self.make_data()
        rng = torch.random.get_rng_state().clone()
        with (patch.object(self.teacher, 'transition', wraps=self.teacher.transition) as transition,
              patch.object(self.teacher, 'initialize_from_image', wraps=self.teacher.initialize_from_image) as initialize):
            buffer = prepare_base_targets(self.teacher, directory, count=2, own_count=2,
                                          batch_size=2, teacher_fingerprint='teacher', seed=17)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(transition.call_count, 1)
        self.assertEqual(initialize.call_count, 1)
        self.assertTrue(torch.equal(buffer['features'][buffer['domain'] == 0], torch.zeros(2, 128)))
        self.assertGreater(float(buffer['features'][buffer['domain'] == 1].abs().sum()), 0)
        self.assertEqual(initialize.call_args.args[0].shape, (2, 3, 32, 32))
        self.assertEqual(buffer['target_rgb'].shape, (4, 3, 32, 32))
        self.assertTrue(all(buffer[name].shape[0] == 4 for name in GEOMETRY_KEYS))
        self.assertFalse(buffer['identity']['future_source_feedback'])

    def test_validation_reports_both_domains_restores_modes_rng_and_training_has_real_gradients(self):
        buffer = prepare_base_targets(self.teacher, self.make_data(), count=2, own_count=2,
                                      teacher_fingerprint='teacher', batch_size=2)
        head = build_base_student(self.teacher).train()
        rng = torch.random.get_rng_state().clone()
        report, preview = evaluate_base_student(head, buffer, batch_size=2, ssim=1., mae=0.)
        self.assertEqual(report['frames'], 4)
        self.assertEqual(set(report['domains']), {'gt', 'own'})
        self.assertFalse(report['passed'])
        self.assertEqual(len(preview), 4)
        self.assertTrue(head.training)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        sample = {key: value[np.array([0, 3])] for key, value in buffer.items() if torch.is_tensor(value)}
        loss, metrics = base_distillation_loss(head, sample)
        loss.backward()
        self.assertGreater(float(head.rgb.weight.grad.abs().sum()), 0)
        self.assertGreater(metrics['train_loss'], 0)
        with self.assertRaises(ValueError):
            evaluate_base_student(head, {**buffer, 'identity': {**buffer['identity'], 'split': 'train'}})

    def test_only_monotonic_budget_extension_is_accepted(self):
        before = {'steps': 2000, 'lr': .0002, 'target_ssim': .98}
        validate_resume_configuration(before, {**before, 'steps': 6000}, completed_step=2000)
        for changed in ({**before, 'steps': 1999}, {**before, 'lr': .001},
                        {**before, 'target_ssim': .9}, {**before, 'steps': True}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_resume_configuration(before, changed, completed_step=2000)

    def test_real_optimizer_and_rng_resume_budget_extension_matches_uninterrupted_training(self):
        buffers = {name: prepare_base_targets(self.teacher, self.make_data(name), count=2,
                    teacher_fingerprint='teacher', batch_size=2) for name in ('train', 'val')}
        source = Path(self.temporary.name)/'source.pt'
        source.write_bytes(b'immutable-test-teacher')
        def args(directory, steps):
            return Namespace(teacher=source, data_dir=Path(self.temporary.name), run_dir=directory,
                device='cpu', steps=steps, train_count=2, val_count=2, batch=2, validate_every=1,
                preview_every=1, save_every=1, required_passes=2, seed=173, lr=.0002,
                target_ssim=1., target_mae=0.)
        def buffer(teacher, directory, **unused):
            return buffers[Path(directory).name]
        direct, resumed = Path(self.temporary.name)/'direct', Path(self.temporary.name)/'resumed'
        with (patch('turkeyworld.distill_base.load_world_checkpoint', side_effect=lambda *a, **k: (copy.deepcopy(self.teacher), {'training_steps': 12})),
              patch('turkeyworld.distill_base.prepare_base_targets', side_effect=buffer)):
            BaseDistillation(args(direct, 2)).run()
            BaseDistillation(args(resumed, 1)).run()
            first = torch.load(resumed/'checkpoints/base_student_last.pt', weights_only=False)
            self.assertEqual(first['step'], 1)
            BaseDistillation(args(resumed, 2)).run()
        expected = torch.load(direct/'checkpoints/base_student_last.pt', weights_only=False)
        actual = torch.load(resumed/'checkpoints/base_student_last.pt', weights_only=False)
        self.assertEqual(actual['step'], 2)
        self.assertEqual(actual['configuration']['steps'], 2)
        for name, value in expected['model'].items():
            self.assertTrue(torch.equal(value, actual['model'][name]), name)
        for parameter, state in expected['optimizer']['state'].items():
            for name, value in state.items():
                self.assertTrue(torch.equal(value, actual['optimizer']['state'][parameter][name]), name)
        self.assertTrue(torch.equal(expected['rng_torch'], actual['rng_torch']))
        self.assertTrue(np.array_equal(expected['rng_numpy'][1], actual['rng_numpy'][1]))
        self.assertEqual(expected['rng_python'], actual['rng_python'])


if __name__ == '__main__':
    unittest.main()
