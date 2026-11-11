"""Small real latent solvers and transactional teacher/RF bundle loading."""
import copy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from turkeyworld.latent_flow_renderer import LatentFlowRenderer
from turkeyworld.models import NeuralWorldModel, STATE_DIM, STATE_SCHEMA, WorldModelConfig, load_world_checkpoint
from turkeyworld.renderer_student import (build_renderer_student, export_renderer_student,
    load_renderer_student, renderer_condition_identity, codec_tensor_fingerprint, renderer_student_config,
    initialize_renderer_student)
from turkeyworld.train import normalize_world_config


def tiny_config(**overrides):
    values = dict(kind='teacher', architecture='vjepa2_ac', image_size=32,
        endless=True, renderer_mode='flow_matching', renderer_codec='vqgan',
        renderer_codec_config=dict(image_size=32, base_channels=4, latent_channels=8,
            embedding_dim=4, n_embed=8, data_init=False, kmeans_iterations=0),
        flow_steps=3, flow_width=32, flow_depth=1, flow_heads=4,
        vjepa_width=32, vjepa_encoder_depth=1, vjepa_predictor_depth=1,
        vjepa_heads=4, vjepa_patch_size=8, vjepa_context_frames=2,
        spatial_entities=True, spatial_combat=True,
        flow_patch_size=2, flow_block_conditioning=True)
    values.update(overrides)
    return WorldModelConfig(**values)


class TinyTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = tiny_config()
        self.vjepa_backend = nn.Module()
        self.vjepa_backend.renderer = nn.Linear(STATE_DIM, 12)
        self.vjepa_backend.feature_head = nn.Linear(16, 128)
        self.vjepa_backend.core = nn.Linear(STATE_DIM, 16)
        self.flow_renderer = LatentFlowRenderer(kind='teacher', image_size=32,
            mode='flow_matching', steps=3, width=32, depth=1, heads=4,
            codec_config=self.config.renderer_codec_config,
            spatial_conditioning=True, patch_size=2, block_conditioning=True)
        self.flow_renderer.codec_loaded.fill_(True)
        self.flow_renderer.latent_mean.fill_(.1)
        self.flow_renderer.latent_std.fill_(.7)


class RendererStudentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name)/'renderer.pt'
        self.teacher = TinyTeacher().eval()

    def student(self):
        return build_renderer_student(self.teacher, width=32, depth=1, heads=4, mlp_hidden=64)

    def export(self):
        student = self.student()
        with torch.no_grad():
            student.output.weight.fill_(.001)
        export_renderer_student(self.teacher, student, self.path,
            training_steps=7, metrics={'summary': {'held_out_mae': .1}})
        return student

    def test_new_student_keeps_conditioning_codec_and_rng_without_touching_teacher(self):
        original = {key: value.clone() for key, value in self.teacher.state_dict().items()}
        rng = torch.random.get_rng_state().clone()
        with patch.object(torch.cuda, 'manual_seed_all', side_effect=AssertionError('No CUDA RNG seeding')):
            student = self.student()
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
        self.assertIs(student.codec, self.teacher.flow_renderer.codec)
        self.assertEqual((student.kind, student.mode, student.steps), ('student', 'rectified_flow', 4))
        self.assertEqual(student.patch, 2)
        self.assertTrue(student.uses_wall_conditioning)
        self.assertTrue(hasattr(student, 'spatial_condition'))
        self.assertEqual(len(student.block_spatial_condition), 1)
        self.assertEqual(len(student.block_wall_condition), 1)
        self.assertFalse(hasattr(student, 'residual_adapter'))
        for name in ('latent_mean', 'latent_std', 'fixed_noise', 'codec_loaded'):
            torch.testing.assert_close(getattr(student, name), getattr(self.teacher.flow_renderer, name), rtol=0, atol=0)
        self.assertFalse(student.codec.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in student.codec.parameters()))
        for name, value in self.teacher.state_dict().items():
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)

    def test_real_student_sampler_uses_four_euler_velocity_evaluations_not_teacher_heun(self):
        student = self.student().eval()
        teacher = self.teacher.flow_renderer.eval()
        base = torch.full((1, 3, 32, 32), .5)
        state = torch.zeros(1, STATE_DIM)
        state[:, :5] = torch.tensor([.25, .25, 0., 1., 1.])
        grid = torch.ones(1, 16, 16, dtype=torch.long)
        with torch.inference_mode(), patch.object(student, 'velocity', wraps=student.velocity) as velocity:
            result = student.sample(base, state, noise=teacher.fixed_noise, map_condition=grid)
        self.assertEqual(velocity.call_count, 4)
        self.assertEqual(result.shape, base.shape)
        self.assertTrue(bool(torch.isfinite(result).all()))
        with torch.inference_mode(), patch.object(teacher, 'velocity', wraps=teacher.velocity) as velocity:
            teacher.sample(base, state, noise=teacher.fixed_noise, map_condition=grid)
        self.assertEqual(velocity.call_count, 6)  # Three Heun steps, two evaluations each.

    def test_loading_changes_only_flow_and_allows_the_same_dynamics_to_keep_learning(self):
        expected_student = self.export()
        with torch.no_grad():
            self.teacher.vjepa_backend.core.weight.add_(.03)
        nonflow = {name: value.clone() for name, value in self.teacher.state_dict().items()
            if not name.startswith('flow_renderer.')}
        ids = {name: id(parameter) for name, parameter in self.teacher.named_parameters()
            if not name.startswith('flow_renderer.')}
        codec = self.teacher.flow_renderer.codec
        before_identity = renderer_condition_identity(self.teacher)
        metadata = load_renderer_student(self.teacher, self.path)
        self.assertEqual(metadata['training_steps'], 7)
        self.assertEqual(metadata['solver'], 'euler')
        self.assertIs(self.teacher.flow_renderer.codec, codec)
        self.assertEqual(self.teacher.config.kind, 'teacher')
        self.assertEqual(self.teacher.config.flow_kind, 'student')
        self.assertEqual(self.teacher.config.flow_steps, 4)
        self.assertEqual(renderer_condition_identity(self.teacher), before_identity)
        for name, value in self.teacher.state_dict().items():
            if name in nonflow:
                torch.testing.assert_close(value, nonflow[name], rtol=0, atol=0)
        self.assertEqual({name: id(parameter) for name, parameter in self.teacher.named_parameters()
            if name in ids}, ids)
        for name, value in expected_student.state_dict().items():
            torch.testing.assert_close(value, self.teacher.flow_renderer.state_dict()[name], rtol=0, atol=0)

    def test_changed_base_map_appearance_or_codec_is_rejected_transactionally(self):
        self.export()
        for parameter in ('base', 'appearance', 'codec'):
            with self.subTest(parameter=parameter):
                teacher = copy.deepcopy(self.teacher)
                module = {'base': teacher.vjepa_backend.renderer,
                    'appearance': teacher.vjepa_backend.feature_head,
                    'codec': teacher.flow_renderer.codec}[parameter]
                with torch.no_grad():
                    next(module.parameters()).add_(.01)
                original = {name: value.clone() for name, value in teacher.state_dict().items()}
                original_flow = teacher.flow_renderer
                original_config = teacher.config
                with self.assertRaisesRegex(ValueError, 'mismatch'):
                    load_renderer_student(teacher, self.path)
                self.assertIs(teacher.flow_renderer, original_flow)
                self.assertIs(teacher.config, original_config)
                for name, value in teacher.state_dict().items():
                    torch.testing.assert_close(value, original[name], rtol=0, atol=0)

    def test_corrupt_schema_codec_tensor_or_missing_weight_cannot_partially_replace_flow(self):
        self.export()
        payload = torch.load(self.path, map_location='cpu', weights_only=False)
        for kind in ('schema', 'codec_tensor', 'missing_weight', 'untrained', 'unloaded_codec'):
            with self.subTest(kind=kind):
                changed = copy.deepcopy(payload)
                if kind == 'schema':
                    changed['state_schema']['slots'] = 8
                elif kind == 'codec_tensor':
                    key = next(key for key in changed['model'] if key.startswith('codec.') and changed['model'][key].is_floating_point())
                    changed['model'][key].add_(1.)
                elif kind == 'missing_weight':
                    del changed['model']['output.weight']
                elif kind == 'untrained':
                    changed['training_steps'] = 0
                else:
                    changed['model']['codec_loaded'].fill_(False)
                torch.save(changed, self.path)
                flow = self.teacher.flow_renderer
                identity = renderer_condition_identity(self.teacher)
                with self.assertRaises((ValueError, RuntimeError)):
                    load_renderer_student(self.teacher, self.path)
                self.assertIs(self.teacher.flow_renderer, flow)
                self.assertEqual(renderer_condition_identity(self.teacher), identity)

    def test_default_flow_kind_preserves_legacy_config_normalization_and_rejects_invalid_override(self):
        config = asdict(tiny_config())
        legacy = {key: value for key, value in config.items() if key != 'flow_kind'}
        self.assertEqual(normalize_world_config(legacy), config)
        self.assertEqual(normalize_world_config(config), config)
        with self.assertRaises(ValueError):
            tiny_config(flow_kind='rectified')
        with self.assertRaises(ValueError):
            build_renderer_student(self.teacher, steps=12)
        with self.assertRaises(ValueError):
            export_renderer_student(self.teacher, self.student(), self.path, training_steps=0)
        self.assertFalse(self.path.exists())

    def test_full_world_roundtrip_keeps_teacher_backbone_with_student_euler_renderer(self):
        model = NeuralWorldModel(tiny_config()).eval()
        model.flow_renderer.codec_loaded.fill_(True)
        student = build_renderer_student(model, width=32, depth=1, heads=4, mlp_hidden=64)
        export_renderer_student(model, student, self.path, training_steps=1)
        original = {name: value.clone() for name, value in model.state_dict().items()
            if not name.startswith('flow_renderer.')}
        load_renderer_student(model, self.path)
        world = self.path.with_name('hybrid.pt')
        torch.save({'stage': 'world', 'config': asdict(model.config), 'model': model.state_dict(),
            'state_schema': STATE_SCHEMA, 'training_steps': 1569}, world)
        reloaded, _ = load_world_checkpoint(world)
        self.assertEqual(reloaded.config.kind, 'teacher')
        self.assertEqual((reloaded.flow_renderer.kind, reloaded.flow_renderer.steps), ('student', 4))
        self.assertEqual(reloaded.core.width, model.core.width)
        for name, value in reloaded.state_dict().items():
            if name in original:
                torch.testing.assert_close(value, original[name], rtol=0, atol=0)

    def test_play_cli_loads_bundle_before_map_initialization_and_uses_no_engine_during_rollout(self):
        from turkeyworld.neural_play import NeuralRuntime, _arguments
        from turkeyworld.engine import Game
        model = NeuralWorldModel(tiny_config()).eval()
        model.flow_renderer.codec_loaded.fill_(True)
        student = build_renderer_student(model, width=32, depth=1, heads=4, mlp_hidden=64)
        export_renderer_student(model, student, self.path, training_steps=3)
        world = self.path.with_name('teacher.pt')
        torch.save({'stage': 'world', 'config': asdict(model.config), 'model': model.state_dict(),
            'state_schema': STATE_SCHEMA, 'training_steps': 1569}, world)
        args = _arguments(['--world', str(world), '--renderer-checkpoint', str(self.path),
            '--device', 'cpu', '--headless', '--steps', '1', '--warmup', '0', '--display-size', '32'])
        runtime = NeuralRuntime(args)
        self.assertEqual(runtime.model.config.kind, 'teacher')
        self.assertEqual(runtime.model.flow_renderer.kind, 'student')
        self.assertEqual(runtime.model.flow_renderer._map_condition.shape[-2:], (16, 16))
        with patch.object(Game, 'step', side_effect=AssertionError('No engine step')), \
             patch.object(Game, 'render', side_effect=AssertionError('No engine render')):
            runtime.step([0., 0., 0., 0.])
        report = runtime.report(.1)
        self.assertEqual(report['renderer_solver'], 'euler')
        self.assertEqual(report['renderer_solver_steps'], 4)
        self.assertEqual(report['renderer_training_steps'], 3)
        self.assertEqual(report['renderer_checkpoint'], str(self.path.resolve()))
        self.assertEqual(report['ground_truth_corrections'], 0)
        self.assertFalse(report['gameplay_correctness_evaluated'])

    def compact_codec_fixture(self):
        from turkeyworld.compact_codec import build_compact_codec
        source_config = dict(image_size=32, n_embed=1488, embedding_dim=16,
            data_init=False, kmeans_iterations=0)
        self.teacher.config = tiny_config(renderer_codec_config=source_config)
        self.teacher.flow_renderer = LatentFlowRenderer(kind='teacher', image_size=32,
            mode='flow_matching', steps=3, width=32, depth=1, heads=4,
            codec_config=source_config, spatial_conditioning=True, patch_size=2,
            block_conditioning=True).eval()
        self.teacher.flow_renderer.codec_loaded.fill_(True)
        self.teacher.flow_renderer.latent_mean.fill_(.1)
        self.teacher.flow_renderer.latent_std.fill_(.7)
        source = self.teacher.flow_renderer
        compact = build_compact_codec(source.codec)
        payload = {'stage': 'distilled_vqgan_codec', 'codec_config': compact.get_config(),
            'model': compact.state_dict(), 'teacher_codec_fingerprint': codec_tensor_fingerprint(source.codec),
            'training_steps': 5, 'frozen': True, 'fixed_codebook': True,
            'quality_gate': {'passed': True, 'summary': {'decoder_ssim_mean': .96}},
            'latent_normalization': {'mean': source.latent_mean.flatten().tolist(),
                                     'std': source.latent_std.flatten().tolist()}}
        path = self.path.with_name('codec.pt')
        torch.save(payload, path)
        return path, compact, payload

    def test_qualified_compact_codec_roundtrip_preserves_original_teacher_and_codebook_space(self):
        path, compact, _ = self.compact_codec_fixture()
        original_identity = renderer_condition_identity(self.teacher)
        original_codec = self.teacher.flow_renderer.codec
        student = build_renderer_student(self.teacher, codec_checkpoint=path,
            width=32, depth=1, heads=4, mlp_hidden=64)
        self.assertIsNot(student.codec, original_codec)
        self.assertEqual(sum(parameter.numel() for parameter in student.codec.parameters()), 997539)
        self.assertEqual(codec_tensor_fingerprint(student.codec), codec_tensor_fingerprint(compact))
        self.assertEqual(student.codec.config.latent_size, 8)
        self.assertEqual(student.codec.config.embedding_dim, 16)
        metadata = export_renderer_student(self.teacher, student, self.path, training_steps=7)
        self.assertEqual(metadata['conditioning_identity'], original_identity)
        self.assertNotEqual(metadata['student_codec_identity']['codec_sha256'], original_identity['codec_sha256'])
        source_nonflow = {name: value.clone() for name, value in self.teacher.state_dict().items()
            if not name.startswith('flow_renderer.')}
        loaded = load_renderer_student(self.teacher, self.path)
        self.assertEqual(loaded['codec_distillation']['stage'], 'distilled_vqgan_codec')
        self.assertIsNot(self.teacher.flow_renderer.codec, original_codec)
        self.assertEqual(self.teacher.config.renderer_codec_config, compact.get_config())
        self.assertEqual(self.teacher.config.kind, 'teacher')
        self.assertEqual(self.teacher.flow_renderer.kind, 'student')
        for name, value in self.teacher.state_dict().items():
            if name in source_nonflow:
                torch.testing.assert_close(value, source_nonflow[name], rtol=0, atol=0)

    def test_compact_codec_rejects_unqualified_wrong_source_normalization_or_changed_codebook(self):
        path, _, payload = self.compact_codec_fixture()
        original = renderer_condition_identity(self.teacher)
        for kind in ('unqualified', 'wrong_teacher', 'normalization', 'codebook', 'over_budget'):
            with self.subTest(kind=kind):
                changed = copy.deepcopy(payload)
                if kind == 'unqualified':
                    changed['quality_gate']['passed'] = False
                elif kind == 'wrong_teacher':
                    changed['teacher_codec_fingerprint'] = 'wrong-source'
                elif kind == 'normalization':
                    changed['latent_normalization']['std'][0] += .1
                elif kind == 'codebook':
                    changed['model']['quantize.embedding.weight'][0, 0] += .1
                else:
                    changed['codec_config'] = self.teacher.flow_renderer.codec.get_config()
                    changed['model'] = self.teacher.flow_renderer.codec.state_dict()
                torch.save(changed, path)
                with self.assertRaises(ValueError):
                    build_renderer_student(self.teacher, codec_checkpoint=path,
                        width=32, depth=1, heads=4, mlp_hidden=64)
                self.assertEqual(renderer_condition_identity(self.teacher), original)

    def test_default_rf_profile_has_actual_seven_million_velocity_budget_separate_from_codec(self):
        self.compact_codec_fixture()
        config = renderer_student_config(self.teacher)
        with torch.device('meta'):
            world = NeuralWorldModel(config)
        codec_ids = {id(parameter) for parameter in world.flow_renderer.codec.parameters()}
        velocity = sum(parameter.numel() for parameter in world.flow_renderer.parameters()
            if id(parameter) not in codec_ids)
        self.assertEqual(velocity, 6_992_192)
        self.assertEqual((world.config.kind, world.flow_renderer.kind, world.flow_renderer.steps),
                         ('teacher', 'student', 4))

    def test_compact_codec_in_renderer_bundle_cannot_bypass_approval_provenance(self):
        path, _, _ = self.compact_codec_fixture()
        student = build_renderer_student(self.teacher, codec_checkpoint=path,
            width=32, depth=1, heads=4, mlp_hidden=64)
        export_renderer_student(self.teacher, student, self.path, training_steps=1)
        payload = torch.load(self.path, map_location='cpu', weights_only=False)
        payload.pop('codec_distillation')
        torch.save(payload, self.path)
        original = self.teacher.flow_renderer
        with self.assertRaises(ValueError):
            load_renderer_student(self.teacher, self.path)
        self.assertIs(self.teacher.flow_renderer, original)

    def test_embedded_compact_base_is_validated_and_auto_attached_without_changing_original_world(self):
        from turkeyworld.base_student import build_base_student, export_base_student, load_base_student
        model = NeuralWorldModel(tiny_config()).eval()
        model.flow_renderer.codec_loaded.fill_(True)
        world = self.path.with_name('original_teacher.pt')
        torch.save({'stage': 'world', 'config': asdict(model.config), 'model': model.state_dict(),
            'state_schema': STATE_SCHEMA, 'training_steps': 1569}, world)
        original = {name: value.clone() for name, value in model.state_dict().items()
            if not name.startswith('flow_renderer.')}
        head = build_base_student(model)
        base_path = self.path.with_name('base.pt')
        export_base_student(model, head, base_path, training_steps=3)
        load_base_student(model, base_path)
        student = build_renderer_student(model, width=32, depth=1, heads=4, mlp_hidden=64)
        metadata = export_renderer_student(model, student, self.path, training_steps=4)
        self.assertNotIn('base_student_bundle', metadata)
        fresh, _ = load_world_checkpoint(world)
        loaded = load_renderer_student(fresh, self.path)
        self.assertEqual(loaded['base_student']['head_parameters'], 996844)
        self.assertTrue(hasattr(fresh, 'base_student'))
        self.assertEqual(fresh.base_student_metadata['training_steps'], 3)
        for name, value in fresh.state_dict().items():
            if name in original:
                torch.testing.assert_close(value, original[name], rtol=0, atol=0)
        grid = torch.ones(1, 16, 16, dtype=torch.long)
        state = torch.zeros(1, STATE_DIM)
        state[:, :5] = torch.tensor([.25, .25, 0., 1., 1.])
        features = torch.ones(1, 128)
        with torch.inference_mode():
            maps = fresh.encode_map(grid)
            with patch.object(fresh.base_student, 'forward', wraps=fresh.base_student.forward) as called, \
                 patch.object(fresh.vjepa_backend, 'render_with_aux', side_effect=AssertionError('Old Base RGB path')):
                rgb = fresh.render_base(state, features, maps)
        self.assertEqual(called.call_count, 1)
        torch.testing.assert_close(called.call_args.args[1], features)
        self.assertEqual(rgb.shape, (1, 3, 32, 32))
        self.assertTrue(bool(torch.isfinite(rgb).all()))
        # One CLI path loads both bundled renderer components automatically.
        from turkeyworld.neural_play import NeuralRuntime, _arguments
        args = _arguments(['--world', str(world), '--renderer-checkpoint', str(self.path),
            '--headless', '--warmup', '0', '--steps', '1', '--display-size', '32'])
        runtime = NeuralRuntime(args)
        self.assertEqual(runtime.model.base_student_metadata['head_parameters'], 996844)
        self.assertEqual(runtime.model.flow_renderer.kind, 'student')

    def test_tampered_embedded_base_cannot_partially_attach_to_the_teacher(self):
        from turkeyworld.base_student import build_base_student, export_base_student, load_base_student
        base_path = self.path.with_name('base.pt')
        export_base_student(self.teacher, build_base_student(self.teacher), base_path, training_steps=1)
        load_base_student(self.teacher, base_path)
        student = self.student()
        export_renderer_student(self.teacher, student, self.path, training_steps=1)
        payload = torch.load(self.path, map_location='cpu', weights_only=False)
        fresh = TinyTeacher()
        fresh.load_state_dict({name: value for name, value in self.teacher.state_dict().items()
            if not name.startswith('base_student.')}, strict=True)
        original = fresh.flow_renderer
        key = next(iter(payload['base_student_bundle']['model']))
        payload['base_student_bundle']['model'][key].add_(1.)
        torch.save(payload, self.path)
        with self.assertRaises(ValueError):
            load_renderer_student(fresh, self.path)
        self.assertIs(fresh.flow_renderer, original)
        self.assertFalse(hasattr(fresh, 'base_student'))

    def test_unqualified_codec_requires_explicit_build_and_play_optins_preserves_validation(self):
        from turkeyworld.neural_play import _arguments
        path, _, payload = self.compact_codec_fixture()
        payload['quality_gate'] = {'passed': False, 'summary': {
            'decoder_ssim_mean': .94405, 'encoder_ssim_mean': .92943}}
        torch.save(payload, path)
        options = dict(codec_checkpoint=path, steps=1, width=32, depth=1, heads=4, mlp_hidden=64)
        with self.assertRaises(ValueError):
            build_renderer_student(self.teacher, **options)
        with self.assertRaises(ValueError):
            build_renderer_student(self.teacher, **options, allow_unqualified_codec='yes')
        student = build_renderer_student(self.teacher, **options, allow_unqualified_codec=True)
        metadata = export_renderer_student(self.teacher, student, self.path, training_steps=2)
        proof = metadata['codec_distillation']
        self.assertIs(proof['quality_gate']['passed'], False)
        self.assertEqual(proof['quality_gate'], payload['quality_gate'])
        self.assertEqual(proof['acceptance']['scope'], 'experimental_preview')
        self.assertIs(proof['acceptance']['quality_gate_passed'], False)
        self.assertIs(proof['acceptance']['allow_unqualified_codec'], True)
        self.assertEqual(proof['acceptance']['reason'], 'compressed renderer preview requested; validation preserved')
        self.assertIs(metadata['codec_quality_qualified'], False)
        self.assertIs(metadata['experimental_preview'], True)
        self.assertIs(metadata['gameplay_qualified'], False)
        flow = self.teacher.flow_renderer
        with self.assertRaises(ValueError):
            load_renderer_student(self.teacher, self.path)
        self.assertIs(self.teacher.flow_renderer, flow)
        with self.assertRaises(ValueError):
            load_renderer_student(self.teacher, self.path, allow_unqualified_codec='yes')
        loaded = load_renderer_student(self.teacher, self.path, allow_unqualified_codec=True)
        self.assertIs(loaded['codec_quality_qualified'], False)
        self.assertIs(loaded['experimental_preview'], True)
        self.assertEqual(loaded['codec_distillation']['quality_gate'], payload['quality_gate'])
        default = _arguments(['--world', str(path), '--renderer-checkpoint', str(self.path)])
        self.assertFalse(default.allow_unqualified_codec)
        explicit = _arguments(['--world', str(path), '--renderer-checkpoint', str(self.path),
            '--allow-unqualified-codec'])
        self.assertTrue(explicit.allow_unqualified_codec)

    def test_experimental_codec_does_not_relax_boolean_gate_or_integrity_checks(self):
        path, _, payload = self.compact_codec_fixture()
        for kind in ('missing_gate', 'changed_codebook', 'bad_source'):
            with self.subTest(kind=kind):
                changed = copy.deepcopy(payload)
                changed['quality_gate']['passed'] = False
                if kind == 'missing_gate':
                    changed['quality_gate'] = {'passed': 0}
                elif kind == 'changed_codebook':
                    changed['model']['quantize.embedding.weight'][0, 0] += .1
                else:
                    changed['teacher_codec_fingerprint'] = 'wrong-original-codec'
                torch.save(changed, path)
                with self.assertRaises(ValueError):
                    build_renderer_student(self.teacher, codec_checkpoint=path,
                        width=32, depth=1, heads=4, mlp_hidden=64, allow_unqualified_codec=True)

    def test_real_one_euler_step_export_load_and_cli_have_one_velocity_evaluation(self):
        from turkeyworld.neural_play import NeuralRuntime, _arguments
        model = NeuralWorldModel(tiny_config()).eval()
        model.flow_renderer.codec_loaded.fill_(True)
        world = self.path.with_name('teacher.pt')
        torch.save({'stage': 'world', 'config': asdict(model.config), 'model': model.state_dict(),
            'state_schema': STATE_SCHEMA, 'training_steps': 1598}, world)
        student = build_renderer_student(model, steps=1, width=32, depth=1, heads=4, mlp_hidden=64)
        metadata = export_renderer_student(model, student, self.path, training_steps=3)
        self.assertEqual(metadata['solver_steps'], 1)
        self.assertEqual(metadata['config']['flow_steps'], 1)
        args = _arguments(['--world', str(world), '--renderer-checkpoint', str(self.path),
            '--headless', '--warmup', '0', '--steps', '1', '--display-size', '32'])
        runtime = NeuralRuntime(args)
        self.assertEqual(runtime.model.flow_renderer.steps, 1)
        with patch.object(runtime.model.flow_renderer, 'velocity',
                wraps=runtime.model.flow_renderer.velocity) as velocity:
            runtime.step([0., 0., 0., 0.])
        self.assertEqual(velocity.call_count, 1)
        self.assertEqual(runtime.report(.1)['renderer_solver_steps'], 1)
        four = build_renderer_student(model, steps=4, width=32, depth=1, heads=4, mlp_hidden=64)
        self.assertEqual(sum(p.numel() for p in four.parameters()),
                         sum(p.numel() for p in student.parameters()))

    def test_one_step_metadata_and_config_must_agree_and_invalid_counts_never_commit(self):
        for steps in (0, 2, 12, True, 1.0, '1'):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                build_renderer_student(self.teacher, steps=steps)
        one = build_renderer_student(self.teacher, steps=1, width=32, depth=1, heads=4, mlp_hidden=64)
        export_renderer_student(self.teacher, one, self.path, training_steps=1)
        original = torch.load(self.path, map_location='cpu', weights_only=False)
        for change in ('metadata4', 'config4', 'metadata_bool'):
            with self.subTest(change=change):
                payload = copy.deepcopy(original)
                if change == 'metadata4':
                    payload['solver_steps'] = 4
                elif change == 'config4':
                    payload['config']['flow_steps'] = 4
                else:
                    payload['solver_steps'] = True
                torch.save(payload, self.path)
                flow = self.teacher.flow_renderer
                before = renderer_condition_identity(self.teacher)
                with self.assertRaises(ValueError):
                    load_renderer_student(self.teacher, self.path)
                self.assertIs(self.teacher.flow_renderer, flow)
                self.assertEqual(renderer_condition_identity(self.teacher), before)
        one.steps = 4
        with self.assertRaises(ValueError):
            export_renderer_student(self.teacher, one, self.path, training_steps=1)

    def test_four_to_one_initialization_copies_velocity_not_teacher_or_training_counters(self):
        source = self.export()
        target = build_renderer_student(self.teacher, steps=1, seed=999,
            width=32, depth=1, heads=4, mlp_hidden=64)
        teacher_flow = self.teacher.flow_renderer
        before = {name: value.clone() for name, value in self.teacher.state_dict().items()}
        codec_versions = {name: value._version for name, value in self.teacher.flow_renderer.codec.state_dict().items()}
        rng = torch.random.get_rng_state().clone()
        metadata = initialize_renderer_student(self.teacher, target, self.path)
        self.assertEqual(metadata['initialization_source_solver_steps'], 4)
        self.assertEqual(metadata['target_solver_steps'], 1)
        self.assertFalse(metadata['optimizer_state_copied'])
        self.assertFalse(metadata['training_counters_copied'])
        self.assertEqual(target.steps, 1)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
        self.assertIs(self.teacher.flow_renderer, teacher_flow)
        self.assertEqual(self.teacher.config.renderer_mode, 'flow_matching')
        self.assertEqual(self.teacher.config.flow_steps, 3)
        for name, value in self.teacher.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        self.assertEqual({name: value._version for name, value in teacher_flow.codec.state_dict().items()}, codec_versions)
        for name, value in target.state_dict().items():
            torch.testing.assert_close(value, source.state_dict()[name], rtol=0, atol=0)

    def test_initializer_rejects_other_architecture_without_touching_either_network(self):
        self.export()
        target = build_renderer_student(self.teacher, steps=1, width=64, depth=1, heads=4, mlp_hidden=64)
        before = {name: value.clone() for name, value in target.state_dict().items()}
        original = self.teacher.flow_renderer
        with self.assertRaises(ValueError):
            initialize_renderer_student(self.teacher, target, self.path)
        self.assertIs(self.teacher.flow_renderer, original)
        for name, value in target.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_old_renderer_and_base_bundles_accept_only_dynamics_mode_upgrade_without_reverting_it(self):
        from turkeyworld.base_student import build_base_student, export_base_student, load_base_student
        old = NeuralWorldModel(tiny_config()).eval()
        old.flow_renderer.codec_loaded.fill_(True)
        original_state = {name: value.clone() for name, value in old.state_dict().items()}
        base_path = self.path.with_name('base.pt')
        export_base_student(old, build_base_student(old), base_path, training_steps=1)
        load_base_student(old, base_path)
        renderer = build_renderer_student(old, steps=1, width=32, depth=1, heads=4, mlp_hidden=64)
        export_renderer_student(old, renderer, self.path, training_steps=1)
        upgraded = NeuralWorldModel(tiny_config(player_motion_mode='axis_fraction')).eval()
        missing, unexpected = upgraded.load_state_dict(original_state, strict=False)
        self.assertEqual(unexpected, [])
        self.assertTrue(all(key.startswith('vjepa_backend.player_pass_head.') for key in missing))
        head = upgraded.vjepa_backend.player_pass_head
        before = {key: value.clone() for key, value in head.state_dict().items()}
        load_renderer_student(upgraded, self.path)
        self.assertEqual(upgraded.config.player_motion_mode, 'axis_fraction')
        self.assertEqual(upgraded.vjepa_backend.config.player_motion_mode, 'axis_fraction')
        self.assertIs(upgraded.vjepa_backend.player_pass_head, head)
        self.assertTrue(hasattr(upgraded, 'base_student'))
        for key, value in head.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        bad = NeuralWorldModel(tiny_config(player_motion_mode='axis_fraction')).eval()
        bad.load_state_dict(original_state, strict=False)
        with torch.no_grad():
            next(bad.vjepa_backend.renderer.parameters()).add_(.01)
        with self.assertRaises(ValueError):
            load_renderer_student(bad, self.path)


if __name__ == '__main__':
    unittest.main()
