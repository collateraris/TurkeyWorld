"""Same-noise continuous endpoints, isolated Flow gradients and held-out targets."""
import copy
import json
from dataclasses import asdict
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from test_single_enemy_core import model
from turkeyworld.distill_renderer import (select_pose_panel, prepare_teacher_buffer,
    prepare_student_conditions, reflow_loss, evaluate_renderer_student, RendererDistillation, _restore_random_state)
from turkeyworld.distill_renderer import one_step_reflow_loss, _decode
from turkeyworld.engine import Game
from turkeyworld.renderer_student import build_renderer_student
from test_gradient_safety import assert_tree_equal


def source(directory, seed):
    directory.mkdir(parents=True)
    game = Game(seed=seed, endless=True, enemies_enabled=False)
    state = game.state_vector()
    frame = game.render(32, overlays=False)
    path = directory/f'episode_{seed:010d}.npz'
    np.savez_compressed(path, split=np.asarray(directory.name), seed=np.asarray(seed),
        states=np.stack([state]*3), frames=np.stack([frame]*3), map=game.map)
    return path


class AnalyticStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(.7))
        self.seen = []
        self.previous = []

    def velocity(self, noisy, time, base, state, previous=None, map_condition=None):
        self.seen.append((noisy.detach().clone(), time.detach().clone(), base.requires_grad,
                          state.requires_grad, map_condition.requires_grad))
        self.previous.append(previous)
        return self.value.expand_as(noisy)


class RendererDistillationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_analytic_same_noise_reflow_is_continuous_not_reencoded_or_independent_noise(self):
        student = AnalyticStudent()
        noise = torch.linspace(-1, 1, 2*4*8*8).reshape(2, 4, 8, 8)
        target = (noise+.7).requires_grad_()
        batch = dict(endpoint=target, noise=noise, base_latents=torch.ones_like(noise).requires_grad_(),
            state=torch.zeros(2, 20, requires_grad=True), map=torch.zeros(2, 16, 16, dtype=torch.long))
        loss, metrics = reflow_loss(student, batch, time_values=torch.tensor([.2, .8]),
                                    self_condition=False, decoded_weight=0)
        self.assertLess(float(loss.detach()), 1e-8)
        torch.testing.assert_close(student.seen[0][0], noise+torch.tensor([.2, .8])[:, None, None, None]*.7)
        self.assertEqual(student.seen[0][2:], (False, False, False))
        student.value.data.zero_()
        loss, _ = reflow_loss(student, batch, time_values=torch.tensor([.2, .8]),
                             self_condition=False, decoded_weight=0)
        loss.backward()
        self.assertLess(float(student.value.grad), 0.)
        self.assertIsNone(target.grad)
        self.assertIsNone(batch['base_latents'].grad)
        self.assertIsNone(batch['state'].grad)

    def test_mapped_cuda_rng_states_are_restored_as_cpu_byte_tensors_without_gpu(self):
        class DeviceMappedState:
            def __init__(self, cpu_value):
                self.value, self.transfers = cpu_value, 0
            def cpu(self):
                self.transfers += 1
                return self.value
        import random
        first = DeviceMappedState(torch.arange(8, dtype=torch.uint8))
        second = DeviceMappedState(torch.arange(16, dtype=torch.uint8))
        saved = {'rng_torch': torch.random.get_rng_state(), 'rng_numpy': np.random.get_state(),
                 'rng_python': random.getstate(), 'rng_cuda': [first, second]}
        def requires_cpu_byte_tensor(states):
            self.assertTrue(all(isinstance(value, torch.Tensor) and value.device.type == 'cpu'
                                and value.dtype == torch.uint8 for value in states))
        with patch('torch.cuda.set_rng_state_all', side_effect=requires_cpu_byte_tensor) as cuda_restore:
            _restore_random_state(saved, SimpleNamespace(type='cuda'))
        self.assertEqual((first.transfers, second.transfers), (1, 1))
        self.assertEqual(cuda_restore.call_count, 1)
        with patch('torch.cuda.set_rng_state_all', side_effect=AssertionError('CPU resume never touches CUDA')):
            _restore_random_state(saved, torch.device('cpu'))

    def test_one_step_objective_is_exact_tzero_single_velocity_and_does_not_sample_random_time(self):
        student = AnalyticStudent(); student.steps = 1; student.kind = 'student'
        noise = torch.randn(2, 4, 8, 8)
        target = (noise+.7).requires_grad_()
        batch = dict(endpoint=target, noise=noise.requires_grad_(), base_latents=torch.zeros_like(noise).requires_grad_(),
            state=torch.zeros(2, 20, requires_grad=True), map=torch.zeros(2, 16, 16, dtype=torch.long))
        rng = torch.random.get_rng_state().clone()
        loss, metrics = one_step_reflow_loss(student, batch, decoded_weight=0, ssim_weight=0, edge_weight=0)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(len(student.seen), 1)
        torch.testing.assert_close(student.seen[0][0], noise.detach(), rtol=0, atol=0)
        self.assertTrue(torch.equal(student.seen[0][1], torch.zeros(2)))
        self.assertIsNone(student.previous[0])
        self.assertLess(float(loss.detach()), 1e-8)
        self.assertEqual((metrics['velocity_evaluations'], metrics['solver_steps'], metrics['flow_time_mean']), (1, 1, 0.))
        student.value.data.zero_()
        loss, _ = one_step_reflow_loss(student, batch, decoded_weight=0, ssim_weight=0, edge_weight=0)
        loss.backward()
        self.assertLess(float(student.value.grad), 0.)
        for name in ('endpoint', 'noise', 'base_latents', 'state'):
            self.assertIsNone(batch[name].grad)
        student.steps = 4
        with self.assertRaisesRegex(ValueError, 'one-Euler'):
            one_step_reflow_loss(student, batch)

    def test_real_one_step_loss_matches_actual_euler_endpoint_and_learns_through_frozen_decoder(self):
        teacher = model().eval().requires_grad_(False)
        student = build_renderer_student(teacher, steps=1, width=32, depth=1, heads=4, mlp_hidden=64)
        game = Game(endless=True, enemies_enabled=False)
        state = torch.tensor(game.state_vector())[None].requires_grad_()
        grid = torch.tensor(game.map)[None]
        base = torch.rand(1, 3, 32, 32)
        with torch.no_grad():
            encoded_base = student.normalize_latents(student.codec.encode(base)['quantized'])
            noise = student._noise(None, 1, 'cpu').clone()
            target = noise+.3
            rgb = _decode(teacher.flow_renderer, target)
        batch = dict(endpoint=target, noise=noise, base_latents=encoded_base,
                     base=base, state=state, map=grid, teacher_rgb=rgb)
        with patch.object(student, 'velocity', wraps=student.velocity) as velocity:
            loss, metrics = one_step_reflow_loss(student, batch)
            self.assertEqual(velocity.call_count, 1)
            supplied = velocity.call_args
            direct = student.velocity(*supplied.args, **supplied.kwargs)
        with torch.no_grad():
            actual = student.latent_sample(base, state.detach(), noise=noise, map_condition=grid)
        torch.testing.assert_close(actual, noise+direct.detach(), rtol=0, atol=0)
        loss.backward()
        self.assertGreater(float(student.output.weight.grad.norm()), 0.)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in student.codec.parameters()))
        self.assertIsNone(state.grad)

    def test_real_teacher_targets_never_transition_and_flow_training_keeps_all_teacher_weights_exact(self):
        teacher = model().requires_grad_(False).eval()
        with torch.no_grad():
            teacher.flow_renderer.output.weight.normal_(std=.01)
        before = {name: value.clone() for name, value in teacher.state_dict().items()}
        student = build_renderer_student(teacher, width=32, depth=1, heads=4, mlp_hidden=64)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source(root/'train', 101); source(root/'val', 202)
            with patch.object(teacher, 'transition', side_effect=AssertionError('No world transition')):
                train = prepare_teacher_buffer(teacher, root/'train', count=2, batch_size=2,
                    teacher_fingerprint='frozen', cache_path=root/'train.pt')
                validation = prepare_teacher_buffer(teacher, root/'val', count=2, batch_size=2,
                    teacher_fingerprint='frozen')
            self.assertEqual(train['identity']['noise_policy'], 'shared_runtime_fixed_noise')
            self.assertNotEqual(train['identity']['fingerprint'], validation['identity']['fingerprint'])
            self.assertNotEqual(train['identity']['frames'][0]['seed'], validation['identity']['frames'][0]['seed'])
            torch.testing.assert_close(train['noise'][0], teacher.flow_renderer._noise(None, 1, 'cpu')[0])
            optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=1e-3)
            student.train()
            loss, _ = reflow_loss(student, train, self_condition=False)
            loss.backward()
            self.assertGreater(float(student.output.weight.grad.norm()), 0.)
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in student.codec.parameters()))
            optimizer.step()
            for name, value in teacher.state_dict().items():
                self.assertTrue(torch.equal(value, before[name]), name)
            report, previews = evaluate_renderer_student(student, validation, batch_size=2, step=1)
            self.assertEqual(report['frames'], 2)
            self.assertTrue(report['source_held_out'])
            self.assertEqual(report['student_steps'], 4)
            self.assertIn('teacher_vs_source', report)
            self.assertTrue(student.training)
            self.assertFalse(student.codec.training)
            self.assertEqual(len(previews), 2)
            with self.assertRaisesRegex(ValueError, 'held-out'):
                evaluate_renderer_student(student, train)
            # Cache identity prevents silently changing teacher targets on resume.
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                prepare_teacher_buffer(teacher, root/'train', count=2, batch_size=2,
                    teacher_fingerprint='different', cache_path=root/'train.pt')

    def test_pose_selection_is_unique_repeatable_split_checked_and_does_not_use_global_rng(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'val'
            source(path, 14)
            state = copy.deepcopy(np.random.get_state())
            panel = select_pose_panel(path, 3, 173, image_size=32)
            self.assertEqual(panel, select_pose_panel(path, 3, 173, image_size=32))
            self.assertEqual(len(set(panel)), 3)
            self.assertTrue(np.array_equal(state[1], np.random.get_state()[1]))
            with self.assertRaisesRegex(ValueError, 'Insufficient'):
                select_pose_panel(path, 4, 1, image_size=32)
            saved = dict(np.load(panel[0][0], allow_pickle=False))
            saved['split'] = np.asarray('train')
            np.savez_compressed(panel[0][0], **saved)
            with self.assertRaisesRegex(ValueError, 'split/state'):
                select_pose_panel(path, 2, 1, image_size=32)

    def test_compact_base_encoder_conditions_have_new_identity_without_changing_teacher_labels(self):
        teacher = model().eval().requires_grad_(False)
        student = build_renderer_student(teacher, width=32, depth=1, heads=4, mlp_hidden=64)
        # A separate encoder stands for a qualified compact codec. The public
        # bundle loader validates qualification; this cache helper must use
        # the actual attached encoder and bind its exact weights.
        student.codec = copy.deepcopy(teacher.flow_renderer.codec)
        with torch.no_grad():
            student.codec.encoder.conv_in.weight.add_(.02)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source(root/'val', 15)
            reference = prepare_teacher_buffer(teacher, root/'val', count=2, batch_size=2,
                                                teacher_fingerprint='teacher')
            originals = {name: value.clone() for name, value in reference.items() if torch.is_tensor(value)}
            artifact = root/'base.pt'; artifact.write_bytes(b'qualified-base-fixture')
            teacher.base_student = nn.Identity()
            changed_base = (reference['base']+.02).clamp(0, 1)
            with patch.object(teacher, 'render_base', return_value=changed_base) as base_call, \
                    patch.object(student.codec, 'encode', wraps=student.codec.encode) as encoder:
                conditions = prepare_student_conditions(teacher, student, reference, batch_size=2,
                    base_checkpoint=artifact, cache_path=root/'student_conditions.pt')
            self.assertEqual(base_call.call_count, 1)
            torch.testing.assert_close(encoder.call_args[0][0], changed_base, rtol=0, atol=0)
            torch.testing.assert_close(conditions['base'], changed_base, rtol=0, atol=0)
            for name in ('noise', 'endpoint', 'teacher_rgb', 'source_rgb', 'state', 'map'):
                torch.testing.assert_close(conditions[name], originals[name], rtol=0, atol=0)
            for name, value in originals.items():
                torch.testing.assert_close(reference[name], value, rtol=0, atol=0)
            self.assertNotEqual(conditions['identity']['fingerprint'], reference['identity']['fingerprint'])
            self.assertFalse(conditions['identity']['student_conditioning']['teacher_targets_modified'])
            with patch.object(teacher, 'render_base', side_effect=AssertionError('Reuse exact cache')):
                cached = prepare_student_conditions(teacher, student, reference, base_checkpoint=artifact,
                    cache_path=root/'student_conditions.pt')
            torch.testing.assert_close(cached['base_latents'], conditions['base_latents'], rtol=0, atol=0)
            artifact.write_bytes(b'changed-compact-base')
            with self.assertRaisesRegex(ValueError, 'cache codec/Base'):
                prepare_student_conditions(teacher, student, reference, base_checkpoint=artifact,
                    cache_path=root/'student_conditions.pt')

    def test_real_short_training_resume_optimizer_rng_best_export_and_completed_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source(root/'rgb/train', 101); source(root/'rgb/val', 202)
            teacher = model().eval()
            checkpoint = root/'teacher.pt'
            torch.save({'stage': 'world', 'config': asdict(teacher.config),
                        'model': teacher.state_dict(), 'training_steps': 7}, checkpoint)
            def args(name):
                return SimpleNamespace(teacher=checkpoint, data_dir=root/'rgb', run_dir=root/name,
                    steps=2, train_count=2, val_count=2, batch=2, validate_every=1, preview_every=1,
                    save_every=1, required_passes=99, seed=173, lr=1e-3, device='cpu',
                    target_ssim=1., target_mae=0.)
            def tiny(teacher, **options):
                return build_renderer_student(teacher, **options, width=32, depth=1, heads=4, mlp_hidden=64)
            with patch('turkeyworld.renderer_student.build_renderer_student', side_effect=tiny):
                RendererDistillation(args('full')).run()
                calls = 0
                def interrupted(*values, **options):
                    nonlocal calls
                    calls += 1
                    if calls == 2:
                        raise RuntimeError('Controlled interruption after saved update1')
                    return reflow_loss(*values, **options)
                with patch('turkeyworld.distill_renderer.reflow_loss', side_effect=interrupted):
                    with self.assertRaisesRegex(RuntimeError, 'Controlled interruption'):
                        RendererDistillation(args('resume')).run()
                RendererDistillation(args('resume')).run()
                for_run = lambda name: torch.load(root/name/'checkpoints/renderer_student_last.pt', weights_only=False)
                full, resumed = for_run('full'), for_run('resume')
                self.assertEqual((full['step'], resumed['step']), (2, 2))
                for key in ('model', 'optimizer', 'rng_torch', 'rng_python'):
                    assert_tree_equal(self, full[key], resumed[key])
                self.assertTrue(np.array_equal(full['rng_numpy'][1], resumed['rng_numpy'][1]))
                for name in ('full', 'resume'):
                    exported = torch.load(root/name/'checkpoints/renderer_student.pt', weights_only=False)
                    self.assertEqual(exported['training_steps'], for_run(name)['best']['step'])
                    self.assertEqual(exported['metrics'], for_run(name)['best']['report'])
                    self.assertEqual(exported['solver_steps'], 4)
                with patch('turkeyworld.distill_renderer.reflow_loss', side_effect=AssertionError('No extra update')):
                    RendererDistillation(args('resume')).run()
                extended = args('resume'); extended.steps = 3
                RendererDistillation(extended).run()
                uninterrupted = args('full3'); uninterrupted.steps = 3
                RendererDistillation(uninterrupted).run()
                for key in ('model', 'optimizer', 'rng_torch', 'rng_python'):
                    assert_tree_equal(self, for_run('resume')[key], for_run('full3')[key])
                self.assertEqual(for_run('resume')['initial_budget'], 2)
                self.assertEqual(for_run('resume')['budget_history'][0]['at_optimizer_step'], 2)
                with self.assertRaisesRegex(ValueError, 'resume configuration'):
                    RendererDistillation(args('resume')).run()
                # A new one-step phase imports only old four-step weights.
                # Its teacher labels remain the original CFM snapshot, while
                # Adam/RNG and the optimizer counter start fresh.
                old_artifact = root/'full/checkpoints/renderer_student.pt'
                before_source = old_artifact.read_bytes()
                one = args('one'); one.steps = 1; one.solver_steps = 1; one.initialize_renderer = old_artifact
                RendererDistillation(one).run()
                one_checkpoint = for_run('one')
                self.assertEqual(one_checkpoint['step'], 1)
                self.assertEqual(one_checkpoint['initialization']['source_solver_steps'], 4)
                self.assertEqual(one_checkpoint['initialization']['target_solver_steps'], 1)
                self.assertTrue(one_checkpoint['initialization']['optimizer_reset'])
                self.assertEqual(one_checkpoint['initialization']['initial_optimizer_step'], 0)
                self.assertEqual(max(float(value['step']) for value in one_checkpoint['optimizer']['state'].values()), 1.)
                one_bundle = torch.load(root/'one/checkpoints/renderer_student.pt', weights_only=False)
                self.assertEqual(one_bundle['solver_steps'], 1)
                initial = json.loads((root/'one/validation_initial.json').read_text())
                self.assertEqual((initial['step'], initial['student_steps']), (0, 1))
                recorded = json.loads((root/'one/metrics.jsonl').read_text().splitlines()[0])
                self.assertEqual((recorded['velocity_evaluations'], recorded['flow_time_mean']), (1, 0.))
                self.assertEqual(before_source, old_artifact.read_bytes())

    def test_experimental_codec_opt_in_trains_without_turning_failed_codec_gate_into_pass(self):
        from test_distill_codec import codec_config, teacher_flow, renderer_buffer
        from turkeyworld.compact_codec import build_compact_codec
        from turkeyworld.distill_codec import (prepare_codec_targets, codec_distillation_loss,
            evaluate_codec_student, export_codec_student)
        from turkeyworld.models import NeuralWorldModel
        from turkeyworld.vq_world import teacher175_vq_config
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source(root/'rgb/train', 101); source(root/'rgb/val', 202)
            config = teacher175_vq_config(codec_config(), vjepa_width=32, vjepa_heads=4,
                vjepa_encoder_depth=1, vjepa_predictor_depth=1, vjepa_readout_mlp_ratio=1,
                flow_width=16, flow_depth=1, flow_heads=4, flow_steps=1,
                flow_adapter_width=0, flow_adapter_depth=0, flow_adapter_heads=0)
            teacher = NeuralWorldModel(config)
            original = teacher_flow()
            teacher.flow_renderer.load_codec({'codec_config': original.codec.get_config(), 'model': original.codec.state_dict()})
            compact = build_compact_codec(teacher.flow_renderer.codec)
            targets = prepare_codec_targets(teacher.flow_renderer, renderer_buffer('val'), batch_size=2)
            loss, _ = codec_distillation_loss(compact, targets, teacher.flow_renderer.latent_std)
            optimizer = torch.optim.AdamW([p for p in compact.parameters() if p.requires_grad], lr=1e-3)
            loss.backward(); optimizer.step()
            report, _ = evaluate_codec_student(compact, teacher.flow_renderer.codec, targets,
                                               ssim=1., mae=0., batch_size=4)
            self.assertFalse(report['passed'])
            codec = root/'compact.pt'
            export_codec_student(compact, teacher.flow_renderer, codec, training_steps=1, report=report)
            checkpoint = root/'teacher.pt'
            torch.save({'stage': 'world', 'config': asdict(config), 'model': teacher.state_dict(), 'training_steps': 7}, checkpoint)
            def args(name, allow):
                return SimpleNamespace(teacher=checkpoint, data_dir=root/'rgb', run_dir=root/name,
                    codec_checkpoint=codec, base_checkpoint=None, allow_unqualified_codec=allow,
                    steps=1, train_count=2, val_count=2, batch=2, validate_every=1, preview_every=1,
                    save_every=1, required_passes=99, seed=173, lr=1e-3, device='cpu', target_ssim=1., target_mae=0.)
            def tiny(teacher, **options):
                return build_renderer_student(teacher, **options, width=32, depth=1, heads=4, mlp_hidden=64)
            with patch('turkeyworld.renderer_student.build_renderer_student', side_effect=tiny):
                with self.assertRaisesRegex(ValueError, 'Unqualified|qualified'):
                    RendererDistillation(args('strict', False)).run()
                RendererDistillation(args('experimental', True)).run()
            status = json.loads((root/'experimental/status.json').read_text())
            self.assertFalse(status['codec_quality_gate_passed'])
            self.assertTrue(status['experimental_preview'])
            self.assertFalse(status['world_quality_qualified'])
            exported = torch.load(root/'experimental/checkpoints/renderer_student.pt', weights_only=False)
            self.assertFalse(exported['codec_distillation']['quality_gate']['passed'])
            self.assertEqual(exported['codec_distillation']['acceptance']['scope'], 'experimental_preview')


if __name__ == '__main__':
    unittest.main()
