"""Compact E/D training, original terminal-code coverage and exact resume."""
import copy
from dataclasses import asdict
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from turkeyworld.compact_codec import build_compact_codec
from turkeyworld.distill_codec import (prepare_codec_targets, codec_distillation_loss,
    evaluate_codec_student, export_codec_student, CodecDistillation, _same_codebook)
from turkeyworld.distill_renderer import _digest
from turkeyworld.latent_flow_renderer import LatentFlowRenderer
from turkeyworld.models import NeuralWorldModel
from turkeyworld.vqgan import VQGAN, VQGANConfig
from turkeyworld.vq_world import teacher175_vq_config
from test_gradient_safety import assert_tree_equal


def codec_config():
    return VQGANConfig(image_size=32, base_channels=8, latent_channels=16, embedding_dim=16,
        n_embed=1488, attention=False, data_init=False, kmeans_iterations=0)


def teacher_flow():
    config = codec_config()
    flow = LatentFlowRenderer(kind='teacher', image_size=32, width=16, depth=1, heads=4,
        codec_config=asdict(config), steps=1, patch_size=2)
    source = VQGAN(config)
    flow.load_codec({'codec_config': source.get_config(), 'model': source.state_dict()})
    flow.latent_mean.fill_(.13); flow.latent_std.fill_(.7)
    return flow.eval().requires_grad_(False)


def renderer_buffer(split, fingerprint='teacher'):
    generator = torch.Generator().manual_seed(17 if split == 'train' else 19)
    seed = 10 if split == 'train' else 20
    return {'identity': {'split': split, 'teacher_fingerprint': fingerprint, 'fingerprint': split+'-panel',
                        'frames': [{'seed': seed, 'episode': split+'.npz', 'frame': 0}]},
        'state': torch.zeros(2, 20), 'base': torch.rand(2, 3, 32, 32, generator=generator),
        'source_rgb': torch.rand(2, 3, 32, 32, generator=generator),
        'teacher_rgb': torch.rand(2, 3, 32, 32, generator=generator),
        'endpoint': torch.randn(2, 16, 8, 8, generator=generator)}


class CompactCodecDistillationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_four_domains_include_actual_flow_terminal_and_encoder_decoder_learn_with_fixed_quantizer(self):
        teacher = teacher_flow()
        teacher_before = {name: value.clone() for name, value in teacher.codec.state_dict().items()}
        buffer = renderer_buffer('train')
        original_endpoint = buffer['endpoint'].clone()
        targets = prepare_codec_targets(teacher, buffer, batch_size=2)
        self.assertEqual(torch.bincount(targets['domain']).tolist(), [2, 2, 2, 2])
        expected = teacher.codec.quantize(buffer['endpoint']*.7+.13)['quantized']
        torch.testing.assert_close(targets['quantized'][targets['domain'] == 3], expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(buffer['endpoint'], original_endpoint))
        student = build_compact_codec(teacher.codec)
        self.assertEqual(sum(p.numel() for p in student.parameters()), 997539)
        self.assertTrue(_same_codebook(student, teacher.codec))
        optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=1e-3)
        before_e, before_d = student.encoder.conv_in.weight.clone(), student.decoder.conv_out.weight.clone()
        student.train()
        loss, metrics = codec_distillation_loss(student, targets, teacher.latent_std)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(student.encoder.conv_in.weight.grad.norm()), 0.)
        self.assertGreater(float(student.decoder.conv_out.weight.grad.norm()), 0.)
        self.assertTrue(all(p.grad is None for p in student.quantize.parameters()))
        optimizer.step()
        self.assertFalse(torch.equal(before_e, student.encoder.conv_in.weight))
        self.assertFalse(torch.equal(before_d, student.decoder.conv_out.weight))
        self.assertTrue(_same_codebook(student, teacher.codec))
        for name, value in teacher.codec.state_dict().items():
            self.assertTrue(torch.equal(value, teacher_before[name]), name)
        self.assertFalse(student.quantize.training)

    def test_heldout_report_export_binding_and_training_split_guard(self):
        teacher = teacher_flow()
        student = build_compact_codec(teacher.codec)
        targets = prepare_codec_targets(teacher, renderer_buffer('val'), batch_size=2)
        student.train()
        rng = torch.random.get_rng_state().clone()
        report, preview = evaluate_codec_student(student, teacher.codec, targets, batch_size=4)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertTrue(student.training)
        self.assertFalse(student.quantize.training)
        self.assertEqual((report['frames'], report['encoder_frames']), (8, 6))
        self.assertIn('flow_terminal', report['per_domain'])
        self.assertIn('encoder_index_agreement', report['summary'])
        self.assertEqual(len(preview), 4)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'codec.pt'
            export_codec_student(student, teacher, path, training_steps=1, report=report)
            saved = torch.load(path, weights_only=False)
            self.assertEqual(saved['stage'], 'distilled_vqgan_codec')
            self.assertEqual(saved['latent_normalization'], {'mean': teacher.latent_mean.flatten().tolist(),
                                                           'std': teacher.latent_std.flatten().tolist()})
            self.assertEqual(saved['quality_gate']['passed'], report['passed'])
            self.assertTrue(saved['fixed_codebook'])
            self.assertTrue(saved['frozen'])
            torch.testing.assert_close(saved['model']['quantize.embedding.weight'],
                                       teacher.codec.quantize.embedding.weight, rtol=0, atol=0)
        self.assertIsNotNone(report['per_domain']['base_rgb']['encoder']['ssim_mean'])
        self.assertIsNotNone(report['per_domain']['flow_terminal']['decoder']['ssim_mean'])
        self.assertIsNone(report['per_domain']['flow_terminal']['encoder']['ssim_mean'])
        self.assertFalse(report['per_domain']['flow_terminal']['encoder_applicable'])
        with self.assertRaisesRegex(ValueError, 'held-out'):
            evaluate_codec_student(student, teacher.codec,
                prepare_codec_targets(teacher, renderer_buffer('train'), batch_size=2))

    def test_actual_short_training_resume_keeps_optimizer_rng_exact_and_exports_selected_best(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = teacher175_vq_config(codec_config(), vjepa_width=32, vjepa_heads=4,
                vjepa_encoder_depth=1, vjepa_predictor_depth=1, vjepa_readout_mlp_ratio=1,
                flow_width=16, flow_depth=1, flow_heads=4, flow_steps=1,
                flow_adapter_width=0, flow_adapter_depth=0, flow_adapter_heads=0)
            world = NeuralWorldModel(config)
            source = teacher_flow()
            world.flow_renderer.load_codec({'codec_config': source.codec.get_config(), 'model': source.codec.state_dict(),
                'latent_normalization': {'mean': source.latent_mean.flatten().tolist(), 'std': source.latent_std.flatten().tolist()}})
            checkpoint = root/'world.pt'
            torch.save({'stage': 'world', 'config': asdict(world.config), 'model': world.state_dict(), 'training_steps': 7}, checkpoint)
            for split in ('train', 'val'):
                torch.save(renderer_buffer(split, _digest(checkpoint)), root/f'teacher_{split}_buffer.pt')
            def args(name):
                return SimpleNamespace(teacher_world=checkpoint, renderer_data_buffers=root, run_dir=root/name,
                    device='cpu', steps=2, batch=4, validate_every=1, preview_every=1, save_every=1,
                    required_passes=99, seed=173, lr=1e-3, target_ssim=1., target_mae=0.)
            CodecDistillation(args('full')).run()
            calls = 0
            def interrupted(*values, **options):
                nonlocal calls
                calls += 1
                if calls == 2: raise RuntimeError('Controlled interrupt')
                return codec_distillation_loss(*values, **options)
            with patch('turkeyworld.distill_codec.codec_distillation_loss', side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, 'Controlled interrupt'):
                    CodecDistillation(args('resume')).run()
            CodecDistillation(args('resume')).run()
            load = lambda name: torch.load(root/name/'checkpoints/codec_student_last.pt', weights_only=False)
            full, resumed = load('full'), load('resume')
            for key in ('model', 'optimizer', 'rng_torch', 'rng_python'):
                assert_tree_equal(self, full[key], resumed[key])
            self.assertTrue(np.array_equal(full['rng_numpy'][1], resumed['rng_numpy'][1]))
            artifact = torch.load(root/'resume/checkpoints/codec_student.pt', weights_only=False)
            self.assertEqual(artifact['training_steps'], resumed['best']['step'])
            self.assertEqual(artifact['quality_gate']['summary'], resumed['best']['report']['summary'])
            self.assertEqual(artifact['parameters'], 997539)
            with patch('turkeyworld.distill_codec.codec_distillation_loss', side_effect=AssertionError('No extra update')):
                CodecDistillation(args('resume')).run()
            # A larger budget extends the existing state; no schedule/model/
            # optimizer/random-state reset or manual checkpoint rewrite.
            extended = args('resume'); extended.steps = 3
            CodecDistillation(extended).run()
            reference = args('full3'); reference.steps = 3
            CodecDistillation(reference).run()
            extension, uninterrupted = load('resume'), load('full3')
            self.assertEqual(extension['initial_budget'], 2)
            self.assertEqual(extension['step'], 3)
            self.assertEqual(len(extension['budget_history']), 1)
            self.assertEqual({key: extension['budget_history'][0][key]
                for key in ('from_steps', 'to_steps', 'at_optimizer_step')},
                {'from_steps': 2, 'to_steps': 3, 'at_optimizer_step': 2})
            for key in ('model', 'optimizer', 'rng_torch', 'rng_python'):
                assert_tree_equal(self, extension[key], uninterrupted[key])
            self.assertTrue(np.array_equal(extension['rng_numpy'][1], uninterrupted['rng_numpy'][1]))
            for change in ('decrease', 'learning_rate'):
                invalid = args('resume')
                invalid.steps = 2 if change == 'decrease' else 4
                if change == 'learning_rate': invalid.lr *= .5
                with self.assertRaisesRegex(ValueError, 'resume configuration'):
                    CodecDistillation(invalid).run()


if __name__ == '__main__': unittest.main()
