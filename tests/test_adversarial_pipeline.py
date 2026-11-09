"""Real final Euler samples train G/D and resume critic state in the pipeline."""
from argparse import Namespace
import contextlib
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Optional PyTorch dependency is not installed')
class AdversarialPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def arguments(self, root):
        return Namespace(run_dir=root, device='cpu', seed=173, teacher_steps=1,
            student_steps=2, temporal_updates=1, reflow_pairs=1, batch=1, lr=1e-4,
            log_every=1, save_every=1, validate_every=100, quality_endpoint_batch=1,
            adversarial=True, adversarial_weight=.02, adversarial_feature_weight=.05,
            adversarial_warmup=0, adversarial_ramp=1, adversarial_r1_gamma=1.,
            adversarial_r1_interval=2, adversarial_lr=1e-4)

    def setup_data(self, root):
        from turkeyworld.prepare_training_data import prepare
        from turkeyworld.training_pipeline import MemoryData
        prepare(root/'data', rgb_train=1, rgb_val=1, clip_frames=2, size=128,
                long_train=0, long_val=0, seed=341)
        return (MemoryData(root/'data/rgb/train', torch.device('cpu')),
                MemoryData(root/'data/rgb/val', torch.device('cpu')))

    def student(self):
        from turkeyworld.models import NeuralWorldModel, WorldModelConfig
        return NeuralWorldModel(WorldModelConfig(kind='student', image_size=128,
            endless=True, renderer_mode='rectified_flow', flow_steps=1,
            flow_width=32, flow_depth=1, flow_heads=4))

    def test_real_euler_endpoint_receives_gan_gradient_and_critic_restores_exactly(self):
        from turkeyworld.training_pipeline import Experiment
        torch.manual_seed(173)
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            args = self.arguments(root)
            data, validation = self.setup_data(root)
            source_before = data.data['images'].copy()
            experiment = Experiment(args)
            experiment.state['teacher_fingerprint'] = 'teacher-A'
            model = self.student()
            original_head = model.flow_renderer.output.weight.detach().clone()
            sample = model.flow_renderer.sample
            generated = []

            def retain_final_gradient(*values, **options):
                image = sample(*values, **options)
                if torch.is_grad_enabled() and image.requires_grad:
                    image.retain_grad()
                    generated.append(image)
                return image

            with patch.object(model.flow_renderer, 'sample', side_effect=retain_final_gradient):
                experiment.joint(model, 'student_adversarial', 2, data, validation)
            self.assertEqual(len(generated), 2)
            self.assertTrue(all(image.grad is not None and image.grad.abs().sum() > 0 for image in generated))
            self.assertGreater(float((model.flow_renderer.output.weight.detach()-original_head).abs().sum()), 0.)
            controller = experiment._adversarial_trainer
            self.assertEqual(controller.updates, 2)
            self.assertTrue(all(parameter.grad is None for parameter in controller.discriminator.parameters()))
            self.assertFalse(controller.discriminator.training)
            self.assertFalse(any(parameter.requires_grad for parameter in controller.discriminator.parameters()))
            np.testing.assert_array_equal(data.data['images'], source_before)
            records = [json.loads(line) for line in (root/'metrics.jsonl').read_text().splitlines()]
            updates = [record for record in records if 'train_loss' in record]
            self.assertEqual(len(updates), 2)
            self.assertGreater(updates[-1]['adversarial_r1'], 0.)
            self.assertEqual(updates[-1]['adversarial_step'], 2)
            checkpoint = torch.load(root/'checkpoints/student_adversarial.pt', map_location='cpu', weights_only=False)
            self.assertEqual(checkpoint['adversarial_training']['updates'], 2)
            restored = Experiment(args)
            restored.state['teacher_fingerprint'] = 'teacher-A'
            new_model = self.student()
            optimizer = torch.optim.AdamW(new_model.parameters(), lr=args.lr)
            self.assertEqual(restored.restore(new_model, optimizer, 'student_adversarial'), 2)
            rng_before = torch.random.get_rng_state().clone()
            restored._setup_adversarial('student_adversarial')
            torch.testing.assert_close(torch.random.get_rng_state(), rng_before)
            self.assertEqual(restored._adversarial_trainer.updates, 2)
            for name, value in controller.discriminator.state_dict().items():
                torch.testing.assert_close(value, restored._adversarial_trainer.discriminator.state_dict()[name], rtol=0, atol=0)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, new_model.state_dict()[name], rtol=0, atol=0)

    def test_changed_teacher_restarts_student_critic_and_its_ramp(self):
        from turkeyworld.training_pipeline import Experiment
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            args = self.arguments(root)
            data, validation = self.setup_data(root)
            first = Experiment(args)
            first.state['teacher_fingerprint'] = 'teacher-A'
            first.joint(self.student(), 'student_adversarial', 1, data, validation)
            self.assertEqual(first._adversarial_trainer.updates, 1)
            changed = Experiment(args)
            changed.state['teacher_fingerprint'] = 'teacher-B'
            model = self.student()
            original = {name: value.clone() for name, value in model.state_dict().items()}
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
            self.assertEqual(changed.restore(model, optimizer, 'student_adversarial'), 0)
            changed._setup_adversarial('student_adversarial')
            self.assertEqual(changed._adversarial_trainer.updates, 0)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, original[name], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
