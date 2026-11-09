"""Teacher growth preserves trained outputs and adds genuine trainable capacity."""
from dataclasses import asdict
import hashlib
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Optional PyTorch dependency is not installed')
class TeacherExpansionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def small_teacher(self):
        from turkeyworld.models import NeuralWorldModel, WorldModelConfig
        model = NeuralWorldModel(WorldModelConfig(kind='teacher', image_size=128, endless=True,
            renderer_mode='flow_matching', flow_steps=2, flow_width=32, flow_depth=1, flow_heads=4))
        # A nonzero trained-looking velocity makes identity checks meaningful.
        with torch.no_grad():
            model.flow_renderer.output.weight.normal_(std=.01)
            model.flow_renderer.output.bias.fill_(.02)
        return model

    def test_existing_weights_velocity_and_complete_samples_are_exactly_preserved(self):
        from turkeyworld.engine import Game
        from turkeyworld.model_expansion import expand_teacher_model
        original = self.small_teacher().eval()
        expanded, report = expand_teacher_model(original, adapter_width=32, adapter_depth=2, adapter_heads=4)
        game = Game(endless=True)
        state = torch.tensor(game.state_vector())[None]
        grid = torch.tensor(game.map, dtype=torch.long)[None]
        noisy = torch.randn(1, 3, 128, 128)
        base = torch.rand_like(noisy)
        instant = torch.tensor([.37])
        with torch.inference_mode():
            before = original.flow_renderer.velocity(noisy, instant, base, state)
            after = expanded.flow_renderer.velocity(noisy, instant, base, state)
            self.assertGreater(float(before.abs().sum()), 0.)
            torch.testing.assert_close(before, after, rtol=0, atol=0)
            source_features = original.encode_map(grid)
            grown_features = expanded.encode_map(grid)
            source_rgb = original.render(state, torch.zeros(1, 128), source_features)
            grown_rgb = expanded.render(state, torch.zeros(1, 128), grown_features)
            torch.testing.assert_close(source_rgb, grown_rgb, rtol=0, atol=0)
        self.assertTrue(report['initial_function_preserved'])
        self.assertTrue(report['all_original_tensors_exact'])
        self.assertEqual(report['generator_parameters_after']-report['generator_parameters_before'],
                         report['added_parameters'])

    def test_zero_adapter_learns_and_attention_receives_gradients(self):
        from turkeyworld.engine import Game
        from turkeyworld.model_expansion import expand_teacher_model
        expanded, _ = expand_teacher_model(self.small_teacher(), adapter_width=32, adapter_depth=2, adapter_heads=4)
        expanded.requires_grad_(False)
        adapter = expanded.flow_renderer.residual_adapter
        adapter.requires_grad_(True)
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=.01, weight_decay=0)
        state = torch.tensor(Game(endless=True).state_vector())[None]
        noisy, base = torch.randn(1, 3, 128, 128), torch.rand(1, 3, 128, 128)
        target = torch.randn_like(noisy)
        saw_attention_gradient = False
        for _ in range(4):
            optimizer.zero_grad(set_to_none=True)
            residual = adapter(noisy, torch.tensor([.5]), base, state)
            loss = (residual-target).square().mean()
            loss.backward()
            gradient = adapter.blocks[0].attention.in_proj_weight.grad
            saw_attention_gradient |= bool(gradient is not None and gradient.abs().sum() > 0)
            self.assertTrue(all(parameter.grad is not None for parameter in adapter.parameters()))
            optimizer.step()
        self.assertTrue(saw_attention_gradient)
        self.assertGreater(float(adapter.output.weight.detach().abs().sum()), 0.)

    def test_production_profile_has_about_75_million_real_generator_parameters(self):
        from turkeyworld.models import NeuralWorldModel, WorldModelConfig
        with torch.device('meta'):
            model = NeuralWorldModel(WorldModelConfig(kind='teacher', image_size=128, endless=True,
                renderer_mode='flow_matching', flow_steps=12, flow_adapter_width=512,
                flow_adapter_depth=12, flow_adapter_heads=8))
        total = sum(parameter.numel() for parameter in model.parameters())
        residual = sum(parameter.numel() for parameter in model.flow_renderer.residual_adapter.parameters())
        self.assertEqual(total, 75_204_173)
        self.assertEqual(residual, 58_370_752)

    def test_migration_retains_progress_preserves_source_and_resets_unnamed_optimizer(self):
        from turkeyworld.model_expansion import expand_teacher_checkpoint
        from turkeyworld.models import load_world_checkpoint
        model = self.small_teacher()
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory)/'trained.pt', Path(directory)/'grown.pt'
            torch.save({'stage': 'world', 'config': asdict(model.config), 'model': model.state_dict(),
                        'training_steps': 17250, 'optimizer': {'state': {}, 'param_groups': []},
                        'teacher_quality_gate': {'consecutive_passes': 2}}, source)
            before = hashlib.sha256(source.read_bytes()).hexdigest()
            report = expand_teacher_checkpoint(source, output, adapter_width=32, adapter_depth=2, adapter_heads=4)
            loaded, payload = load_world_checkpoint(output)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)
            self.assertEqual(payload['training_steps'], 17250)
            self.assertEqual(report['source_training_steps'], 17250)
            self.assertEqual(report['source_checkpoint_sha256'], before)
            self.assertNotIn('optimizer', payload)
            self.assertNotIn('teacher_quality_gate', payload)
            self.assertEqual(payload['source_teacher_quality_gate']['consecutive_passes'], 2)
            self.assertEqual(loaded.config.flow_adapter_width, 32)
            self.assertEqual(loaded.config.flow_adapter_depth, 2)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, loaded.state_dict()[name], rtol=0, atol=0)
            with self.assertRaises(FileExistsError):
                expand_teacher_checkpoint(source, output, adapter_width=32, adapter_depth=2, adapter_heads=4)

    def test_student_defaults_unchanged_and_duplicate_growth_is_rejected(self):
        from turkeyworld.model_expansion import expand_teacher_model
        from turkeyworld.models import NeuralWorldModel, WorldModelConfig
        student = NeuralWorldModel(WorldModelConfig(kind='student', renderer_mode='rectified_flow', image_size=128))
        self.assertEqual(student.flow_renderer.width, 128)
        self.assertEqual(len(student.flow_renderer.blocks), 3)
        self.assertFalse(hasattr(student.flow_renderer, 'residual_adapter'))
        with self.assertRaises(ValueError):
            expand_teacher_model(student, adapter_width=32, adapter_depth=1, adapter_heads=4)
        expanded, _ = expand_teacher_model(self.small_teacher(), adapter_width=32, adapter_depth=1, adapter_heads=4)
        with self.assertRaises(ValueError):
            expand_teacher_model(expanded, adapter_width=32, adapter_depth=1, adapter_heads=4)


if __name__ == '__main__':
    unittest.main()
