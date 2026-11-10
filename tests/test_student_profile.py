"""Selected 21.85M single-enemy student capacity, causal gameplay and real RF4 rendering."""

from turkeyworld.engine import STATE_DIM, GLOBAL_DIM, ENTITY_DIM, ENTITY_SLOTS
from dataclasses import asdict
import json
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
class StudentProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def tiny(self):
        from turkeyworld.models import NeuralWorldModel
        from turkeyworld.student_profile import student_config_22m
        config = student_config_22m(vjepa_width=32, vjepa_encoder_depth=1,
            vjepa_predictor_depth=2, vjepa_heads=4, vjepa_context_frames=2,
            flow_width=32, flow_depth=1, flow_heads=4)
        return NeuralWorldModel(config)

    def test_live_teacher_plan_is_pending_without_inventing_a_trained_student(self):
        from turkeyworld.train_500m import MassiveExperiment
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(run_dir=Path(directory), device='cpu', seed=173,
                teacher_steps=1, student_steps=1, temporal_updates=1, reflow_pairs=1,
                chunk=16, plateau_patience=6, plateau_delta=.005, teacher_until_quality=True)
            experiment = MassiveExperiment(args)
            self.assertEqual(experiment.state['student_plan']['status'], 'pending')
            self.assertEqual(experiment.state['model_parameters']['target_student_total'], 21847785)
            self.assertNotIn('student_total', experiment.state['model_parameters'])
            self.assertNotIn('training_steps', experiment.state['student_plan'])

    def test_resumed_student_plan_retains_its_actual_phase_metadata(self):
        from turkeyworld.train_500m import MassiveExperiment
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(run_dir=Path(directory), device='cpu', seed=173,
                teacher_steps=1, student_steps=1, temporal_updates=1, reflow_pairs=1,
                chunk=16, plateau_patience=6, plateau_delta=.005, teacher_until_quality=True)
            first = MassiveExperiment(args)
            plan = first.state['student_plan']
            plan.update(status='training', last_phase='student_vjepa500_distill', last_phase_step=73,
                        actual_parameters=21847785, training_steps=0)
            (Path(directory)/'student_plan.json').write_text(json.dumps(plan), encoding='utf-8')
            resumed = MassiveExperiment(args)
            self.assertEqual(resumed.state['student_plan']['status'], 'training')
            self.assertEqual(resumed.state['student_plan']['last_phase_step'], 73)
            self.assertEqual(resumed.state['student_plan']['actual_parameters'], 21847785)
            self.assertNotIn('training_steps', resumed.state['student_plan'])

    def test_selected_profile_and_all_used_component_counts_are_exact(self):
        from turkeyworld.models import NeuralWorldModel, parameter_counts
        from turkeyworld.student_profile import student_config_22m, STUDENT_22M_PARAMETERS
        config = student_config_22m()
        expected = dict(kind='student', architecture='vjepa2_ac', image_size=128, endless=True,
            vjepa_width=320, vjepa_encoder_depth=8, vjepa_predictor_depth=5, vjepa_heads=5,
            vjepa_context_frames=4, vjepa_readout_mlp_ratio=4, renderer_mode='rectified_flow',
            flow_width=192, flow_depth=4, flow_heads=4, flow_steps=4)
        for name, value in expected.items():
            self.assertEqual(getattr(config, name), value, name)
        with torch.device('meta'):
            model = NeuralWorldModel(config)
        self.assertEqual(STUDENT_22M_PARAMETERS, 21_847_785)
        self.assertEqual(parameter_counts(model)['world_total'], STUDENT_22M_PARAMETERS)
        self.assertEqual(sum(parameter.numel() for parameter in model.core.encoder.parameters()), 10_110_400)
        self.assertEqual(sum(parameter.numel() for parameter in model.core.predictor.parameters()), 6_380_160)
        self.assertEqual(sum(parameter.numel() for parameter in model.flow_renderer.parameters()), 3_027_456)
        self.assertEqual((model.core.memory_tokens, model.core.width), (264, 320))
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        self.assertFalse(any('discriminator' in name or 'ema' in name for name, _ in model.named_parameters()))

    def test_teacher_and_legacy_student_width_depth_stay_fixed_with_one_enemy(self):
        from turkeyworld.models import NeuralWorldModel, WorldModelConfig, parameter_counts
        from turkeyworld.student_profile import student_config_22m
        from turkeyworld.vjepa2_ac import VJEPA500_PROFILE
        teacher = WorldModelConfig(kind='teacher', architecture='vjepa2_ac', image_size=128,
            endless=True, renderer_mode='flow_matching', flow_steps=12,
            flow_adapter_width=512, flow_adapter_depth=12, flow_adapter_heads=8, **VJEPA500_PROFILE)
        before = asdict(teacher)
        first, second = student_config_22m(), student_config_22m()
        first.vjepa_width = 32
        self.assertEqual(second.vjepa_width, 320)
        self.assertEqual(asdict(teacher), before)
        with torch.device('meta'):
            teacher_model = NeuralWorldModel(teacher)
            legacy = NeuralWorldModel(WorldModelConfig(kind='student', architecture='vjepa2_ac', image_size=128,
                endless=True, renderer_mode='rectified_flow', flow_steps=4))
        self.assertEqual(parameter_counts(teacher_model)['world_total'], 499_012_073)
        self.assertEqual(parameter_counts(legacy)['world_total'], 11_244_393)

    def test_expanded_runner_selects_the_new_student_without_changing_legacy_default(self):
        from turkeyworld.train_250m import ExpandedExperiment
        selected = object.__new__(ExpandedExperiment).make_student_config()
        self.assertEqual(selected.kind, 'student')
        self.assertEqual((selected.vjepa_width, selected.vjepa_encoder_depth, selected.vjepa_predictor_depth),
                         (320, 8, 5))
        self.assertEqual((selected.flow_width, selected.flow_depth, selected.flow_steps), (192, 4, 4))

    def test_causal_visual_memory_carries_own_predictions_and_actions_without_future_images(self):
        from turkeyworld.engine import Game
        model = self.tiny().eval()
        game = Game(seed=811, endless=True)
        state = torch.tensor(game.state_vector())[None]
        rgb = torch.tensor(game.render(128, overlays=False)).permute(2, 0, 1)[None].float()/255
        features = model.encode_map(torch.tensor(game.map, dtype=torch.long)[None])
        with torch.inference_mode():
            video = rgb[:, None].expand(-1, 3, -1, -1, -1).clone()
            changed = video.clone()
            changed[:, 2] = torch.rand_like(changed[:, 2])
            first, second = model.core.encode_video(video), model.core.encode_video(changed)
            torch.testing.assert_close(first[:, :2], second[:, :2], rtol=0, atol=0)
            memory = model.initialize_from_image(video[:, :2], state, features)
            untouched = state.clone()
            action = torch.tensor([[.8, -.2, .15, 1.]])
            with (patch.object(Game, 'step', side_effect=AssertionError('No source transitions')),
                patch.object(Game, 'render', side_effect=AssertionError('No source renders')),
                patch.object(model.core.encoder, 'forward', side_effect=AssertionError('No future source image input'))):
                result = model.transition(state, action, features, memory)
                continued = model.transition(result['state'], action, features, result['memory'])
            torch.testing.assert_close(state, untouched, rtol=0, atol=0)
            self.assertEqual(continued['memory'].shape, (1, 132, 32))
            self.assertEqual(continued['latent_prediction'].shape, (1, 64, 32))
            blocks = continued['memory'].reshape(1, 2, 66, 32)
            torch.testing.assert_close(blocks[:, -1, 2:], continued['latent_prediction'], rtol=0, atol=0)
            torch.testing.assert_close(blocks[:, -2, 0], model.core.predictor.action_encoder(action), rtol=0, atol=0)

    def test_event_losses_and_actual_four_step_rf_render_reach_trainable_weights(self):
        from turkeyworld.engine import Game
        model = self.tiny().train()
        game = Game(seed=812, endless=True)
        state = torch.tensor(game.state_vector())[None]
        rgb = torch.tensor(game.render(128, overlays=False)).permute(2, 0, 1)[None].float()/255
        features = model.encode_map(torch.tensor(game.map, dtype=torch.long)[None])
        memory = model.initialize_from_image(rgb, state, features)
        result = model.transition(state, torch.tensor([[1., 0., .1, 1.]]), features, memory)
        with patch.object(model.flow_renderer, 'velocity', wraps=model.flow_renderer.velocity) as velocity:
            predicted = model.render(result['state'], result['features'], features)
        self.assertEqual(velocity.call_count, 4)
        self.assertEqual(predicted.shape, (1, 3, 128, 128))
        self.assertTrue(torch.isfinite(predicted).all())
        self.assertTrue(((predicted >= 0) & (predicted <= 1)).all())
        loss = torch.nn.functional.cross_entropy(result['target_logits'], torch.tensor([1]))
        loss = loss+torch.nn.functional.binary_cross_entropy_with_logits(result['attack_logits'],
            torch.ones_like(result['attack_logits']))
        loss = loss+result['latent_prediction'].square().mean()+(predicted-rgb).square().mean()
        loss.backward()
        for parameter in (model.vjepa_backend.hit_target_head[-1].weight,
                          model.vjepa_backend.attack_head[-1].weight,
                          model.core.predictor.projection.weight,
                          model.core.encoder.patch_embed.weight,
                          model.flow_renderer.output.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)


if __name__ == '__main__':
    unittest.main()
