from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from turkeyworld.engine import Game
from turkeyworld.models import NeuralWorldModel, WorldModelConfig, STATE_SCHEMA, load_world_checkpoint
from turkeyworld.rssm_migration import build_rssm_student, configure_rssm_training


def tiny_config(architecture='dreamer_rssm'):
    return WorldModelConfig(kind='student', architecture=architecture, image_size=32,
        endless=True, navigation_only=True, spatial_entities=True, spatial_combat=True,
        player_motion_mode='axis_fraction' if architecture == 'dreamer_rssm' else 'delta',
        vjepa_width=32, vjepa_encoder_depth=1, vjepa_predictor_depth=1,
        vjepa_heads=4, vjepa_patch_size=16, vjepa_readout_mlp_ratio=2,
        rssm_deter_dim=32, rssm_hidden_dim=32, rssm_stoch=4, rssm_classes=4,
        rssm_embed_dim=32, rssm_action_dim=36, rssm_blocks=4)


def inputs(model):
    game = Game(seed=0, endless=True, enemies_enabled=False)
    state = torch.tensor(game.state_vector())[None]
    grid = torch.tensor(game.map)[None]
    image = torch.tensor(game.render(size=32, overlays=False)).permute(2, 0, 1)[None].float()/255
    features = model.encode_map(grid)
    return state, image, features


class RSSMWorldBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_prior_is_deterministic_and_450_steps_never_encode_future_rgb(self):
        model = NeuralWorldModel(tiny_config()).eval()
        state, image, fmap = inputs(model)
        with torch.inference_mode():
            memory = model.initialize_from_image(image, state, fmap)
            self.assertEqual(tuple(memory.shape), (1, 48))
            before = memory.clone()
            action = state.new_tensor([[1., 0., 0., 0.]])
            first = model.transition(state, action, fmap, memory)
            again = model.transition(state, action, fmap, before)
            self.assertTrue(torch.equal(first['state'], again['state']))
            self.assertTrue(torch.equal(first['memory'], again['memory']))
            def forbidden(*args, **kwargs):
                raise AssertionError('Autonomous RSSM encoded an observation')
            model.vjepa_backend.obs_encoder.forward = forbidden
            for _ in range(450):
                result = model.transition(state, action, fmap, memory)
                state, memory = result['state'], result['memory']
                self.assertTrue(bool(torch.isfinite(state).all()))
                self.assertTrue(bool(torch.isfinite(memory).all()))
                self.assertEqual(memory.dtype, torch.float32)

    def test_next_source_rgb_changes_posterior_but_cannot_change_prior_readout(self):
        model = NeuralWorldModel(tiny_config()).eval()
        state, image, fmap = inputs(model)
        with torch.no_grad():
            memory = model.initialize_from_image(image, state, fmap)
            action = state.new_tensor([[1., .5, -.5, 0.]])
            a = model.vjepa_backend.observe_step(state, action, fmap, memory, image, sample=False)
            b = model.vjepa_backend.observe_step(state, action, fmap, memory, 1-image, sample=False)
        for key in ('state', 'memory', 'player_pass_logits', 'rssm_readout'):
            self.assertTrue(torch.equal(a['prior_result'][key], b['prior_result'][key]), key)
        self.assertFalse(torch.equal(a['rssm_posterior_logits'], b['rssm_posterior_logits']))

    def test_raw_pass_loss_trains_rssm_without_frozen_render_or_map_gradients(self):
        model = NeuralWorldModel(tiny_config())
        configure_rssm_training(model)
        state, image, fmap = inputs(model)
        memory = model.initialize_from_image(image, state, fmap)
        action = state.new_tensor([[1., 0., 0., 0.]])
        result = model.transition(state, action, fmap, memory)
        loss = F.binary_cross_entropy_with_logits(result['player_pass_logits'], torch.ones(1, 2))
        loss.backward()
        self.assertGreater(float(model.vjepa_backend.player_pass_head[-1].weight.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in model.vjepa_backend.renderer.parameters()))
        self.assertTrue(all(p.grad is None for p in model.vjepa_backend.feature_head.parameters()))
        self.assertFalse(hasattr(model.core, 'encoder'))
        self.assertFalse(hasattr(model.core, 'predictor'))

    def test_warm_start_copies_render_assets_and_roundtrip_is_explicit_rssm(self):
        teacher = NeuralWorldModel(tiny_config('vjepa2_ac')).eval()
        student, report = build_rssm_student(teacher)
        self.assertTrue(report['frozen_render_assets_bitexact'])
        source = teacher.state_dict()
        for name, tensor in student.state_dict().items():
            if name.startswith(('vjepa_backend.renderer.', 'vjepa_backend.feature_head.')):
                self.assertTrue(torch.equal(tensor, source[name]), name)
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'rssm.pt'
            torch.save({'stage': 'world', 'config': asdict(student.config),
                'state_schema': STATE_SCHEMA, 'model': student.state_dict(), 'rssm_migration': report}, path)
            loaded, metadata = load_world_checkpoint(path)
            self.assertEqual(loaded.config.architecture, 'dreamer_rssm')
            self.assertEqual(loaded.core.memory_size, 48)
            self.assertEqual(metadata['rssm_migration']['ema_target_encoder'], 'absent')


if __name__ == '__main__':
    unittest.main()
