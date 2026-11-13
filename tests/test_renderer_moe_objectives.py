from dataclasses import dataclass
from copy import deepcopy
from types import SimpleNamespace
import json
import unittest
from unittest.mock import patch

import torch
from torch import nn

from test_renderer_enemy_objectives import batch
from test_renderer_geometry_opacity import working_model
from turkeyworld.renderer_geometry_opacity import enable_geometry_opacity
from turkeyworld.renderer_moe import configure_renderer_moe
from turkeyworld.renderer_moe_objectives import moe_conditions, token_role_loss, moe_refinement_loss
from turkeyworld.renderer_refinement import frozen_fingerprint


@dataclass
class Config:
    navigation_only: bool = False


class FeatureModel(nn.Module):
    def __init__(self):
        super().__init__(); self.weight = nn.Parameter(torch.ones(1), requires_grad=False)
        self.config = Config(); self.vjepa_backend = SimpleNamespace(config=Config()); self.calls = []
    def encode_map(self, grid): return grid.float()
    def initialize_from_image(self, image, previous, fmap):
        self.calls.append((self.config.navigation_only, previous.clone(), image.clone()))
        return previous
    def transition(self, state, action, fmap, memory):
        return {'features': state.new_full((len(state), 128), 2 if self.config.navigation_only else 1)}
    def render_geometry(self, state, fmap): return {'pose': state[:, :2]}


class RendererMoEObjectivesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_current_frozen_features_use_recorded_NAV_and_combat_profiles_and_static_zero(self):
        model = FeatureModel(); source = batch((True, True, False))
        source['navigation_only'] = torch.tensor([False, True, False])
        old_config, old_backend = model.config, model.vjepa_backend.config
        a = moe_conditions(model, source)
        self.assertEqual([row[0] for row in model.calls], [False, True])
        self.assertTrue(torch.equal(model.calls[0][1], source['previous_state'][:1]))
        self.assertTrue(torch.equal(model.calls[1][2], source['previous_rgb'][1:2]))
        self.assertTrue(torch.equal(a['features'][:, 0], torch.tensor([1., 2., 0.])))
        self.assertIs(model.config, old_config); self.assertIs(model.vjepa_backend.config, old_backend)
        poisoned = deepcopy(source); poisoned['state'][:, :2] += .1; poisoned['target'] = 1 - source['target']
        b = moe_conditions(model, poisoned)
        self.assertTrue(torch.equal(a['features'], b['features']))
        with patch.object(model, 'transition', side_effect=RuntimeError('probe')):
            with self.assertRaises(RuntimeError): moe_conditions(model, source)
        self.assertIs(model.config, old_config); self.assertIs(model.vjepa_backend.config, old_backend)

    def test_one_pixel_enemy_has_a_present_role_and_all_label_gradients_are_stopped(self):
        labels = torch.zeros(1, 3, 32, 32, requires_grad=True)
        with torch.no_grad(): labels[:, 0] = 1; labels[:, 0, 0, 0] = 0; labels[:, 1, 0, 0] = 1
        logits = torch.zeros(1, 4, 3, requires_grad=True)
        role, balance, metrics = token_role_loss(logits, logits.softmax(-1), labels)
        self.assertEqual(metrics['present_roles'], 2)
        self.assertAlmostEqual(float(role.detach()), float(torch.tensor(3.).log()), places=6)
        (role + balance).backward()
        self.assertIsNone(labels.grad)
        self.assertGreater(float(logits.grad[..., 1].abs().sum()), 0)
        self.assertGreater(metrics['label_mass'][1], 0)

    def test_real_Base_RF_experts_router_get_gradients_without_world_codec_private_or_Source_gradients(self):
        model = working_model(); enable_geometry_opacity(model); configure_renderer_moe(model)
        source = batch((False, False)); source['navigation_only'] = torch.tensor([False, True])
        for key, value in source.items():
            if value.is_floating_point(): value.requires_grad_()
        before = frozen_fingerprint(model)
        for kind in ('main', 'far', 'geometry'):
            model.zero_grad(set_to_none=True)
            with patch('turkeyworld.engine.Game.render', side_effect=AssertionError('Recorded Source only')):
                loss, metrics = moe_refinement_loss(model, source, kind=kind)
                loss.backward()
            for head in (model.base_student, model.flow_renderer):
                self.assertGreater(float(head.output_moe.output.weight.grad.abs().sum()), 0)
                self.assertGreater(float(head.output_moe.router.weight.grad.abs().sum()), 0)
            self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))
            self.assertTrue(all(v.grad is None for v in source.values() if v.is_floating_point()))
            self.assertEqual(before, frozen_fingerprint(model))
            self.assertEqual(set(metrics['renderer_moe_heads']), {'base', 'rf1'})
            json.dumps(metrics, allow_nan=False)


if __name__ == '__main__': unittest.main()
