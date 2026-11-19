from contextlib import nullcontext
from types import SimpleNamespace
from unittest import TestCase, mock

import torch

from turkeyworld.rrdb_material_sr import RRDBMaterialSR
from turkeyworld.trt_visual import enable_visual_trt
from turkeyworld.vjepa_sr_feedback import FrozenBaseRenderer


class VisualAccelerationTests(TestCase):
    def test_background_boundary_preserves_wall_materials_and_sprites(self):
        from tests.test_rrdb_material_sr import SpriteStub
        with mock.patch('turkeyworld.models.build_upscaler', return_value=SpriteStub()):
            sr = RRDBMaterialSR(sprite_config={}, channels=8, growth=4, blocks=1,
                                wall_texture_memory=True, weapon_memory=True).eval()
        sr.enemies_enabled = False
        rgb = torch.rand(1, 3, 32, 32)
        with torch.no_grad():
            before = sr.forward_with_aux(rgb)
        calls = []
        def factory(name, module, samples, outputs):
            self.assertEqual(len(samples), 1)
            self.assertEqual(len(outputs), 5)
            def execute(value):
                calls.append(name)
                return module(value)
            return execute
        rt = SimpleNamespace(sr=sr, feedback_renderer=None)
        self.assertEqual(enable_visual_trt(rt, factory), ['display_rrdb'])
        with torch.no_grad():
            after = sr.forward_with_aux(rgb)
        self.assertEqual(calls, ['display_rrdb'])
        for key in ('image', 'pre_sprite_image', 'material_logits', 'seam_logits', 'sprite_alpha'):
            self.assertTrue(torch.equal(before[key], after[key]), key)
        self.assertTrue(torch.equal(before['weapon_memory']['alpha'], after['weapon_memory']['alpha']))
        self.assertTrue(torch.equal(before['wall_memory']['rgb'], after['wall_memory']['rgb']))

    def test_cached_map_reencodes_when_same_grid_tensor_is_edited(self):
        frozen = FrozenBaseRenderer.__new__(FrozenBaseRenderer)
        frozen.model = mock.Mock()
        frozen.model.encode_map.side_effect = lambda grid: grid.float() * 2
        frozen.model.render.side_effect = lambda state, features, fmap: fmap
        grid = torch.zeros(1,16,16, dtype=torch.long)
        state = torch.zeros(1,20)
        frozen.cache_game_map(grid)
        with mock.patch('turkeyworld.vjepa_sr_feedback.environment_profile',
                        side_effect=lambda *args, **kwargs: nullcontext()):
            self.assertTrue(torch.equal(frozen.base(state, grid, navigation_only=True), grid.float()))
            self.assertEqual(frozen.model.encode_map.call_count, 1)
            grid[:, 2, 1] = 1
            self.assertTrue(torch.equal(frozen.base(state, grid), grid.float()*2))
            self.assertEqual(frozen.model.encode_map.call_count, 2)
