from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from turkeyworld.sprite_scene import base_scene_state, set_sprite_scene
from turkeyworld.vjepa_validation import save_autoregressive_preview


class SpriteSceneTests(unittest.TestCase):
    def test_background_preserves_camera_weapon_and_original_state(self):
        sr = SimpleNamespace(config={'source_sprite_compositor': True})
        state = torch.rand(3,20)
        old = state.clone()
        clean = base_scene_state(state, sr, navigation_only=False)
        self.assertTrue(torch.equal(state, old))
        self.assertTrue(torch.equal(clean[:, :8], state[:, :8]))
        self.assertTrue(torch.equal(clean[:, -1], state[:, -1]))
        self.assertTrue(torch.equal(clean[:, 13], torch.zeros(3)))
        self.assertTrue(torch.equal(clean[:, 16], torch.ones(3)))
        self.assertIs(base_scene_state(state, sr, navigation_only=True), state)
        self.assertIs(base_scene_state(state, None, navigation_only=False), state)
        grid = torch.zeros(3, 16, 16, dtype=torch.long)
        set_sprite_scene(sr, state, grid, navigation_only=False)
        self.assertTrue(torch.equal(sr.predicted_sprite_state, old))
        self.assertTrue(torch.equal(sr.predicted_sprite_map, grid))
        with self.assertRaisesRegex(ValueError, 'batch'):
            set_sprite_scene(sr, state, grid[:1], navigation_only=False)
        set_sprite_scene(sr, None, None, navigation_only=True)
        self.assertIsNone(sr.predicted_sprite_map)
        self.assertFalse(sr.enemies_enabled)

    def test_preview_renders_clean_low_and_restores_map_state(self):
        class Base:
            def encode_map(self, grid):
                return grid
            def render_base(self, state, features, fmap):
                self.state = state
                return state[:,13,None,None,None].expand(-1,3,32,32)

        class SR(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = {'architecture':'rrdb_material_sprite_sr', 'source_sprite_compositor': True}
                self.enemies_enabled = False
                self.predicted_sprite_state = self.predicted_sprite_map = None
            def forward(self, low):
                self.seen = low, self.predicted_sprite_state, self.predicted_sprite_map, self.enemies_enabled
                return torch.nn.functional.interpolate(low, scale_factor=4)

        sr, base = SR(), Base()
        grid = torch.randint(0,2,(1,16,16))
        state = torch.ones(4,20)
        images = np.full((4,32,32,3), 255, np.uint8)
        preview = dict(context=images[:1], generated=images, source=images, sr_predicted_states=state,
                       sr_map=grid, sr_enemies_enabled=True, metadata=dict(step=2,generated_finite=True,caption='Predicted view'))
        with TemporaryDirectory() as temp:
            info = save_autoregressive_preview(preview, Path(temp)/'preview.png', sr_model=sr,
                                                sr_base_model=base, sr_metadata={'training_steps':7})
        self.assertEqual(info['sr_sprite_frames'], list(range(7)))
        low, own, map_batch, enabled = sr.seen
        self.assertTrue(enabled)
        self.assertTrue(torch.equal(own,state))
        self.assertTrue(torch.equal(map_batch,grid.expand(4,-1,-1)))
        self.assertTrue(torch.equal(low,torch.zeros_like(low)))
        self.assertIsNone(sr.predicted_sprite_map)
        self.assertFalse(sr.enemies_enabled)
